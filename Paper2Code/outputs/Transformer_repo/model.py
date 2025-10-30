"""
model.py

This module implements the TransformerModel class based on the "Attention Is All You Need" paper.
It includes the full Transformer encoder–decoder architecture, along with
all submodules: multi-head attention, position-wise feed-forward networks, sinusoidal positional encoding,
residual connections, and layer normalization. The forward() method supports training
(with teacher forcing) and the generate() method supports beam search decoding.

This code strictly adheres to the design and configuration provided in config.yaml.

Dependencies:
    - torch==1.9.0
    - numpy==1.21.0

Author: Your Name
Date: YYYY-MM-DD
"""

import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.

    Adds positional information to token embeddings using sine and cosine functions.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Args:
            d_model: The dimension of the embeddings.
            dropout: Dropout rate applied after adding positional encodings.
            max_len: Maximum length of the sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape: (max_len, d_model)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of the same shape with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module with scaled dot-product attention."""

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Dimensionality of the model.
            num_heads: Number of attention heads.
            d_k: Dimensionality of queries and keys for each head.
            d_v: Dimensionality of values for each head.
            dropout: Dropout rate to apply after attention softmax.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear projections for queries, keys, values.
        self.linear_q = nn.Linear(d_model, num_heads * d_k)
        self.linear_k = nn.Linear(d_model, num_heads * d_k)
        self.linear_v = nn.Linear(d_model, num_heads * d_v)
        
        # Final linear projection
        self.linear_out = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute multi-head attention.

        Args:
            query: Tensor of shape (batch_size, len_q, d_model)
            key:   Tensor of shape (batch_size, len_k, d_model)
            value: Tensor of shape (batch_size, len_v, d_model) with len_k == len_v.
            mask:  Optional tensor of shape (batch_size, 1, len_q, len_k), where 0 indicates positions to mask.
                 
        Returns:
            Tensor of shape (batch_size, len_q, d_model) after applying multi-head attention.
        """
        batch_size = query.size(0)
        len_q = query.size(1)

        # Linear projections
        q = self.linear_q(query)  # (batch_size, len_q, num_heads*d_k)
        k = self.linear_k(key)    # (batch_size, len_k, num_heads*d_k)
        v = self.linear_v(value)  # (batch_size, len_v, num_heads*d_v)

        # Reshape and transpose to get dimensions (batch_size, num_heads, seq_len, d_k or d_v)
        q = q.view(batch_size, len_q, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, len_q, d_k)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)       # (batch_size, num_heads, len_k, d_k)
        v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)       # (batch_size, num_heads, len_v, d_v)

        # Compute scaled dot-product attention
        # scores shape: (batch_size, num_heads, len_q, len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask expected shape: (batch_size, 1, len_q, len_k) or broadcastable to that shape
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # attention output shape: (batch_size, num_heads, len_q, d_v)
        output = torch.matmul(attn, v)

        # Concatenate heads: shape -> (batch_size, len_q, num_heads*d_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, self.num_heads * self.d_v)

        # Final linear projection
        output = self.linear_out(output)
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Input and output dimension.
            d_ff: Inner-layer dimension.
            dropout: Dropout rate applied between layers.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) after applying feed-forward network.
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def subsequent_mask(size: int) -> Tensor:
    """Generate a mask to hide future positions (used in the decoder).
    
    Args:
        size: int, the sequence length.
    Returns:
        A tensor of shape (1, size, size) with 1's in allowed positions and 0's elsewhere.
    """
    # Lower triangular matrix
    attn_shape = (1, size, size)
    subsequent_mask = torch.tril(torch.ones(attn_shape, dtype=torch.uint8))
    return subsequent_mask  # (1, size, size)


class EncoderLayer(nn.Module):
    """Single layer of the Transformer encoder."""

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Dimensionality of model.
            num_heads: Number of heads in multi-head attention.
            d_k: Dimensionality of each head's queries/keys.
            d_v: Dimensionality of each head's values.
            d_ff: Dimensionality of feed-forward network inner layer.
            dropout: Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, src_len, d_model)
            mask: Optional mask tensor for self-attention.
        Returns:
            Output tensor of the same shape after applying self-attention and feed-forward network.
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """Single layer of the Transformer decoder."""

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: Dimensionality of model.
            num_heads: Number of heads in multi-head attention.
            d_k: Dimensionality of each head's queries/keys.
            d_v: Dimensionality of each head's values.
            d_ff: Dimensionality of feed-forward network inner layer.
            dropout: Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        enc_output: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Target tensor of shape (batch_size, tgt_len, d_model)
            enc_output: Encoder output tensor of shape (batch_size, src_len, d_model)
            tgt_mask: Mask for target self-attention.
            memory_mask: Optional mask for encoder-decoder attention.
        Returns:
            Tensor of shape (batch_size, tgt_len, d_model) after applying decoder computations.
        """
        # Masked self-attention for target.
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        # Encoder-decoder attention.
        enc_dec_output = self.enc_dec_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(enc_dec_output))
        # Feed-forward network.
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TransformerModel(nn.Module):
    """
    TransformerModel implements the full Transformer architecture.
    
    It includes shared token embeddings, sinusoidal positional encoding,
    a stack of encoder layers, and a stack of decoder layers using multi-head attention.
    
    Public Methods:
        - forward(x: Dict[str, Tensor], mode: str) -> Tensor: Forward pass for training (and validation).
        - generate(input: Tensor, beam_size: int, max_len: int) -> Tensor: Beam search decoding for inference.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Args:
            params: Configuration dictionary containing model hyperparameters.
                Expected parameters (with defaults if not provided):
                    - num_layers: int, number of encoder/decoder layers (default: 6)
                    - d_model: int, model dimensionality (default: 512)
                    - d_ff: int, feed-forward inner layer dimensionality (default: 2048)
                    - num_heads: int, number of attention heads (default: 8)
                    - d_k: int, dimension of keys/queries per head (default: 64)
                    - d_v: int, dimension of values per head (default: 64)
                    - dropout: float, dropout rate (default: 0.1)
                    - positional_encoding: str, type of positional encoding (default: "sinusoidal")
                    - shared_embeddings: bool, whether to share embeddings and softmax weights (default: True)
                    - vocab_size: int, vocabulary size (default: 37000)
                    - bos_token_id: int, beginning-of-sequence token id (default: 1)
                    - eos_token_id: int, end-of-sequence token id (default: 2)
                    - length_penalty: float, length penalty for beam search (default: 0.6)
        """
        super(TransformerModel, self).__init__()
        # Model hyperparameters from config with defaults.
        self.num_layers: int = params.get("num_layers", 6)
        self.d_model: int = params.get("d_model", 512)
        self.d_ff: int = params.get("d_ff", 2048)
        self.num_heads: int = params.get("num_heads", 8)
        self.d_k: int = params.get("d_k", 64)
        self.d_v: int = params.get("d_v", 64)
        self.dropout_rate: float = params.get("dropout", 0.1)
        self.pos_encoding_type: str = params.get("positional_encoding", "sinusoidal")
        self.shared_embeddings: bool = params.get("shared_embeddings", True)
        self.vocab_size: int = params.get("vocab_size", 37000)
        self.length_penalty: float = params.get("length_penalty", 0.6)

        # Special token ids (defaults can be overridden by params)
        self.bos_token_id: int = params.get("bos_token_id", 1)
        self.eos_token_id: int = params.get("eos_token_id", 2)

        # Shared token embeddings.
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        # Scaling factor for embeddings as per paper.
        self.embed_scale: float = math.sqrt(self.d_model)

        # Positional encoding (only sinusoidal implemented as default)
        if self.pos_encoding_type == "sinusoidal":
            self.pos_enc = PositionalEncoding(self.d_model, self.dropout_rate)
        else:
            raise ValueError("Only 'sinusoidal' positional encoding is implemented.")

        # Build Encoder stack.
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.num_heads, self.d_k, self.d_v, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])

        # Build Decoder stack.
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.num_heads, self.d_k, self.d_v, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])

        # Final projection layer to output logits.
        # If shared embeddings is enabled, the same weight matrix is used in the output projection.
        if self.shared_embeddings:
            self.out_linear = nn.Linear(self.d_model, self.vocab_size)
            self.out_linear.weight = self.embedding.weight
        else:
            self.out_linear = nn.Linear(self.d_model, self.vocab_size)

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode the source sequence.

        Args:
            src: Tensor of shape (batch_size, src_len)
            src_mask: Optional mask tensor for source (e.g. padding mask).

        Returns:
            Encoder output tensor of shape (batch_size, src_len, d_model).
        """
        # Embed and scale.
        src_emb = self.embedding(src) * self.embed_scale
        # Add positional encoding.
        src_emb = self.pos_enc(src_emb)
        # Pass through encoder layers.
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output

    def decode(self, tgt: Tensor, enc_output: Tensor, tgt_mask: Optional[Tensor] = None,
               memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Decode the target sequence using encoder outputs.

        Args:
            tgt: Tensor of shape (batch_size, tgt_len)
            enc_output: Encoder outputs of shape (batch_size, src_len, d_model)
            tgt_mask: Mask for target self-attention.
            memory_mask: Optional mask for encoder-decoder attention.

        Returns:
            Decoder output tensor of shape (batch_size, tgt_len, d_model).
        """
        tgt_emb = self.embedding(tgt) * self.embed_scale
        tgt_emb = self.pos_enc(tgt_emb)
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, memory_mask)
        return dec_output

    def forward(self, x: Dict[str, Tensor], mode: str = "train") -> Tensor:
        """
        Forward pass of the model used for training (with teacher forcing).

        Args:
            x: Dictionary with keys:
                "src": Tensor of shape (batch_size, src_len)
                "tgt": Tensor of shape (batch_size, tgt_len) (optional in inference mode)
            mode: Mode string ("train" or "eval").

        Returns:
            Logits tensor of shape (batch_size, tgt_len, vocab_size)
        """
        src = x.get("src")  # Required: source sequence tensor
        if src is None:
            raise ValueError("Input dictionary must include 'src' key.")
        # (Optional) Create source mask if needed. Here we assume None or handled externally.
        src_mask = None

        # Encode source
        enc_output = self.encode(src, src_mask)

        # If target is provided, use teacher forcing.
        tgt = x.get("tgt")
        if tgt is None:
            # In case no target is provided, simply return encoder output.
            return enc_output

        # Create subsequent mask for target to prevent attending to future tokens.
        batch_size, tgt_len = tgt.size()
        # Subsequent mask shape: (1, tgt_len, tgt_len)
        subsequent_mask_tensor = subsequent_mask(tgt_len).to(tgt.device)
        # Expand to batch size if necessary.
        tgt_mask = subsequent_mask_tensor  # Optionally combine with padding mask.

        # Decode target with encoder memory.
        dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask=None)
        # Compute output logits.
        logits = self.out_linear(dec_output)
        return logits

    def generate(self, src: Tensor, beam_size: int = 4, max_len: Optional[int] = None) -> Tensor:
        """
        Beam search decoding for inference.

        Args:
            src: Source sequence tensor of shape (1, src_len). (Assumes batch size 1)
            beam_size: Beam search width.
            max_len: Maximum length of the generated sequence. If None, uses src length + 50.
        
        Returns:
            Tensor of shape (generated_seq_len,) containing token ids of the best sequence.
        """
        # Encode the source.
        enc_output = self.encode(src)  # shape: (1, src_len, d_model)
        src_len = src.size(1)
        if max_len is None:
            max_len = src_len + 50

        device = src.device

        # Initialize beam with a tuple (sequence, cumulative log probability)
        beam = [{
            "seq": torch.tensor([self.bos_token_id], dtype=torch.long, device=device), 
            "log_prob": 0.0,
            "ended": False
        }]

        for _ in range(max_len):
            new_beam = []
            # For each hypothesis in the current beam.
            for hyp in beam:
                if hyp["ended"]:
                    new_beam.append(hyp)
                    continue
                # Prepare decoder input: unsqueeze to have batch dimension.
                tgt_seq = hyp["seq"].unsqueeze(0)  # shape: (1, cur_len)
                cur_len = tgt_seq.size(1)
                # Create subsequent mask for current target length.
                tgt_mask = subsequent_mask(cur_len).to(device)
                # Decode current sequence.
                dec_output = self.decode(tgt_seq, enc_output, tgt_mask=tgt_mask, memory_mask=None)
                # Get logits for last time step.
                logits = self.out_linear(dec_output)  # shape: (1, cur_len, vocab_size)
                # Consider only the last token.
                logits = logits[:, -1, :]  # shape: (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)  # shape: (1, vocab_size)
                log_probs = log_probs.squeeze(0)  # shape: (vocab_size,)

                # Get top beam_size tokens and their log probabilities.
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token_id = top_indices[i].item()
                    token_log_prob = top_log_probs[i].item()
                    new_seq = torch.cat([hyp["seq"], torch.tensor([token_id], dtype=torch.long, device=device)])
                    new_log_prob = hyp["log_prob"] + token_log_prob
                    ended = (token_id == self.eos_token_id)
                    new_beam.append({
                        "seq": new_seq,
                        "log_prob": new_log_prob,
                        "ended": ended
                    })

            # Keep top beam_size hypotheses in beam.
            # Apply length penalty: score = log_prob / (len(seq)^length_penalty)
            beam = sorted(
                new_beam, 
                key=lambda x: x["log_prob"] / (len(x["seq"]) ** self.length_penalty),
                reverse=True
            )[:beam_size]

            # If all hypotheses have ended, break.
            if all(hyp["ended"] for hyp in beam):
                break

        # Return the hypothesis with the highest score (apply length penalty).
        best_hyp = max(beam, key=lambda x: x["log_prob"] / (len(x["seq"]) ** self.length_penalty))
        return best_hyp["seq"]
