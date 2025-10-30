"""evaluation.py

This module implements the Evaluation class for evaluating the Transformer model
on both machine translation and constituency parsing tasks. It computes the BLEU
score for translation and the F1 score (of constituent spans) for parsing by
using beam search decoding as described in "Attention Is All You Need".

Usage:
    evaluator = Evaluation(model, test_data, config)
    translation_metrics = evaluator.evaluate_translation()
    parsing_metrics = evaluator.evaluate_parsing()

Dependencies:
    - torch==1.9.0
    - torchtext==0.10.0 (for BLEU computation)
    - tqdm==4.62.3

Author: Your Name
Date: YYYY-MM-DD
"""

import math
import logging
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import DataLoader

# Attempt to import BLEU score function from torchtext.
try:
    from torchtext.data.metrics import bleu_score
except ImportError as error:
    raise ImportError(
        "Could not import 'bleu_score' from torchtext.data.metrics. "
        "Ensure that torchtext>=0.10.0 is installed."
    ) from error

logging.basicConfig(level=logging.INFO)


class Evaluation:
    """Evaluation class for computing metrics on model outputs.

    This class supports evaluation for both machine translation (via BLEU score)
    and constituency parsing (via F1 score computed from constituent spans).
    It leverages beam search decoding through the model's generate() method.

    Attributes:
        model (torch.nn.Module): The Transformer model instance.
        test_data (DataLoader): DataLoader providing test examples.
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): Device on which model computations are run.
    """

    def __init__(self, model: torch.nn.Module, test_data: DataLoader, config: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation instance.

        Args:
            model (torch.nn.Module): A fully initialized TransformerModel instance.
            test_data (DataLoader): PyTorch DataLoader supplying test data.
            config (Dict[str, Any]): Configuration dictionary (from config.yaml).
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        # Set model in evaluation mode.
        self.model.eval()
        # Get the device from the model parameters (default to CPU if not found).
        self.device = next(self.model.parameters()).device if next(self.model.parameters(), None) is not None else torch.device("cpu")
        # Default token identifiers.
        self.pad_id: int = 0
        self.bos_token_id: int = getattr(self.model, "bos_token_id", 1)
        self.eos_token_id: int = getattr(self.model, "eos_token_id", 2)

    def evaluate_translation(self) -> Dict[str, float]:
        """Evaluates machine translation quality using BLEU score.

        The method uses beam search decoding on test examples, computes translations, and
        compares them against reference translations.

        Returns:
            A dictionary with computed BLEU score, e.g., {"BLEU": bleu_value}.
        """
        logging.info("Starting translation evaluation...")
        # Extract translation inference configuration.
        inference_cfg = self.config.get("inference", {}).get("translation", {})
        beam_search_cfg = inference_cfg.get("beam_search", {})
        beam_size: int = int(beam_search_cfg.get("beam_size", 4))
        length_penalty: float = float(beam_search_cfg.get("length_penalty", 0.6))
        max_output_expr: str = str(beam_search_cfg.get("max_output_length_expr", "input_length + 50"))

        # Set the model's length penalty parameter.
        if hasattr(self.model, "length_penalty"):
            self.model.length_penalty = length_penalty

        candidate_corpus: List[List[str]] = []
        reference_corpus: List[List[str]] = []

        with torch.no_grad():
            for batch in self.test_data:
                # Expect batch to be a dictionary with keys: "src", "src_lengths", "tgt", "tgt_lengths".
                src_batch = batch.get("src")
                tgt_batch = batch.get("tgt")
                src_lengths = batch.get("src_lengths")
                if src_batch is None or tgt_batch is None or src_lengths is None:
                    logging.warning("Batch missing required keys; skipping batch.")
                    continue

                batch_size: int = src_batch.size(0)
                for i in range(batch_size):
                    # Obtain source sequence (remove padding using src_lengths).
                    src_len: int = int(src_lengths[i].item())
                    src_seq = src_batch[i, :src_len].unsqueeze(0).to(self.device)  # Shape: (1, seq_len)
                    input_length: int = src_seq.size(1)
                    # Safely evaluate maximum output length expression.
                    try:
                        max_len: int = int(eval(max_output_expr, {"__builtins__": None}, {"input_length": input_length}))
                    except Exception as eval_error:
                        logging.error("Error evaluating max_output_length_expr: %s", eval_error)
                        max_len = input_length + 50
                    # Generate translation using beam search.
                    generated_seq = self.model.generate(src_seq, beam_size=beam_size, max_len=max_len)
                    # Detokenize candidate and reference sequences.
                    candidate_tokens: List[str] = self._detokenize(generated_seq)
                    reference_tokens: List[str] = self._detokenize(tgt_batch[i, :].to(self.device))
                    candidate_corpus.append(candidate_tokens)
                    reference_corpus.append(reference_tokens)

        # Format references for bleu_score: each candidate should have a list of reference sentences.
        formatted_references: List[List[List[str]]] = [[ref] for ref in reference_corpus]
        bleu: float = bleu_score(candidate_corpus, formatted_references)
        logging.info("Translation evaluation complete. BLEU score: %.2f", bleu)
        return {"BLEU": bleu}

    def evaluate_parsing(self) -> Dict[str, float]:
        """Evaluates constituency parsing performance using F1 score.

        The method decodes parse trees via beam search, extracts constituent spans from both
        predicted and reference trees, and computes F1 based on their precision and recall.

        Returns:
            A dictionary with computed F1 score, e.g., {"F1": f1_value}.
        """
        logging.info("Starting parsing evaluation...")
        # Extract parsing inference configuration.
        inference_cfg = self.config.get("inference", {}).get("parsing", {})
        beam_search_cfg = inference_cfg.get("beam_search", {})
        beam_size: int = int(beam_search_cfg.get("beam_size", 21))
        length_penalty: float = float(beam_search_cfg.get("length_penalty", 0.3))
        max_output_expr: str = str(beam_search_cfg.get("max_output_length_expr", "input_length + 300"))

        # Set the model's length penalty parameter.
        if hasattr(self.model, "length_penalty"):
            self.model.length_penalty = length_penalty

        total_predicted: int = 0
        total_gold: int = 0
        total_correct: int = 0
        num_examples: int = 0

        with torch.no_grad():
            for batch in self.test_data:
                # Expect batch to be a dictionary with keys: "src", "src_lengths", "tgt", "tgt_lengths".
                src_batch = batch.get("src")
                tgt_batch = batch.get("tgt")
                src_lengths = batch.get("src_lengths")
                if src_batch is None or tgt_batch is None or src_lengths is None:
                    logging.warning("Batch missing required keys; skipping batch.")
                    continue

                batch_size: int = src_batch.size(0)
                for i in range(batch_size):
                    src_len: int = int(src_lengths[i].item())
                    src_seq = src_batch[i, :src_len].unsqueeze(0).to(self.device)
                    input_length: int = src_seq.size(1)
                    try:
                        max_len: int = int(eval(max_output_expr, {"__builtins__": None}, {"input_length": input_length}))
                    except Exception as eval_error:
                        logging.error("Error evaluating max_output_length_expr for parsing: %s", eval_error)
                        max_len = input_length + 300
                    # Generate parse tree using beam search.
                    generated_seq = self.model.generate(src_seq, beam_size=beam_size, max_len=max_len)
                    # Convert token ids to string representations.
                    pred_tree: str = self._detokenize_to_string(generated_seq)
                    gold_tree: str = self._detokenize_to_string(tgt_batch[i, :].to(self.device))
                    # Extract constituent spans from both trees.
                    pred_spans: Set[tuple] = self._extract_constituents(pred_tree)
                    gold_spans: Set[tuple] = self._extract_constituents(gold_tree)
                    total_predicted += len(pred_spans)
                    total_gold += len(gold_spans)
                    total_correct += len(pred_spans.intersection(gold_spans))
                    num_examples += 1

        precision: float = (total_correct / total_predicted) if total_predicted > 0 else 0.0
        recall: float = (total_correct / total_gold) if total_gold > 0 else 0.0
        f1: float = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        logging.info("Parsing evaluation complete over %d examples. F1 score: %.2f", num_examples, f1)
        return {"F1": f1}

    def _detokenize(self, token_tensor: torch.Tensor) -> List[str]:
        """Converts a tensor of token ids to a list of token strings,
        filtering out special tokens.

        Args:
            token_tensor (torch.Tensor): Tensor containing token ids.

        Returns:
            List[str]: List of token strings.
        """
        token_ids: List[int] = token_tensor.tolist()
        # Filter out pad, BOS, and EOS tokens.
        filtered_ids: List[int] = [token for token in token_ids if token not in {self.pad_id, self.bos_token_id, self.eos_token_id}]
        # Without a vocabulary mapping, convert token ids to strings.
        return [str(token) for token in filtered_ids]

    def _detokenize_to_string(self, token_tensor: torch.Tensor) -> str:
        """Converts a tensor of token ids into a space-separated string.

        Args:
            token_tensor (torch.Tensor): Tensor containing token ids.

        Returns:
            str: Detokenized string.
        """
        tokens: List[str] = self._detokenize(token_tensor)
        return " ".join(tokens)

    def _extract_constituents(self, tree_str: str) -> Set[tuple]:
        """Extracts constituent spans from a bracketed parse tree string.

        The method recursively parses the bracketed tree and returns a set of spans,
        each represented as a tuple (start_index, end_index) corresponding to the yield of words.

        Args:
            tree_str (str): Linearized parse tree string (e.g., "(S (NP John) (VP (V saw) (NP Mary)))").

        Returns:
            Set[tuple]: A set of constituent spans.
        """
        tokens: List[str] = tree_str.replace("(", " ( ").replace(")", " ) ").split()

        def recursive_parse(tokens_list: List[str], index: int, current_index: int) -> (Set[tuple], int, int):
            """Recursively parses tokens to extract constituent spans.

            Args:
                tokens_list (List[str]): Tokenized parse tree.
                index (int): Current index in tokens_list.
                current_index (int): Current word index in the yield.

            Returns:
                Tuple containing:
                    - A set of spans.
                    - Updated index in tokens_list.
                    - Updated current_index after processing leaves.
            """
            assert tokens_list[index] == "(", f"Expected '(' at token index {index}."
            index += 1  # Skip '('
            # Skip the label token.
            index += 1
            start: int = current_index
            spans_set: Set[tuple] = set()
            while tokens_list[index] != ")":
                if tokens_list[index] == "(":
                    child_spans, index, current_index = recursive_parse(tokens_list, index, current_index)
                    spans_set = spans_set.union(child_spans)
                else:
                    # Process a leaf (word) token.
                    current_index += 1
                    index += 1
            index += 1  # Skip ')'
            end: int = current_index
            if end - start > 1:
                spans_set.add((start, end))
            return spans_set, index, current_index

        try:
            spans, _, _ = recursive_parse(tokens, 0, 0)
        except Exception as error:
            logging.error("Error parsing tree string: %s. Returning empty constituent set.", error)
            spans = set()
        return spans
