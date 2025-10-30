"""dataset_loader.py

This module implements the DatasetLoader class, which is responsible for loading and
preprocessing data for both machine translation and English constituency parsing tasks.
It reads raw text data, trains or loads tokenizers (BPE for translation and WordPiece for parsing),
builds vocabularies (with special tokens), and creates PyTorch DataLoader objects with
a dynamic batching strategy based on an approximate token count per batch.

Dependencies:
    - torch==1.9.0
    - torchtext==0.10.0
    - tokenizers==0.10.3
    - numpy==1.21.0
    - tqdm==4.62.3

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import logging
from typing import Any, Dict, List, Tuple, Callable, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers import pre_tokenizers, trainers, processors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def _read_file(file_path: str) -> List[str]:
    """Reads a file and returns a list of stripped lines.

    Args:
        file_path: Path to the text file.

    Returns:
        A list of strings, one per line.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def pad_sequences(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    """Pads a list of integer sequences to the same length.

    Args:
        sequences: List of sequences (list of ints).
        pad_id: The integer id used for padding.

    Returns:
        A tensor of shape (batch_size, max_seq_length) containing the padded sequences.
    """
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


def translation_collate_fn(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate function for translation data that pads source and target sequences.

    Args:
        batch: A list of examples, each a dict with keys "src" and "tgt".
        pad_id: Padding token id.

    Returns:
        A dictionary with padded tensors and lengths for source and target sequences.
    """
    src_seqs = [sample["src"] for sample in batch]
    tgt_seqs = [sample["tgt"] for sample in batch]
    src_padded = pad_sequences(src_seqs, pad_id)
    tgt_padded = pad_sequences(tgt_seqs, pad_id)
    src_lengths = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_seqs], dtype=torch.long)
    return {"src": src_padded, "src_lengths": src_lengths,
            "tgt": tgt_padded, "tgt_lengths": tgt_lengths}


def parsing_collate_fn(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate function for parsing data. The structure is similar to translation:
    input sentence and linearized parse tree are both padded.

    Args:
        batch: A list of examples, each a dict with keys "src" (input sentence)
               and "tgt" (linearized parse tree).
        pad_id: Padding token id.

    Returns:
        A dictionary with padded tensors and lengths for 'src' and 'tgt'.
    """
    src_seqs = [sample["src"] for sample in batch]
    tgt_seqs = [sample["tgt"] for sample in batch]
    src_padded = pad_sequences(src_seqs, pad_id)
    tgt_padded = pad_sequences(tgt_seqs, pad_id)
    src_lengths = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_seqs], dtype=torch.long)
    return {"src": src_padded, "src_lengths": src_lengths,
            "tgt": tgt_padded, "tgt_lengths": tgt_lengths}


class DynamicBatchSampler(Sampler[List[int]]):
    """A dynamic batch sampler that groups examples into batches such that the total
    token count (based on the maximum sequence lengths within the batch) does not exceed
    a specified token budget.

    Args:
        dataset: The dataset to sample from.
        token_batch_size: The approximate maximum number of tokens per batch (for both
                          source and target sequences).
        length_fn: A callable that takes an example from the dataset and returns a tuple
                   of (src_length, tgt_length).
    """

    def __init__(self, dataset: Dataset, token_batch_size: int,
                 length_fn: Callable[[Dict[str, List[int]]], Tuple[int, int]]) -> None:
        self.dataset = dataset
        self.token_batch_size = token_batch_size
        self.length_fn = length_fn
        # Precompute lengths for numerical efficiency.
        self.lengths = [self.length_fn(self.dataset[i]) for i in range(len(self.dataset))]
        # Sort indices based on the maximum of source and target lengths.
        self.sorted_indices = sorted(range(len(self.dataset)),
                                     key=lambda i: max(self.lengths[i]))

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        max_src = 0
        max_tgt = 0
        for idx in self.sorted_indices:
            src_len, tgt_len = self.lengths[idx]
            candidate_max_src = max(max_src, src_len)
            candidate_max_tgt = max(max_tgt, tgt_len)
            if ((candidate_max_src * (len(batch) + 1) <= self.token_batch_size) and
                    (candidate_max_tgt * (len(batch) + 1) <= self.token_batch_size)):
                batch.append(idx)
                max_src = candidate_max_src
                max_tgt = candidate_max_tgt
            else:
                if batch:
                    yield batch
                batch = [idx]
                max_src = src_len
                max_tgt = tgt_len
        if batch:
            yield batch

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


class TranslationDataset(Dataset):
    """Dataset for machine translation.

    Each example is a dictionary with keys:
        "src": List[int] representing the source sentence token IDs.
        "tgt": List[int] representing the target sentence token IDs.
    """

    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 tokenizer: Tokenizer) -> None:
        if len(src_texts) != len(tgt_texts):
            raise ValueError("Source and target texts must have the same number of lines.")
        self.data: List[Dict[str, List[int]]] = []
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")
        for src, tgt in zip(src_texts, tgt_texts):
            src_encoding = tokenizer.encode(src)
            tgt_encoding = tokenizer.encode(tgt)
            src_ids = [bos_id] + src_encoding.ids + [eos_id]
            tgt_ids = [bos_id] + tgt_encoding.ids + [eos_id]
            self.data.append({"src": src_ids, "tgt": tgt_ids})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]


class ParsingDataset(Dataset):
    """Dataset for English constituency parsing.

    Each line in the raw data file is expected to be tab-separated with the first part
    as the input sentence and the second part as the linearized parse tree.
    Each example is a dictionary with keys:
        "src": List[int] representing the sentence token IDs.
        "tgt": List[int] representing the linearized parse tree token IDs.
    """

    def __init__(self, raw_lines: List[str], tokenizer: Tokenizer) -> None:
        self.data: List[Dict[str, List[int]]] = []
        bos_id = tokenizer.token_to_id("<BOS>")
        eos_id = tokenizer.token_to_id("<EOS>")
        for line in raw_lines:
            parts = line.split("\t")
            if len(parts) < 2:
                # Skip lines that do not have both sentence and tree.
                continue
            sentence, tree = parts[0].strip(), parts[1].strip()
            sentence_ids = [bos_id] + tokenizer.encode(sentence).ids + [eos_id]
            tree_ids = [bos_id] + tokenizer.encode(tree).ids + [eos_id]
            self.data.append({"src": sentence_ids, "tgt": tree_ids})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]


class DatasetLoader:
    """DatasetLoader class for loading and preprocessing translation and parsing datasets.

    This class uses the tokenizers library to build/train tokenizers and torch's DataLoader
    utilities with dynamic batching to yield batches of tokenized, padded data.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the DatasetLoader with a configuration dictionary.

        Args:
            config: Configuration dictionary, typically loaded from config.yaml.
        """
        self.config = config
        # Training token batch size (approximate tokens per batch for both source and target)
        self.token_batch_size: int = config.get("training", {}).get("token_batch_size", 25000)

        # Translation dataset configuration
        translation_config = config.get("datasets", {}).get("translation", {})
        self.translation_vocab_size: int = translation_config.get("bpe_vocab_size", 37000)
        self.translation_dataset_source: str = translation_config.get("source", "WMT2014_EnglishGerman")
        self.translation_dataset_target: str = translation_config.get("target", "WMT2014_EnglishGerman")
        # Default file paths for translation data
        self.translation_train_src_path: str = os.path.join("data", "translation", "train.src")
        self.translation_train_tgt_path: str = os.path.join("data", "translation", "train.tgt")
        self.translation_valid_src_path: str = os.path.join("data", "translation", "valid.src")
        self.translation_valid_tgt_path: str = os.path.join("data", "translation", "valid.tgt")

        # Parsing dataset configuration
        parsing_config = config.get("datasets", {}).get("parsing", {})
        self.parsing_vocab_size: int = parsing_config.get("vocab_size", 16000)
        self.parsing_dataset_id: str = parsing_config.get("dataset", "WSJ_PennTreebank")
        # Default file paths for parsing data
        self.parsing_train_path: str = os.path.join("data", "parsing", "train.txt")
        self.parsing_valid_path: str = os.path.join("data", "parsing", "valid.txt")

        # Placeholders for tokenizers (to be loaded/trained later)
        self.translation_tokenizer: Tokenizer = None  # type: ignore
        self.parsing_tokenizer: Tokenizer = None  # type: ignore

    def _get_translation_tokenizer(self) -> Tokenizer:
        """Gets or trains the BPE tokenizer for the translation task.

        Returns:
            A trained Tokenizer object.
        """
        tokenizer_dir = "tokenizers"
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "translation_tokenizer.json")
        if os.path.exists(tokenizer_path):
            logging.info("Loading existing translation tokenizer from %s", tokenizer_path)
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            logging.info("Training new translation tokenizer...")
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.BpeTrainer(vocab_size=self.translation_vocab_size,
                                          special_tokens=["<PAD>", "<BOS>", "<EOS>"])
            # Read training data lines from both source and target files.
            src_lines = _read_file(self.translation_train_src_path)
            tgt_lines = _read_file(self.translation_train_tgt_path)
            combined_lines = src_lines + tgt_lines
            # Train tokenizer using an iterator over lines.
            tokenizer.train_from_iterator(tqdm(combined_lines, desc="Training translation tokenizer"),
                                          trainer=trainer)
            tokenizer.save(tokenizer_path)
            logging.info("Translation tokenizer saved to %s", tokenizer_path)
        return tokenizer

    def _get_parsing_tokenizer(self) -> Tokenizer:
        """Gets or trains the WordPiece tokenizer for the parsing task.

        Returns:
            A trained Tokenizer object.
        """
        tokenizer_dir = "tokenizers"
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "parsing_tokenizer.json")
        if os.path.exists(tokenizer_path):
            logging.info("Loading existing parsing tokenizer from %s", tokenizer_path)
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            logging.info("Training new parsing tokenizer...")
            tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordPieceTrainer(vocab_size=self.parsing_vocab_size,
                                                special_tokens=["<PAD>", "<BOS>", "<EOS>"])
            # For parsing, combine sentences and parse trees from the training file.
            raw_lines = _read_file(self.parsing_train_path)
            combined_lines = []
            for line in raw_lines:
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                sentence = parts[0].strip()
                tree = parts[1].strip()
                combined_lines.extend([sentence, tree])
            tokenizer.train_from_iterator(tqdm(combined_lines, desc="Training parsing tokenizer"),
                                          trainer=trainer)
            tokenizer.save(tokenizer_path)
            logging.info("Parsing tokenizer saved to %s", tokenizer_path)
        return tokenizer

    def load_translation_data(self) -> Tuple[DataLoader, DataLoader]:
        """Loads and preprocesses translation data, returning DataLoader objects for training and
        validation.

        Returns:
            A tuple (train_data_loader, valid_data_loader) for translation.
        """
        # Initialize or load the translation tokenizer.
        self.translation_tokenizer = self._get_translation_tokenizer()
        pad_id = self.translation_tokenizer.token_to_id("<PAD>")
        # Read raw training and validation data.
        train_src_lines = _read_file(self.translation_train_src_path)
        train_tgt_lines = _read_file(self.translation_train_tgt_path)
        valid_src_lines = _read_file(self.translation_valid_src_path)
        valid_tgt_lines = _read_file(self.translation_valid_tgt_path)

        # Create dataset instances.
        train_dataset = TranslationDataset(train_src_lines, train_tgt_lines, self.translation_tokenizer)
        valid_dataset = TranslationDataset(valid_src_lines, valid_tgt_lines, self.translation_tokenizer)

        # Define a length function that returns (src_length, tgt_length) for each sample.
        length_fn = lambda sample: (len(sample["src"]), len(sample["tgt"]))

        # Create dynamic batch samplers.
        train_batch_sampler = DynamicBatchSampler(train_dataset, self.token_batch_size, length_fn)
        valid_batch_sampler = DynamicBatchSampler(valid_dataset, self.token_batch_size, length_fn)

        # Create DataLoaders with the custom collate function.
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=lambda batch: translation_collate_fn(batch, pad_id),
            num_workers=0
        )

        valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=lambda batch: translation_collate_fn(batch, pad_id),
            num_workers=0
        )

        logging.info("Translation data loaders created with %d training batches and %d validation batches.",
                     len(train_batch_sampler), len(valid_batch_sampler))
        return train_data_loader, valid_data_loader

    def load_parsing_data(self) -> Tuple[DataLoader, DataLoader]:
        """Loads and preprocesses parsing data, returning DataLoader objects for training and validation.

        Assumes that each line in the raw parsing data file is tab-separated, with the input sentence
        and the corresponding linearized parse tree.

        Returns:
            A tuple (train_data_loader, valid_data_loader) for parsing.
        """
        # Initialize or load the parsing tokenizer.
        self.parsing_tokenizer = self._get_parsing_tokenizer()
        pad_id = self.parsing_tokenizer.token_to_id("<PAD>")
        # Read raw training and validation parsing data.
        train_lines = _read_file(self.parsing_train_path)
        valid_lines = _read_file(self.parsing_valid_path)

        # Create dataset instances.
        train_dataset = ParsingDataset(train_lines, self.parsing_tokenizer)
        valid_dataset = ParsingDataset(valid_lines, self.parsing_tokenizer)

        # Define a length function that returns (src_length, tgt_length) for each sample.
        length_fn = lambda sample: (len(sample["src"]), len(sample["tgt"]))

        # Create dynamic batch samplers.
        train_batch_sampler = DynamicBatchSampler(train_dataset, self.token_batch_size, length_fn)
        valid_batch_sampler = DynamicBatchSampler(valid_dataset, self.token_batch_size, length_fn)

        # Create DataLoaders with the custom collate function.
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=lambda batch: parsing_collate_fn(batch, pad_id),
            num_workers=0
        )

        valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_sampler=valid_batch_sampler,
            collate_fn=lambda batch: parsing_collate_fn(batch, pad_id),
            num_workers=0
        )

        logging.info("Parsing data loaders created with %d training batches and %d validation batches.",
                     len(train_batch_sampler), len(valid_batch_sampler))
        return train_data_loader, valid_data_loader
