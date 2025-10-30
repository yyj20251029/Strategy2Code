"""main.py

Entry point for running the Transformer experiments as described in
"Attention Is All You Need". This script loads the configuration from config.yaml,
initializes data loaders for translation and parsing tasks via DatasetLoader,
instantiates the TransformerModel, and then either trains the model and/or evaluates it
(using BLEU for translation and F1 for parsing).

Usage:
    python main.py [--config CONFIG_FILE] [--task TASK] [--mode MODE]

Arguments:
    --config: Path to the configuration YAML file (default: config.yaml).
    --task: Which task to run: "translation", "parsing", or "both" (default: "translation").
    --mode: Which mode to run: "train", "eval", or "both" (default: "both").

All hyperparameters and settings are derived from config.yaml.
"""

import argparse
import logging
import os
import sys
from typing import Dict

import yaml

# Import custom modules (make sure these files are present in the same directory or PYTHONPATH)
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

def main() -> None:
    # Parse command-line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Transformer Experiment Entry Point"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["translation", "parsing", "both"],
        default="translation",
        help="Task to run: 'translation', 'parsing', or 'both' (default: translation)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Mode to run: 'train', 'eval', or 'both' (default: both)"
    )
    args = parser.parse_args()

    # Load configuration from the YAML file
    config_path: str = args.config
    if not os.path.exists(config_path):
        logging.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config: Dict = yaml.safe_load(config_file)
    except Exception as e:
        logging.error("Failed to load configuration: %s", e)
        sys.exit(1)
    logging.info("Configuration loaded from %s", config_path)

    # Initialize the DatasetLoader using configuration dictionary.
    dataset_loader: DatasetLoader = DatasetLoader(config)

    # Load training and validation DataLoader objects for both tasks if available.
    translation_train_loader = None
    translation_valid_loader = None
    parsing_train_loader = None
    parsing_valid_loader = None

    if args.task in ("translation", "both"):
        logging.info("Loading translation data...")
        translation_train_loader, translation_valid_loader = dataset_loader.load_translation_data()
        logging.info("Translation data loaded: training batches = %d, validation batches = %d",
                     len(translation_train_loader), len(translation_valid_loader))

    if args.task in ("parsing", "both"):
        logging.info("Loading parsing data...")
        parsing_train_loader, parsing_valid_loader = dataset_loader.load_parsing_data()
        logging.info("Parsing data loaded: training batches = %d, validation batches = %d",
                     len(parsing_train_loader), len(parsing_valid_loader))

    # Instantiate the TransformerModel with model hyperparameters from config.
    model_params: Dict = config.get("model", {})
    transformer_model: TransformerModel = TransformerModel(model_params)
    logging.info("TransformerModel instantiated with parameters: %s", model_params)

    # Decide on the training data loader based on the chosen task.
    train_loader = None
    if args.task == "translation":
        if translation_train_loader is None:
            logging.error("Translation training data loader not available.")
            sys.exit(1)
        train_loader = translation_train_loader
    elif args.task == "parsing":
        if parsing_train_loader is None:
            logging.error("Parsing training data loader not available.")
            sys.exit(1)
        train_loader = parsing_train_loader
    elif args.task == "both":
        # Default to translation training data when both tasks are selected.
        train_loader = translation_train_loader if translation_train_loader is not None else parsing_train_loader

    # If training mode is enabled, instantiate Trainer and run training.
    if args.mode in ("train", "both"):
        logging.info("Starting training mode...")
        trainer: Trainer = Trainer(transformer_model, train_loader, config)
        trainer.train()
        logging.info("Training completed.")

    # If evaluation mode is enabled, run evaluation routines.
    if args.mode in ("eval", "both"):
        # Evaluate translation if validation loader exists.
        if args.task in ("translation", "both") and translation_valid_loader is not None:
            logging.info("Starting translation evaluation...")
            evaluator_translation: Evaluation = Evaluation(transformer_model, translation_valid_loader, config)
            translation_metrics: Dict = evaluator_translation.evaluate_translation()
            logging.info("Translation Evaluation Metrics: %s", translation_metrics)
            print("Translation Evaluation Metrics:")
            for metric, value in translation_metrics.items():
                print(f"{metric}: {value:.2f}")
        else:
            logging.info("Translation evaluation skipped due to missing translation validation data.")

        # Evaluate parsing if validation loader exists.
        if args.task in ("parsing", "both") and parsing_valid_loader is not None:
            logging.info("Starting parsing evaluation...")
            evaluator_parsing: Evaluation = Evaluation(transformer_model, parsing_valid_loader, config)
            parsing_metrics: Dict = evaluator_parsing.evaluate_parsing()
            logging.info("Parsing Evaluation Metrics: %s", parsing_metrics)
            print("\nParsing Evaluation Metrics:")
            for metric, value in parsing_metrics.items():
                print(f"{metric}: {value:.2f}")
        else:
            logging.info("Parsing evaluation skipped due to missing parsing validation data.")

if __name__ == "__main__":
    # Configure logging to output INFO level messages.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    main()
