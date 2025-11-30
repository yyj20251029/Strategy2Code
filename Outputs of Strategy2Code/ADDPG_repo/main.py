## main.py

import os
import sys
import logging
import yaml
import pandas as pd

# Import classes from the project modules
from data.loader_yahoo import DataLoader
from strategy.strategy import Strategy
from backtest.backtester import Backtester
from backtest.metrics import Metrics

# Set up basic logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Main:
    """
    Main class acts as the entry point to the backtesting system.
    It loads the configuration from a YAML file, retrieves historical price data,
    generates trading signals with the Adaptive DDPG strategy, runs the backtest simulation,
    and computes key performance metrics.
    """

    def __init__(self, config_path: str):
        """
        Initialize Main with the configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path: str = config_path
        self.config: dict = self.load_config(config_path)
        logging.info("Configuration loaded successfully from %s", config_path)

    def load_config(self, config_path: str) -> dict:
        """
        Load configuration parameters from the YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If an error occurs while parsing the YAML file.
        """
        if not os.path.exists(config_path):
            logging.error("Configuration file not found at %s", config_path)
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, "r") as file:
                config_data: dict = yaml.safe_load(file)
            return config_data
        except yaml.YAMLError as exc:
            logging.error("Error while parsing the configuration file: %s", exc)
            raise

    def run(self) -> None:
        """
        Orchestrate the complete flow:
          1. Load price data via DataLoader.
          2. Generate trading signals using Strategy.
          3. Run portfolio simulation with Backtester.
          4. Compute and output performance metrics using Metrics.
        """
        try:
            # Step 1: Data Retrieval using DataLoader
            logging.info("Starting data retrieval with DataLoader...")
            data_loader = DataLoader(self.config)
            price_data: pd.DataFrame = data_loader.load_price_data()
            logging.info("Price data loaded successfully with shape: %s", price_data.shape)

            # Step 2: Signal Generation using Strategy
            logging.info("Generating trading signals using Strategy...")
            strategy = Strategy(self.config)
            signals: pd.DataFrame = strategy.generate_signals(price_data)
            logging.info("Trading signals generated successfully with shape: %s", signals.shape)

            # Step 3: Run Backtest Simulation using Backtester
            logging.info("Running backtest simulation with Backtester...")
            backtester = Backtester(self.config)
            performance_df: pd.DataFrame = backtester.run_backtest(price_data, signals)
            logging.info("Backtest completed successfully with performance DataFrame shape: %s", performance_df.shape)

            # Step 4: Compute Performance Metrics using Metrics
            logging.info("Computing performance metrics using Metrics...")
            metrics_module = Metrics(self.config)
            metrics: dict = metrics_module.compute(performance_df)

            # Output the computed metrics to console
            logging.info("Performance Metrics Computed:")
            for metric_name, metric_value in metrics.items():
                logging.info("  %s: %s", metric_name, metric_value)
            print("\nFinal Performance Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

        except Exception as e:
            logging.error("An error occurred during execution: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    CONFIG_FILE_PATH: str = "config.yaml"  # Path to the configuration file
    main_app = Main(CONFIG_FILE_PATH)
    main_app.run()
