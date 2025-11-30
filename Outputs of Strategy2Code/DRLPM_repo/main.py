# main.py

import logging
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from data.loader_yahoo import DataLoader
from strategy.strategy import Strategy
from backtest.backtester import Backtester
from backtest.metrics import Metrics


def load_config(config_path: str) -> dict:
    """
    Load and return configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: The configuration dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file is empty.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    if not config:
        raise ValueError("Configuration file is empty.")
    return config


def main() -> None:
    """
    Main entry point for the quantitative trading backtest system.
    
    This function coordinates:
      - Loading configuration from config.yaml.
      - Downloading and cleaning price data via DataLoader.
      - Generating trading signals via Strategy.
      - Running the backtest simulation via Backtester.
      - Computing performance metrics via Metrics.
      - Outputting the results and displaying a portfolio value evolution plot.
    """
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("Main")
    
    # 1. Read Configuration
    config_file: str = "config.yaml"
    try:
        logger.info("Loading configuration from '%s'", config_file)
        config: dict = load_config(config_file)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        sys.exit(1)
    
    # 2. Data Acquisition
    try:
        logger.info("Initializing DataLoader for data acquisition.")
        data_loader = DataLoader(config)
        price_df: pd.DataFrame = data_loader.load_price_data()
        logger.info("Price data loaded successfully. Data shape: %s", price_df.shape)
    except Exception as e:
        logger.error("Error in data acquisition: %s", e)
        sys.exit(1)
    
    # 3. Signal Generation (Strategy)
    try:
        logger.info("Initializing Strategy module and generating signals.")
        strategy = Strategy(config)
        signals_df: pd.DataFrame = strategy.generate_signals(price_df)
        logger.info("Signals generated successfully. Signals shape: %s", signals_df.shape)
    except Exception as e:
        logger.error("Error in signal generation: %s", e)
        sys.exit(1)
    
    # 4. Backtesting
    try:
        logger.info("Initializing Backtester and running backtest simulation.")
        backtester = Backtester(config)
        performance_df: pd.DataFrame = backtester.run_backtest(price_df, signals_df)
        logger.info("Backtest completed successfully. Performance data shape: %s", performance_df.shape)
    except Exception as e:
        logger.error("Error during backtesting: %s", e)
        sys.exit(1)
    
    # 5. Performance Metrics Calculation
    try:
        logger.info("Initializing Metrics module and computing performance metrics.")
        metrics = Metrics(config)
        performance_metrics: dict = metrics.compute(performance_df)
        logger.info("Performance metrics computed successfully.")
    except Exception as e:
        logger.error("Error in performance metrics calculation: %s", e)
        sys.exit(1)
    
    # 6. Output Results
    try:
        print("\nPerformance Metrics:")
        for key, value in performance_metrics.items():
            print(f"{key}: {value:.4f}")
    except Exception as e:
        logger.error("Error printing performance metrics: %s", e)
    
    # 7. Optional Visualization: Plot Portfolio Value Evolution
    try:
        logger.info("Generating portfolio value evolution plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        performance_df["portfolio_value"].plot(ax=ax)
        ax.set_title("Portfolio Value Evolution")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error("Error generating plot: %s", e)


if __name__ == "__main__":
    main()
