## main.py

import logging
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from data.loader_yahoo import DataLoader
from strategy.strategy import Strategy
from backtest.backtester import Backtester
from backtest.metrics import Metrics

class Main:
    """Main class serves as the entry point for the backtesting system.

    It loads the configuration, runs data loading, signal generation, backtesting,
    computes performance metrics, and then outputs the results along with plotting the equity curve.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Main by loading the configuration.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        logging.info("Configuration loaded successfully from %s", config_path)

    def _load_config(self, config_path: str) -> dict:
        """Load the configuration from the given YAML file path.

        Args:
            config_path (str): Path to the config.yaml file.

        Returns:
            dict: The configuration dictionary.
        """
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            if config is None:
                config = {}
            return config
        except Exception as e:
            logging.error("Error loading configuration: %s", e)
            raise e

    def run(self) -> None:
        """Orchestrate the data loading, signal generation, backtesting, and metrics computation."""
        logging.info("Starting backtest system run.")

        # 1. Data Loading
        data_loader = DataLoader(self.config)
        price_df = data_loader.load_price_data()
        logging.info("Price data loaded with shape: %s", price_df.shape)

        # 2. Signal Generation using Strategy
        strategy = Strategy(self.config)
        signals_df = strategy.generate_signals(price_df)
        logging.info("Signals generated with shape: %s", signals_df.shape)

        # 3. Backtesting
        backtester = Backtester(self.config)
        perf_df = backtester.run_backtest(price_df, signals_df)
        logging.info("Backtest completed. Performance DataFrame shape: %s", perf_df.shape)

        # 4. Select a Formation Period Group for performance analysis.
        strat_params = self.config.get("strategy", {}).get("params", {})
        formation_periods = strat_params.get("formation_periods", [3, 6, 9, 12])
        default_period = formation_periods[0] if formation_periods else 3

        strategy_return_col = f"strategy_return_{default_period}"
        portfolio_value_col = f"portfolio_value_{default_period}"

        if strategy_return_col not in perf_df.columns:
            logging.error("Expected column '%s' not found in performance DataFrame.", strategy_return_col)
            raise ValueError("Missing expected strategy return column in performance DataFrame.")

        # Create a new DataFrame with a "returns" column required for Metrics computations.
        metrics_perf_df = pd.DataFrame()
        metrics_perf_df["returns"] = perf_df[strategy_return_col]

        # 5. Compute Performance Metrics
        metrics = Metrics(self.config)
        metrics_dict = metrics.compute(metrics_perf_df)
        logging.info("Performance metrics computed successfully.")

        # Print out the performance metrics.
        print(f"\nPerformance Metrics for Formation Period {default_period}:")
        for metric, value in metrics_dict.items():
            print(f"{metric}: {value:.4f}")

        # 6. Plot the equity curve for the chosen formation period.
        if portfolio_value_col in perf_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(perf_df.index, perf_df[portfolio_value_col], marker="o", linestyle="-")
            plt.title(f"Equity Curve for Formation Period {default_period}")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("Portfolio value column '%s' not found in performance DataFrame.", portfolio_value_col)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main_app = Main("config.yaml")
    main_app.run()
