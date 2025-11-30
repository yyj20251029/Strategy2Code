# backtest/metrics.py

import logging
import numpy as np
import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Metrics:
    """
    Metrics class for computing key performance metrics from backtest results.
    
    This class computes annualized return (CAGR), annualized volatility,
    Sharpe ratio, maximum drawdown, and win rate based on a performance DataFrame.
    It uses configuration parameters (such as the risk-free rate and the list of metrics)
    provided in the config dictionary and assumes 252 trading days per year.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the Metrics class with configuration parameters.

        Args:
            config (dict): Configuration dictionary.
                Expected keys in config['backtest']:
                    - risk_free_rate: Annual risk-free rate (default is 0.0).
                    - metrics: List of metric names to compute (default 
                      ["annual_return", "volatility", "sharpe", "max_drawdown", "win_rate"]).
        """
        backtest_config = config.get("backtest", {})
        self.risk_free_rate: float = float(backtest_config.get("risk_free_rate", 0.0))
        self.trading_days_per_year: int = 252  # Standard assumption of trading days in a year.
        self.metric_names = backtest_config.get(
            "metrics",
            ["annual_return", "volatility", "sharpe", "max_drawdown", "win_rate"]
        )
        logger.info(
            "Metrics initialized with risk_free_rate=%.4f, trading_days_per_year=%d, metrics=%s",
            self.risk_free_rate, self.trading_days_per_year, self.metric_names
        )

    def compute(self, performance_df: pd.DataFrame) -> dict:
        """
        Compute performance metrics from the backtest performance DataFrame.
        
        The DataFrame must include:
          - A DateTimeIndex.
          - A "portfolio_value" column representing daily portfolio equity.
          - Optionally a "daily_return" column. If missing, daily simple returns are computed
            using portfolio_value.pct_change().

        Computed metrics include:
          - annual_return: Compound Annual Growth Rate (CAGR)
          - volatility: Annualized volatility of daily returns
          - sharpe: Sharpe ratio, computed as (annual_return - risk_free_rate) / volatility
          - max_drawdown: Maximum percentage drop from a peak in portfolio value (reported as positive value)
          - win_rate: Fraction of days with positive daily return

        Args:
            performance_df (pd.DataFrame): DataFrame containing backtest results.

        Returns:
            dict: A dictionary containing computed metrics.
        """
        # Validate that the DataFrame is not empty.
        if performance_df.empty:
            logger.error("Performance DataFrame is empty. Cannot compute metrics.")
            raise ValueError("Performance DataFrame is empty.")

        # Ensure the index is a DatetimeIndex.
        if not isinstance(performance_df.index, pd.DatetimeIndex):
            try:
                performance_df.index = pd.to_datetime(performance_df.index)
            except Exception as e:
                logger.error("Error converting index to DatetimeIndex: %s", e)
                raise

        # Check required column "portfolio_value" exists.
        if "portfolio_value" not in performance_df.columns:
            logger.error("'portfolio_value' column is missing in Performance DataFrame.")
            raise ValueError("'portfolio_value' column is required for metric computation.")

        portfolio_values: pd.Series = performance_df["portfolio_value"].dropna()
        if portfolio_values.empty:
            logger.error("No valid portfolio values found after dropping NaNs.")
            raise ValueError("Portfolio value series is empty.")

        # Use provided daily returns if available; otherwise compute simple returns.
        if "daily_return" in performance_df.columns:
            daily_returns: pd.Series = performance_df["daily_return"].dropna()
            logger.info("Using provided 'daily_return' column for metric computation.")
        else:
            daily_returns = portfolio_values.pct_change().dropna()
            logger.info("Computed daily simple returns from 'portfolio_value' column.")

        # Ensure sufficient data exists.
        num_data_points: int = len(portfolio_values)
        if num_data_points < 2:
            logger.error("Insufficient data points (%d) to compute metrics.", num_data_points)
            raise ValueError("At least two data points are required for metric computation.")

        # Annualized Return (CAGR)
        initial_value: float = float(portfolio_values.iloc[0])
        final_value: float = float(portfolio_values.iloc[-1])
        # Number of trading periods is number of returns (num_data_points - 1)
        num_periods: int = num_data_points - 1
        try:
            annual_return: float = (final_value / initial_value) ** (self.trading_days_per_year / num_periods) - 1.0
        except Exception as e:
            logger.error("Error computing annual return: %s", e)
            raise

        # Annualized Volatility: Standard deviation of daily returns scaled by sqrt(trading_days_per_year)
        daily_std: float = float(daily_returns.std())
        annual_volatility: float = daily_std * np.sqrt(self.trading_days_per_year)

        # Sharpe Ratio: (annual_return - risk_free_rate) divided by annual volatility.
        if annual_volatility != 0:
            sharpe_ratio: float = (annual_return - self.risk_free_rate) / annual_volatility
        else:
            logger.warning("Annual volatility is zero; setting Sharpe ratio to 0.")
            sharpe_ratio = 0.0

        # Maximum Drawdown: Calculate running maximum and then the drawdown series.
        running_max: pd.Series = portfolio_values.cummax()
        drawdowns: pd.Series = (portfolio_values - running_max) / running_max
        max_drawdown: float = abs(drawdowns.min())  # Report as a positive number representing magnitude

        # Win Rate: Fraction of days with a positive daily return.
        winning_days: int = int((daily_returns > 0).sum())
        total_trading_days: int = len(daily_returns)
        win_rate: float = winning_days / total_trading_days if total_trading_days > 0 else 0.0

        # Compile the computed metrics in a dictionary.
        computed_metrics: dict = {
            "annual_return": annual_return,
            "volatility": annual_volatility,
            "sharpe": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate
        }

        logger.info("Computed Metrics: %s", computed_metrics)
        return computed_metrics
