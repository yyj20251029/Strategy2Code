## backtest/metrics.py

import numpy as np
import pandas as pd
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class Metrics:
    """
    Metrics class computes key performance indicators (KPIs) from the backtest performance 
    DataFrame produced by the Backtester. The KPIs include annualized return (CAGR), 
    annualized volatility, Sharpe ratio, maximum drawdown, and win rate.

    Attributes:
        risk_free_rate (float): The risk-free rate used in Sharpe ratio computation.
        trading_days (int): Number of trading days used for annualizing volatility (default 252).
    """

    def __init__(self, config: dict):
        """
        Initializes the Metrics instance with configuration parameters.

        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
                           Must contain a "backtest" section with key "risk_free_rate".
                           If not provided, risk_free_rate defaults to 0.0.

        Raises:
            ValueError: If the "backtest" section is missing from the configuration.
        """
        if "backtest" not in config:
            raise ValueError("Configuration must include a 'backtest' section.")
        
        backtest_config = config["backtest"]
        self.risk_free_rate: float = float(backtest_config.get("risk_free_rate", 0.0))
        self.trading_days: int = 252  # Default number of trading days per year for annualization.

        logger.info(
            "Metrics initialized with risk_free_rate=%s and trading_days=%s",
            self.risk_free_rate, self.trading_days
        )

    def compute(self, performance_df: pd.DataFrame) -> dict:
        """
        Compute key performance metrics from the performance DataFrame.
        
        The performance DataFrame must have:
            - A DateTimeIndex in ascending order.
            - A column named "portfolio_value" containing the portfolio's time series.

        The method computes:
            1. Daily returns from the portfolio_value.
            2. Compound Annual Growth Rate (CAGR) as the annualized return.
            3. Annualized volatility calculated from the standard deviation of daily returns.
            4. Sharpe ratio given by (annual_return - risk_free_rate) / annualized volatility.
            5. Maximum drawdown by evaluating the decline from a peak in portfolio_value.
            6. Win rate as the fraction of days with positive daily returns.

        Args:
            performance_df (pd.DataFrame): Performance DataFrame with portfolio values and date index.

        Returns:
            dict: A dictionary containing the computed metrics, with keys:
                  "annual_return", "volatility", "sharpe", "max_drawdown", "win_rate".

        Raises:
            ValueError: If the input DataFrame is empty or does not contain "portfolio_value".
        """
        if performance_df.empty:
            raise ValueError("Performance DataFrame is empty. Cannot compute metrics.")
        
        if "portfolio_value" not in performance_df.columns:
            raise ValueError("Performance DataFrame must contain 'portfolio_value' column.")

        # Ensure the DataFrame is sorted by the index (date)
        perf_df = performance_df.copy()
        perf_df.sort_index(inplace=True)

        # Forward-fill any missing portfolio values (if any)
        perf_df["portfolio_value"].fillna(method="ffill", inplace=True)

        # Extract the portfolio value series
        portfolio_values: pd.Series = perf_df["portfolio_value"]

        # Calculate daily returns and remove the first NaN value
        daily_returns: pd.Series = portfolio_values.pct_change().dropna()
        logger.info("Computed daily returns; sample values: %s", daily_returns.head().to_dict())

        # Compute Compound Annual Growth Rate (CAGR)
        initial_value: float = float(portfolio_values.iloc[0])
        final_value: float = float(portfolio_values.iloc[-1])
        start_date = perf_df.index[0]
        end_date = perf_df.index[-1]
        total_days: float = (end_date - start_date).days
        years: float = total_days / 365.25  # Using 365.25 for calendar year accounting for leap years
        
        if years <= 0:
            annual_return: float = np.nan
            logger.warning("Non-positive duration in years computed (years=%s).", years)
        else:
            annual_return = (final_value / initial_value) ** (1 / years) - 1
        logger.info("CAGR (annual return) computed: %s", annual_return)

        # Compute annualized volatility
        daily_std: float = float(daily_returns.std())
        annualized_volatility: float = daily_std * np.sqrt(self.trading_days)
        logger.info("Annualized volatility computed: %s", annualized_volatility)

        # Compute Sharpe ratio, handling division by zero
        if annualized_volatility == 0:
            sharpe_ratio: float = np.nan
            logger.warning("Annualized volatility is zero; Sharpe ratio set to NaN.")
        else:
            sharpe_ratio = (annual_return - self.risk_free_rate) / annualized_volatility
        logger.info("Sharpe ratio computed: %s", sharpe_ratio)

        # Compute Maximum Drawdown
        running_max: pd.Series = portfolio_values.cummax()
        drawdowns: pd.Series = portfolio_values / running_max - 1.0
        max_drawdown: float = float(drawdowns.min())
        logger.info("Maximum drawdown computed: %s", max_drawdown)

        # Compute Win Rate: fraction of days with positive daily returns
        positive_days: int = int((daily_returns > 0).sum())
        total_return_days: int = int(len(daily_returns))
        win_rate: float = positive_days / total_return_days if total_return_days > 0 else np.nan
        logger.info("Win rate computed: %s", win_rate)

        metrics_dict = {
            "annual_return": float(annual_return),
            "volatility": float(annualized_volatility),
            "sharpe": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate)
        }
        
        logger.info("Metrics computed successfully: %s", metrics_dict)
        return metrics_dict
