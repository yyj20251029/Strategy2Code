## backtest/metrics.py

import logging
from typing import Dict
import numpy as np
import pandas as pd

# Set up module-level logger.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metrics:
    """Metrics class for computing performance metrics from backtest returns.

    This class computes key performance indicators including annualized return,
    volatility, Sharpe ratio, maximum drawdown, and win rate using monthly strategy
    returns from the backtest performance DataFrame. The risk-free rate (annual)
    is read from the configuration.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize Metrics with required configuration settings.

        Reads the annual risk-free rate from config["backtest"]["risk_free_rate"].
        If not specified, defaults to 0.0.

        Args:
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        backtest_config = config.get("backtest", {})
        self.risk_free_rate: float = float(backtest_config.get("risk_free_rate", 0.0))
        logger.info("Metrics initialized with risk_free_rate: %.4f", self.risk_free_rate)

    def annualized_return(self, returns: pd.Series) -> float:
        """Compute the geometric (compound) annualized return from monthly returns.

        The calculation uses the formula:
            annualized_return = (∏ₜ(1 + Rₜ))^(12/N) − 1
        where N is the number of monthly periods.

        Args:
            returns (pd.Series): Series of monthly strategy returns.

        Returns:
            float: Annualized return as a decimal.
        """
        returns = returns.dropna()
        if returns.empty:
            raise ValueError("The returns series is empty; cannot compute annualized return.")
        cumulative_return: float = (returns + 1).prod()
        n_periods: int = len(returns)
        ann_return: float = cumulative_return ** (12.0 / n_periods) - 1
        logger.debug("Calculated annualized return: %.4f", ann_return)
        return ann_return

    def volatility(self, returns: pd.Series) -> float:
        """Compute the annualized volatility (standard deviation) from monthly returns.

        The sample standard deviation is multiplied by the square root of 12.

        Args:
            returns (pd.Series): Series of monthly strategy returns.

        Returns:
            float: Annualized volatility as a decimal.
        """
        returns = returns.dropna()
        if returns.empty:
            raise ValueError("The returns series is empty; cannot compute volatility.")
        monthly_std: float = returns.std(ddof=1)
        ann_vol: float = monthly_std * np.sqrt(12)
        logger.debug("Calculated annualized volatility: %.4f", ann_vol)
        return ann_vol

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Compute the Sharpe ratio using annualized returns and volatility.

        The Sharpe ratio is defined as:
            Sharpe ratio = (annualized_return - risk_free_rate) / annualized_volatility

        Args:
            returns (pd.Series): Series of monthly strategy returns.
            risk_free_rate (float): Annual risk-free rate as a decimal.

        Returns:
            float: Sharpe ratio. Returns np.nan if volatility is zero.
        """
        returns = returns.dropna()
        if returns.empty:
            raise ValueError("The returns series is empty; cannot compute Sharpe ratio.")
        ann_ret: float = self.annualized_return(returns)
        ann_vol: float = self.volatility(returns)
        if ann_vol == 0:
            logger.warning("Volatility is zero; returning np.nan for Sharpe ratio.")
            return np.nan
        sharpe: float = (ann_ret - risk_free_rate) / ann_vol
        logger.debug("Calculated Sharpe ratio: %.4f", sharpe)
        return sharpe

    def max_drawdown(self, returns: pd.Series) -> float:
        """Calculate the maximum drawdown from the monthly returns.

        The process is:
            1. Compute the equity curve as the cumulative product of (1 + returns).
            2. Compute the running maximum of the equity curve.
            3. Calculate the drawdown at each period and return the absolute maximum drawdown.

        Args:
            returns (pd.Series): Series of monthly strategy returns.

        Returns:
            float: Maximum drawdown as a positive decimal (percentage loss).
        """
        returns = returns.dropna()
        if returns.empty:
            raise ValueError("The returns series is empty; cannot compute max drawdown.")
        equity_curve: pd.Series = (returns + 1).cumprod()
        running_max: pd.Series = equity_curve.cummax()
        drawdowns: pd.Series = (equity_curve - running_max) / running_max
        max_dd: float = abs(drawdowns.min())
        logger.debug("Calculated maximum drawdown: %.4f", max_dd)
        return max_dd

    def win_rate(self, returns: pd.Series) -> float:
        """Compute the win rate as the fraction of periods with positive returns.

        Args:
            returns (pd.Series): Series of monthly strategy returns.

        Returns:
            float: Win rate as a float between 0 and 1.
        """
        returns = returns.dropna()
        if returns.empty:
            raise ValueError("The returns series is empty; cannot compute win rate.")
        positive_periods: int = (returns > 0).sum()
        total_periods: int = len(returns)
        win_rate_value: float = positive_periods / total_periods
        logger.debug("Calculated win rate: %.4f", win_rate_value)
        return win_rate_value

    def compute(self, perf_df: pd.DataFrame) -> Dict[str, float]:
        """Compute performance metrics from a performance DataFrame.

        The performance DataFrame must contain a column named "returns" representing
        monthly strategy returns. The following metrics are computed:
            - annual_return: Annualized compound return.
            - volatility: Annualized standard deviation.
            - sharpe: Risk-adjusted return.
            - max_drawdown: Maximum observed drawdown.
            - win_rate: Fraction of months with positive returns.

        Args:
            perf_df (pd.DataFrame): DataFrame containing the backtest performance.
                Must have a "returns" column.

        Returns:
            Dict[str, float]: Dictionary of computed performance metrics.
        """
        if "returns" not in perf_df.columns:
            error_msg = ("The performance DataFrame must contain a column named 'returns' "
                         "representing monthly strategy returns.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        returns_series: pd.Series = perf_df["returns"]
        logger.info("Computing performance metrics from %d return observations.", len(returns_series.dropna()))

        metrics_dict: Dict[str, float] = {
            "annual_return": self.annualized_return(returns_series),
            "volatility": self.volatility(returns_series),
            "sharpe": self.sharpe_ratio(returns_series, self.risk_free_rate),
            "max_drawdown": self.max_drawdown(returns_series),
            "win_rate": self.win_rate(returns_series)
        }
        logger.info("Performance metrics computed: %s", metrics_dict)
        return metrics_dict
