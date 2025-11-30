## strategy/strategy.py

import logging
from typing import Dict, List
import numpy as np
import pandas as pd

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Strategy:
    """Strategy class for generating TSMOM trading signals.

    This class computes formation returns from monthly price data and generates trading signals
    according to the time-series momentum (TSMOM) methodology described in the paper.
    It supports both "directional" (scaled) and "binary" (sign-based) signals for each asset,
    over one or more formation periods.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the Strategy with configuration settings.

        Args:
            config (Dict): Configuration dictionary read from config.yaml.
        """
        strat_config = config.get("strategy", {})
        self.strategy_name: str = strat_config.get("name", "paper_strategy")
        self.strategy_type: str = strat_config.get("type", "tsmom")
        params: Dict = strat_config.get("params", {})
        self.formation_periods: List[int] = params.get("formation_periods", [3, 6, 9, 12])
        self.signal_type: str = params.get("signal_type", "directional")

        if self.signal_type not in {"directional", "binary"}:
            logger.warning("Invalid signal_type in config: %s. Defaulting to 'directional'.", self.signal_type)
            self.signal_type = "directional"

        logger.info("Strategy initialized with formation periods: %s and signal type: %s",
                    self.formation_periods, self.signal_type)

    def compute_formation_returns(self, price_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Compute k-month formation (cumulative) returns for each asset.

        For a given formation period k, and monthly price data, this method calculates the sum
        of previous k monthly returns. Returns are computed from price percentage changes,
        and shifted by one period to avoid look-ahead bias.

        Args:
            price_df (pd.DataFrame): Monthly price data with DateTimeIndex and asset columns.
            k (int): Formation period (in months).

        Returns:
            pd.DataFrame: DataFrame of formation returns where each month t is the sum of returns from t-1 to t-k.
        """
        # Compute simple monthly returns (percentage change)
        monthly_returns: pd.DataFrame = price_df.pct_change()

        # Shift returns by one to use t-1, t-2,..., t-k for formation returns at time t
        shifted_returns: pd.DataFrame = monthly_returns.shift(1)

        # Compute rolling sum over the previous k months with full window (min_periods=k)
        formation_returns: pd.DataFrame = shifted_returns.rolling(window=k, min_periods=k).sum()

        logger.debug("Computed formation returns for period k=%d", k)
        return formation_returns

    def normalize_signals(self, formation_returns: pd.DataFrame) -> pd.DataFrame:
        """Normalize formation returns by the cross-sectional standard deviation.

        In the directional TSMOM strategy, each asset's formation return is divided by the
        cross-sectional standard deviation of formation returns across all assets on the same date.
        When only a single asset is present or if the standard deviation is 0 or NaN, the divisor is set to 1.

        Args:
            formation_returns (pd.DataFrame): DataFrame of formation returns.

        Returns:
            pd.DataFrame: DataFrame of normalized (scaled) signals.
        """
        # Compute the cross-sectional standard deviation along each date (row)
        sigma: pd.Series = formation_returns.std(axis=1, ddof=1)
        # Replace zeros or NaN values with 1 to avoid division errors
        sigma_replaced: pd.Series = sigma.replace(0, 1).fillna(1)
        normalized_signals: pd.DataFrame = formation_returns.div(sigma_replaced, axis=0)

        logger.debug("Normalized signals based on cross-sectional standard deviation.")
        return normalized_signals

    def generate_signals(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Generate TSMOM trading signals from monthly price data.

        For each formation period specified in the configuration, this method computes the formation returns.
        Depending on the signal type defined ("directional" or "binary"), it generates the corresponding signals.
        The output columns are renamed to indicate the formation period (e.g., "3_SPY") and combined across all periods.

        Args:
            price_df (pd.DataFrame): Monthly price data with DateTimeIndex and asset columns.

        Returns:
            pd.DataFrame: A combined DataFrame of signals. Columns are named as "<formation_period>_<ticker>".
        """
        signals_list: List[pd.DataFrame] = []

        for k in self.formation_periods:
            # Compute formation returns for current formation period k.
            formation_returns: pd.DataFrame = self.compute_formation_returns(price_df, k)

            # Generate signals based on configuration
            if self.signal_type == "directional":
                signals: pd.DataFrame = self.normalize_signals(formation_returns)
            elif self.signal_type == "binary":
                signals = np.sign(formation_returns)
                # Optionally replace zeros with +1 to ensure nonzero signal as per paper's formulation
                signals = signals.replace(0, 1)
            else:
                # Fallback to directional if signal_type is not recognized; should not happen.
                signals = self.normalize_signals(formation_returns)

            # Rename columns to include the formation period as a prefix (e.g., "3_SPY")
            # This helps in differentiating signals for different formation periods.
            signals = signals.rename(columns=lambda ticker: f"{k}_{ticker}")
            signals_list.append(signals)

            logger.info("Generated signals for formation period %d", k)

        # Concatenate signals for all formation periods along columns.
        if signals_list:
            signals_df: pd.DataFrame = pd.concat(signals_list, axis=1)
        else:
            signals_df = pd.DataFrame()

        logger.info("Final signals DataFrame generated with columns: %s", signals_df.columns.tolist())
        return signals_df
