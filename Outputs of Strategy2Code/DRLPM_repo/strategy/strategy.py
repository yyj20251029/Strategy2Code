# strategy/strategy.py

import logging
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class Strategy:
    """
    Strategy class for simulating portfolio signal generation inspired by the DRLPM paper.

    This simulation generates a constant portfolio weight signal vector for each trading day
    (once sufficient history is available) that satisfies:
      - The cash weight (first element) is within [0, 1].
      - Risky asset weights (all other elements) are within [-1, 1].
      - The entire vector is L1-normalized (sum of absolute weights equals 1).
      - The arbitrage constraint is applied: if all risky weights (indices 1 to end) are
        either all nonnegative or all nonpositive (and nonzero in aggregate), then the benchmark asset's
        weight (last element) is flipped and the vector is renormalized.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the Strategy with configuration parameters.

        Args:
            config (Dict): Configuration dictionary containing strategy parameters.
                Expected keys:
                  - 'strategy': with subkey 'params' containing:
                        - observation_window (default: 50)
                        - arbitrage_threshold (default: 0.0)
                  - 'data': containing the list of tickers.
        """
        self.config = config
        strategy_config = config.get('strategy', {})
        params = strategy_config.get('params', {})

        self.observation_window: int = int(params.get('observation_window', 50))
        self.arbitrage_threshold: float = float(params.get('arbitrage_threshold', 0.0))

        logger.info("Strategy initialized with observation_window: %d, arbitrage_threshold: %f",
                    self.observation_window, self.arbitrage_threshold)

    def _normalize_and_enforce_constraints(self, raw_weights: np.ndarray) -> np.ndarray:
        """
        Enforce individual bounds, normalize the vector to have L1 norm equal to 1, and apply
        arbitrage constraint by flipping the benchmark asset weight if needed.

        Args:
            raw_weights (np.ndarray): Raw portfolio weight vector.
        
        Returns:
            np.ndarray: Final normalized portfolio weight vector.
        """
        weights = raw_weights.copy().astype(np.float64)

        # Enforce bounds
        # Cash weight (index 0) must be in [0, 1]
        weights[0] = np.clip(weights[0], 0.0, 1.0)
        # Risky asset weights (indices 1 to end) must be in [-1, 1]
        if weights.size > 1:
            weights[1:] = np.clip(weights[1:], -1.0, 1.0)

        # Normalize weights so that the L1 norm (sum of absolute values) equals 1
        l1_norm = np.sum(np.abs(weights))
        if l1_norm > 0:
            weights = weights / l1_norm
        else:
            logger.warning("L1 norm of weights is zero. Returning zero vector.")
            return np.zeros_like(weights)

        # Apply arbitrage constraint on risky weights:
        # If all risky weights (indices 1 to end) are nonnegative or all nonpositive and their
        # aggregate absolute sum is nonzero, then flip the benchmark weight (last element)
        if weights.size > 1:
            risky_weights = weights[1:]
            if (np.all(risky_weights >= 0) or np.all(risky_weights <= 0)) and (np.sum(np.abs(risky_weights)) > 0):
                weights[-1] = -weights[-1]
                logger.info("Arbitrage constraint applied: benchmark weight flipped.")
                # Re-normalize after flipping
                l1_norm = np.sum(np.abs(weights))
                if l1_norm > 0:
                    weights = weights / l1_norm
                else:
                    logger.warning("L1 norm of weights is zero after arbitrage adjustment. Returning zero vector.")
                    return np.zeros_like(weights)

        return weights

    def generate_signals(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate portfolio weight signals based on a price DataFrame.

        For dates where there is not enough historical data (less than observation_window days),
        the signals will be set to NaN. For subsequent dates, a constant weight vector is generated,
        normalized, and adjusted to meet the short-selling and arbitrage constraints.

        Args:
            price_df (pd.DataFrame): DataFrame indexed by dates containing price data.
        
        Returns:
            pd.DataFrame: DataFrame of portfolio weight signals with columns 'w_0', 'w_1', ..., 'w_{n_assets-1}'.
                        The first element ('w_0') represents cash weight, and the remaining elements represent
                        risky weights (with the last one assumed to be the benchmark).
        """
        # Determine number of assets: cash + one for each ticker
        tickers = self.config.get('data', {}).get('tickers', [])
        if not tickers:
            logger.error("No tickers provided in configuration data.")
            raise ValueError("Tickers list is empty in configuration data.")

        n_assets = len(tickers) + 1  # +1 for cash
        signal_columns = [f"w_{i}" for i in range(n_assets)]

        # Initialize an empty DataFrame for signals with the same index as price_df
        signals_df = pd.DataFrame(index=price_df.index, columns=signal_columns, dtype=np.float64)

        # Build a constant raw weight vector (simulation of converged DRLPM portfolio weights)
        # Set the cash candidate (index 0) to 0.5.
        # For non-benchmark risky assets (indices 1 to n_assets-2) assign 0.1 each,
        # and for the benchmark asset (last element) assign 0.2.
        if n_assets < 2:
            logger.error("Number of assets computed is less than 2, cannot form portfolio weights.")
            raise ValueError("Insufficient number of assets to form portfolio weights.")

        raw_weights = np.zeros(n_assets, dtype=np.float64)
        raw_weights[0] = 0.5  # Cash candidate

        if n_assets > 2:
            # Assign constant weight (e.g., 0.1) for each non-benchmark risky asset.
            raw_weights[1:n_assets-1] = 0.1
        # Assign benchmark candidate for the last risky asset.
        raw_weights[-1] = 0.2

        logger.info("Raw weights before normalization: %s", raw_weights)

        # Normalize and enforce financial constraints (bounds, L1 normalization, arbitrage rule)
        normalized_weights = self._normalize_and_enforce_constraints(raw_weights)
        logger.info("Normalized weights after enforcement: %s", normalized_weights)

        # Generate signals for each trading date.
        total_dates = len(signals_df)
        if total_dates <= self.observation_window:
            logger.warning("Insufficient data: total dates (%d) are less than or equal to observation window (%d).",
                           total_dates, self.observation_window)
            signals_df[:] = np.nan
            return signals_df

        # Fill each row with the constant normalized weight vector.
        signals_df.loc[:] = normalized_weights

        # For the initial 'observation_window' days, assign NaN signals to indicate insufficient history.
        signals_df.iloc[:self.observation_window, :] = np.nan

        logger.info("Signals DataFrame generated with constant normalized weights from index %d onwards.",
                    self.observation_window)
        return signals_df
