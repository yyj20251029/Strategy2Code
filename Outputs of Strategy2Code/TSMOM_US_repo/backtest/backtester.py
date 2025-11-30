## backtest/backtester.py

import logging
from typing import Dict
import numpy as np
import pandas as pd

# Set up module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtester class for running the TSMOM backtest.

    This class simulates a monthly rebalanced portfolio over the fixed backtest period.
    It uses monthly price data and corresponding TSMOM signals to compute portfolio returns
    for each formation period group and applies transaction costs and leverage constraints accordingly.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the Backtester with configuration settings.

        Args:
            config (Dict): Configuration dictionary read from config.yaml.
        """
        backtest_cfg = config.get("backtest", {})
        # Set configuration parameters with defaults if not provided.
        self.initial_capital: float = float(backtest_cfg.get("initial_capital", 100000.0))
        self.transaction_cost_bp: float = float(backtest_cfg.get("transaction_cost_bp", 5))
        self.rebalance_frequency: str = backtest_cfg.get("rebalance_frequency", "1M")
        self.allow_short: bool = bool(backtest_cfg.get("allow_short", True))
        self.max_leverage: float = float(backtest_cfg.get("max_leverage", 1.0))
        self.risk_free_rate: float = float(backtest_cfg.get("risk_free_rate", 0.0))
        logger.info("Backtester initialized with initial capital: %.2f, transaction cost: %.2f bp, max leverage: %.2f",
                    self.initial_capital, self.transaction_cost_bp, self.max_leverage)

    def run_backtest(self, price_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Runs the backtest simulation using monthly price data and generated signals.
        
        The method aligns the price and signal DataFrames, computes monthly returns,
        applies a one-period lag on signals, and then iteratively simulates the portfolio's performance.
        For each formation period (extracted from signal column names), the backtester simulates
        portfolio evolution by calculating gross returns, applying transaction costs based on turnover,
        and updating the portfolio value accordingly.
        
        Args:
            price_df (pd.DataFrame): Monthly price data with DateTimeIndex and asset columns.
            signals_df (pd.DataFrame): DataFrame of TSMOM signals with DateTimeIndex.
                Expected column names follow the pattern "<formation_period>_<ticker>".
                
        Returns:
            pd.DataFrame: Performance DataFrame with DateTimeIndex containing columns for strategy returns,
            portfolio value, turnover, and transaction costs for each formation period group.
            Column names are suffixed with the formation period (e.g., "strategy_return_3").
        """
        logger.info("Starting backtest simulation.")
        
        # Compute monthly asset returns from price data.
        asset_returns: pd.DataFrame = price_df.pct_change().dropna()
        logger.info("Computed asset monthly returns.")

        # Shift signals forward by one period to simulate execution delay.
        effective_signals: pd.DataFrame = signals_df.shift(1)

        # Align effective signals and asset returns: take inner join on index.
        common_index = effective_signals.index.intersection(asset_returns.index)
        effective_signals = effective_signals.loc[common_index].copy()
        asset_returns = asset_returns.loc[common_index].copy()
        logger.info("Aligned signals and asset returns over %d periods.", len(common_index))

        # Group signals by formation period.
        # Expected signal column format: "<formation_period>_<ticker>"
        formation_groups: Dict[int, list] = {}
        for col in effective_signals.columns:
            try:
                prefix, ticker = col.split("_", 1)
                period_int = int(prefix)
                if period_int not in formation_groups:
                    formation_groups[period_int] = []
                formation_groups[period_int].append(col)
            except Exception as e:
                logger.warning("Column '%s' does not match expected format '<period>_<ticker>': %s", col, e)

        if not formation_groups:
            logger.error("No valid formation groups found in signals DataFrame.")
            raise ValueError("Signals DataFrame does not contain valid column names for formation groups.")

        # Dictionary to store performance DataFrames for each formation period.
        performance_results = {}

        # Convert transaction cost in basis points to decimal.
        trans_cost_decimal: float = self.transaction_cost_bp / 10000.0

        # Run simulation for each formation period group.
        for period, columns in formation_groups.items():
            logger.info("Simulating backtest for formation period: %d (using columns: %s)", period, columns)
            # Initialize previous signals (for each asset in the group) to zero.
            previous_signals = pd.Series(0, index=columns, dtype=np.float64)
            portfolio_value: float = self.initial_capital
            performance_records = []

            # Iterate over each simulation date.
            for current_date in sorted(common_index):
                # Get current effective signals for this group; fill missing with 0.
                current_signals = effective_signals.loc[current_date, columns].fillna(0)
                # Enforce the max leverage constraint.
                current_signals = current_signals.clip(lower=-self.max_leverage, upper=self.max_leverage)

                # For each column, extract the corresponding ticker and get the asset return.
                signal_return_products = []
                for col in columns:
                    try:
                        # Extract ticker from column name.
                        _, ticker = col.split("_", 1)
                        # Retrieve the asset's return for current_date.
                        ret = asset_returns.at[current_date, ticker]
                        signal_value = current_signals[col]
                        product = signal_value * ret
                        signal_return_products.append(product)
                    except Exception as e:
                        logger.warning("Error processing column '%s' on date %s: %s", col, current_date, e)
                        # In case of error, skip this column by appending 0.
                        signal_return_products.append(0.0)

                # If no valid product exists, skip this date.
                if not signal_return_products:
                    continue

                # Compute the gross return as the average product across assets.
                gross_return = np.mean(signal_return_products)

                # Compute turnover as the average absolute change in signals.
                turnover = np.mean(np.abs(current_signals - previous_signals))

                # Compute transaction cost for this period.
                transaction_cost = turnover * trans_cost_decimal

                # Net return after deducting transaction cost.
                net_return = gross_return - transaction_cost

                # Update portfolio value.
                portfolio_value = portfolio_value * (1 + net_return)

                # Record the performance metrics for this period.
                performance_records.append({
                    "date": current_date,
                    "strategy_return": net_return,
                    "portfolio_value": portfolio_value,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost
                })

                # Update previous signals for next period.
                previous_signals = current_signals.copy()

            # Convert records to DataFrame and set date as index.
            if performance_records:
                perf_df_group = pd.DataFrame(performance_records)
                perf_df_group.set_index("date", inplace=True)
                # Rename columns by appending the formation period to identify the group.
                perf_df_group = perf_df_group.rename(columns={
                    "strategy_return": f"strategy_return_{period}",
                    "portfolio_value": f"portfolio_value_{period}",
                    "turnover": f"turnover_{period}",
                    "transaction_cost": f"transaction_cost_{period}"
                })
                performance_results[period] = perf_df_group
                logger.info("Completed simulation for formation period %d over %d periods.", 
                            period, len(perf_df_group))
            else:
                logger.warning("No performance records for formation period %d.", period)

        if not performance_results:
            logger.error("No simulation results generated for any formation period.")
            raise ValueError("Backtest simulation failed to generate any performance data.")

        # Merge performance DataFrames from all formation periods on the date index.
        merged_performance_df = pd.concat(performance_results.values(), axis=1, join="outer")
        merged_performance_df.sort_index(inplace=True)
        logger.info("Backtest simulation completed. Merged performance DataFrame has %d periods.", 
                    merged_performance_df.shape[0])
        return merged_performance_df
