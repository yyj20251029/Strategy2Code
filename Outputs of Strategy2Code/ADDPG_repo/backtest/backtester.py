## backtest/backtester.py

import pandas as pd
import numpy as np
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class Backtester:
    """
    Backtester simulates portfolio evolution over a fixed date range by rebalancing the portfolio
    based on trading signals. It uses daily price data and target allocation weights from signals,
    executes trades with transaction costs, and updates the portfolio's cash and asset holdings.
    
    The performance is recorded as a daily portfolio value along with cash and asset value.
    
    Configuration Parameters from config.yaml (under "backtest"):
      - initial_capital: float, starting portfolio capital.
      - transaction_cost_bp: int/float, transaction cost in basis points (1 bp = 0.01%).
      - rebalance_frequency: str, frequency of rebalancing (e.g., "1d").
      - allow_short: bool, whether short selling is allowed.
      - max_leverage: float, maximum allowable leverage.
      - risk_free_rate: float, risk free rate (unused in simulation but available for metrics).
    """

    def __init__(self, config: dict):
        """
        Initializes the Backtester with backtesting configuration parameters.

        Args:
            config (dict): Configuration dictionary containing a "backtest" section.
                           Expected keys include "initial_capital", "transaction_cost_bp",
                           "rebalance_frequency", "allow_short", "max_leverage", and "risk_free_rate".
        
        Raises:
            ValueError: If the configuration does not contain required backtest parameters.
        """
        if "backtest" not in config:
            raise ValueError("Configuration must include a 'backtest' section.")

        backtest_config = config["backtest"]

        # Set configuration parameters with explicit types and default values.
        self.initial_capital: float = float(backtest_config.get("initial_capital", 10000.0))
        self.transaction_cost_bp: float = float(backtest_config.get("transaction_cost_bp", 5))
        self.rebalance_frequency: str = str(backtest_config.get("rebalance_frequency", "1d"))
        self.allow_short: bool = bool(backtest_config.get("allow_short", False))
        self.max_leverage: float = float(backtest_config.get("max_leverage", 1.0))
        self.risk_free_rate: float = float(backtest_config.get("risk_free_rate", 0.0))

        # Derive fee rate: transaction_cost_bp is in basis points (1 bp = 0.0001)
        self.fee_rate: float = self.transaction_cost_bp / 10000.0

        logger.info("Backtester initialized with initial_capital=%s, transaction_cost_bp=%s, "
                    "rebalance_frequency=%s, allow_short=%s, max_leverage=%s, risk_free_rate=%s",
                    self.initial_capital, self.transaction_cost_bp, self.rebalance_frequency,
                    self.allow_short, self.max_leverage, self.risk_free_rate)

    def run_backtest(self, price_data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate portfolio evolution over time given historical price data and trading signals.

        The simulation proceeds day-by-day over the common dates in price_data and signals.
        On each rebalancing day (when a signal is available), the portfolio is rebalanced by
        computing the required trades based on target allocations, adjusting cash for transaction fees,
        and updating asset holdings.

        Args:
            price_data (pd.DataFrame): DataFrame of daily prices with dates as index and columns corresponding to tickers.
                                       If the DataFrame uses MultiIndex columns, it is assumed that level 'Close'
                                       contains the prices.
            signals (pd.DataFrame): DataFrame indexed by date with target allocation weights for each ticker.
                                    Each row must be non-negative and sum to 1 (or will be normalized).

        Returns:
            pd.DataFrame: Performance DataFrame indexed by date with columns:
                          - "portfolio_value": Total portfolio value (cash + asset value).
                          - "cash": Cash balance.
                          - "asset_value": Total value of asset holdings.
        
        Raises:
            ValueError: If either input DataFrame is empty or if required price data is missing.
        """
        if price_data.empty:
            raise ValueError("Price data is empty. Cannot run backtest.")
        if signals.empty:
            raise ValueError("Signals DataFrame is empty. Cannot run backtest.")

        # If price_data has MultiIndex columns, extract 'Close' prices.
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Close' in price_data.columns.get_level_values(1):
                price_data = price_data.xs('Close', level=1, axis=1)
                logger.info("Extracted 'Close' prices from MultiIndex columns in price_data.")
            else:
                raise ValueError("Price data MultiIndex does not contain 'Close' prices.")

        # Align the dates between price_data and signals.
        common_dates = price_data.index.intersection(signals.index)
        if common_dates.empty:
            raise ValueError("No common dates between price_data and signals.")
        common_dates = common_dates.sort_values()

        # Use the tickers from the price_data columns.
        tickers = list(price_data.columns)
        if len(tickers) == 0:
            raise ValueError("No ticker columns found in price_data.")

        logger.info("Running backtest over %d common trading days for tickers: %s",
                    len(common_dates), tickers)

        # Initialize simulation variables.
        cash: float = self.initial_capital
        # Holdings: pandas Series with ticker names as index, initial shares are 0.0
        holdings: pd.Series = pd.Series(data=np.zeros(len(tickers)), index=tickers, dtype=np.float64)

        # List to collect daily performance records.
        performance_records = []

        # Loop over each trading day.
        for current_date in common_dates:
            # Get current day's price vector for all tickers.
            try:
                prices_today: pd.Series = price_data.loc[current_date]
            except KeyError:
                logger.warning("Prices for date %s not found in price_data; skipping.", current_date)
                continue

            # Mark-to-market: Evaluate current asset value.
            asset_value: float = (holdings * prices_today).sum()
            portfolio_value: float = cash + asset_value

            # Determine if today is a rebalancing day (i.e., a signal exists for this day).
            if current_date in signals.index:
                signal_row: pd.Series = signals.loc[current_date]

                # Enforce no-short constraint if not allowed.
                if not self.allow_short:
                    signal_row = signal_row.clip(lower=0.0)

                # Normalize target weights to sum to 1.
                weight_sum = signal_row.sum()
                if weight_sum <= 0:
                    # Fallback to equal weights if signal row sums to zero.
                    target_weights: pd.Series = pd.Series(
                        data=np.full(len(tickers), 1.0 / len(tickers)), index=tickers, dtype=np.float64)
                    logger.warning("Signal weights sum to zero on %s; reverting to equal weights.", current_date)
                else:
                    target_weights = signal_row / weight_sum

                # Calculate total portfolio value again (using V0) for rebalancing.
                V0: float = portfolio_value

                # Prepare to accumulate total cash changes from trades.
                total_buy_cost: float = 0.0
                total_sell_proceeds: float = 0.0

                # Dictionary to hold new holdings after trades.
                new_holdings = {}

                # Execute trades for each ticker.
                for ticker in tickers:
                    price = prices_today[ticker]
                    if price <= 0:
                        logger.warning("Price for ticker %s on %s is non-positive; skipping trade.", ticker, current_date)
                        new_holdings[ticker] = holdings[ticker]
                        continue

                    # Target dollar allocation for the ticker.
                    target_dollar: float = target_weights.get(ticker, 0.0) * V0
                    # Current dollar allocation for the ticker.
                    current_dollar: float = holdings.get(ticker, 0.0) * price
                    # Trade amount in dollars required.
                    trade_dollar: float = target_dollar - current_dollar
                    fee: float = abs(trade_dollar) * self.fee_rate

                    # Accumulate cash changes.
                    if trade_dollar > 0:
                        # Buying: cost is trade amount plus fee.
                        total_buy_cost += (trade_dollar + fee)
                    elif trade_dollar < 0:
                        # Selling: proceeds are absolute trade amount minus fee.
                        total_sell_proceeds += (abs(trade_dollar) - fee)

                    # Update holding for ticker: new number of shares equals target dollars divided by current price.
                    new_shares: float = target_dollar / price
                    new_holdings[ticker] = new_shares

                # Update cash concurrently after processing all tickers.
                new_cash: float = cash + total_sell_proceeds - total_buy_cost

                # Log the rebalancing details.
                logger.debug("Rebalancing on %s: V0=%.2f, cash before=%.2f, cash after=%.2f, total_buy_cost=%.2f, total_sell_proceeds=%.2f",
                             current_date, V0, cash, new_cash, total_buy_cost, total_sell_proceeds)

                # Update portfolio state with new cash and new holdings.
                cash = new_cash
                holdings = pd.Series(new_holdings)

                # After rebalancing, recalc asset value and portfolio value.
                asset_value = (holdings * prices_today).sum()
                portfolio_value = cash + asset_value

            # Record the day's performance.
            performance_records.append({
                "date": current_date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "asset_value": asset_value
            })

        # Convert the performance records list to a DataFrame.
        performance_df = pd.DataFrame(performance_records)
        performance_df.set_index("date", inplace=True)
        performance_df.sort_index(inplace=True)

        logger.info("Backtest completed over %d days. Final portfolio value: %.2f", len(performance_df), performance_df["portfolio_value"].iloc[-1])
        return performance_df
