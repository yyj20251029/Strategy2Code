# backtest/backtester.py

import logging
from typing import Dict
import numpy as np
import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)


class Backtester:
    """
    Backtester class for simulating the portfolio rebalancing process over a fixed period.
    
    The backtester uses daily price data and trading signals to simulate portfolio dynamics,
    accounting for automatic weight evolution due to price movements, rebalancing to target weights,
    transaction cost impact, and calculates daily log returns. The performance history is compiled into
    a pandas DataFrame.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the Backtester with configuration parameters.

        Expected keys in config['backtest']:
            - initial_capital: Starting portfolio value.
            - transaction_cost: Flat transaction cost rate per trade.
            - rebalance_frequency: Frequency for rebalancing (e.g., "1d").
            - allow_short: Boolean flag for short selling (not further used in simulation).
            - max_leverage: Maximum leverage permitted.
            - risk_free_rate: Risk-free rate (for performance metrics, if needed).

        Also uses config['data']['tickers'] which determines the asset ordering:
            - The portfolio is constructed as: [cash] + tickers.
        """
        self.config = config
        backtest_config = config.get('backtest', {})
        
        self.initial_capital: float = float(backtest_config.get('initial_capital', 100000.0))
        self.transaction_cost_rate: float = float(backtest_config.get('transaction_cost', 0.0025))
        self.rebalance_frequency: str = backtest_config.get('rebalance_frequency', '1d')
        self.allow_short: bool = bool(backtest_config.get('allow_short', True))
        self.max_leverage: float = float(backtest_config.get('max_leverage', 1.0))
        self.risk_free_rate: float = float(backtest_config.get('risk_free_rate', 0.0))
        
        # Get tickers from data configuration; these represent risky assets.
        data_config = config.get('data', {})
        self.tickers: list = data_config.get('tickers', [])
        
        logger.info("Backtester initialized with initial_capital=%.2f, transaction_cost_rate=%.4f, "
                    "rebalance_frequency=%s, allow_short=%s, max_leverage=%.2f, risk_free_rate=%.2f, "
                    "tickers=%s",
                    self.initial_capital, self.transaction_cost_rate, self.rebalance_frequency,
                    self.allow_short, self.max_leverage, self.risk_free_rate, self.tickers)

    def run_backtest(self, price_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest simulation over the fixed period using price and signal DataFrames.

        This method performs the following steps:
          1. Aligns the price_df and signals_df on their datetime index.
          2. Extracts closing prices for risky assets and constructs an asset price DataFrame,
             including a constant price for cash.
          3. Iterates day-by-day to simulate portfolio evolution:
              a. Compute daily price relative vector Y_t.
              b. Calculate the return factor using previous weights.
              c. Compute auto-evolved weights from price changes.
              d. Retrieve target weights from signals and calculate transaction cost based on risky asset changes.
              e. Update portfolio value and compute daily log returns.
              f. Set the updated target weights as the weights for the next day.
          4. Compiles the simulation records into a performance DataFrame.

        Args:
            price_df (pd.DataFrame): DataFrame containing daily price data.
                It must include the 'Close' prices for each ticker.
            signals_df (pd.DataFrame): DataFrame containing portfolio target weight signals.
                The columns should be named as 'w_0', 'w_1', ..., where 'w_0' corresponds to cash.

        Returns:
            pd.DataFrame: Performance DataFrame with a Date index and columns:
                - "portfolio_value": Portfolio value at the end of the day.
                - "daily_log_return": Daily log return computed as ln(new_value/old_value).
                - "transaction_cost": Transaction cost incurred on that day.
        """
        # Ensure that indices are of type DatetimeIndex
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        if not isinstance(signals_df.index, pd.DatetimeIndex):
            signals_df.index = pd.to_datetime(signals_df.index)
        
        # Align both DataFrames on common dates.
        common_dates = price_df.index.intersection(signals_df.index).sort_values()
        if common_dates.empty:
            logger.error("No common dates between price data and signals data.")
            raise ValueError("Price data and signals data do not share any common dates.")
        
        price_df_aligned = price_df.loc[common_dates].copy()
        signals_df_aligned = signals_df.loc[common_dates].copy()
        logger.info("Aligned price and signals data to %d common dates.", len(common_dates))

        # Prepare asset prices:
        # The portfolio consists of cash (always priced at 1) and risky assets from tickers.
        n_assets = len(self.tickers) + 1  # cash + risky assets
        # Check that signals_df has the expected number of columns.
        if signals_df_aligned.shape[1] != n_assets:
            logger.error("Mismatch in number of assets: signals_df has %d columns, expected %d.",
                         signals_df_aligned.shape[1], n_assets)
            raise ValueError("Signals DataFrame does not have the expected number of asset columns.")

        # Extract closing prices for each ticker.
        # If the price DataFrame has MultiIndex columns, extract the 'Close' field.
        risky_prices = pd.DataFrame(index=common_dates)
        if isinstance(price_df_aligned.columns, pd.MultiIndex):
            for ticker in self.tickers:
                if ticker in price_df_aligned.columns.get_level_values(0):
                    try:
                        close_series = price_df_aligned[ticker]['Close']
                    except KeyError:
                        logger.error("Ticker %s does not have 'Close' in its data.", ticker)
                        raise ValueError(f"'Close' price not found for ticker {ticker}.")
                    risky_prices[ticker] = close_series
                else:
                    logger.error("Ticker %s not found in price data.", ticker)
                    raise ValueError(f"Ticker {ticker} is missing from the price data.")
        else:
            # Assume price_df_aligned columns are directly tickers.
            for ticker in self.tickers:
                if ticker in price_df_aligned.columns:
                    risky_prices[ticker] = price_df_aligned[ticker]
                else:
                    logger.error("Ticker %s not found in price data.", ticker)
                    raise ValueError(f"Ticker {ticker} is missing from the price data.")

        # Construct a full asset price DataFrame with cash and risky assets.
        asset_prices = pd.DataFrame(index=common_dates)
        # Cash asset: constant price of 1.
        asset_prices['asset_0'] = 1.0
        # Add risky asset prices; order must correspond to signals_df columns (w_1, w_2,...)
        for idx, ticker in enumerate(self.tickers, start=1):
            asset_prices[f'asset_{idx}'] = risky_prices[ticker]
        logger.info("Constructed asset prices DataFrame with %d assets.", n_assets)

        # Identify valid simulation dates: those dates in signals_df_aligned where signals are not NaN.
        valid_signals_df = signals_df_aligned.dropna(how="any")
        if valid_signals_df.empty:
            logger.error("No valid trading signals available after dropping NaN values.")
            raise ValueError("Signals DataFrame contains only NaN values.")
        
        valid_dates = valid_signals_df.index.sort_values()
        if len(valid_dates) < 2:
            logger.error("Not enough valid trading days for backtesting (found %d).", len(valid_dates))
            raise ValueError("Insufficient valid trading days for simulation.")

        # Initialize simulation state.
        portfolio_value: float = self.initial_capital
        n_valid = len(valid_dates)
        # Use the first valid day signals as the starting weights.
        previous_weights: np.ndarray = valid_signals_df.iloc[0].to_numpy(dtype=np.float64)
        logger.info("Starting simulation on %s with initial portfolio value %.2f.",
                    valid_dates[0].strftime("%Y-%m-%d"), portfolio_value)

        # List to hold simulation records.
        records = []
        # Record the initial day (no return computed for the very first valid day)
        records.append({
            "date": valid_dates[0],
            "portfolio_value": portfolio_value,
            "daily_log_return": 0.0,
            "transaction_cost": 0.0
        })

        # Iterate over valid trading days (starting from the second valid day)
        for i in range(1, n_valid):
            current_date = valid_dates[i]
            prev_date = valid_dates[i - 1]

            # Build price relative vector Y_t for all assets.
            # For cash (index 0), Y = 1.
            Y_t = np.zeros(n_assets, dtype=np.float64)
            Y_t[0] = 1.0
            for j in range(1, n_assets):
                asset_col = f'asset_{j}'
                current_price = asset_prices.loc[current_date, asset_col]
                previous_price = asset_prices.loc[prev_date, asset_col]
                # Guard against division by zero.
                if previous_price == 0.0:
                    logger.warning("Previous price for %s on %s is zero; setting price relative to 1.", 
                                   asset_col, prev_date)
                    Y_t[j] = 1.0
                else:
                    Y_t[j] = current_price / previous_price
            logger.debug("Date %s: Price relative vector Y_t: %s", current_date, Y_t)

            # Compute daily return factor using previous weights.
            # factor = exp( sum( previous_weights * ln(Y_t) ) )
            try:
                log_Y = np.log(Y_t)
            except Exception as e:
                logger.error("Error computing log(Y_t) on %s: %s", current_date, e)
                raise
            factor = np.exp(np.dot(previous_weights, log_Y))
            logger.debug("Date %s: Return factor: %.6f", current_date, factor)

            # Simulate automatic weight evolution (W'_t):
            numerator = previous_weights * Y_t
            denominator = np.sum(np.abs(previous_weights) * Y_t)
            if denominator == 0:
                logger.warning("Denominator in weight evolution is zero on %s, retaining previous weights.", current_date)
                W_prime = previous_weights.copy()
            else:
                W_prime = numerator / denominator
            logger.debug("Date %s: Auto-evolved weights W'_t: %s", current_date, W_prime)

            # Retrieve target weights (W_target) from signals for current day.
            W_target = signals_df_aligned.loc[current_date].to_numpy(dtype=np.float64)
            logger.debug("Date %s: Target weights W_target: %s", current_date, W_target)

            # Calculate transaction cost on rebalancing for risky assets (indices 1 to end).
            # Compute L1 difference for risky assets between W_prime and W_target.
            cost_change = np.sum(np.abs(W_prime[1:] - W_target[1:]))
            transaction_cost = self.transaction_cost_rate * cost_change
            logger.debug("Date %s: Transaction cost calculated: %.6f", current_date, transaction_cost)

            # Update portfolio value: V_new = V_old * (1 - transaction_cost) * factor
            new_portfolio_value = portfolio_value * (1 - transaction_cost) * factor
            if portfolio_value <= 0:
                logger.warning("Portfolio value is non-positive on %s", current_date)
            daily_log_return = np.log(new_portfolio_value / portfolio_value)
            logger.info("Date %s: Portfolio value updated from %.2f to %.2f with daily log return %.6f",
                        current_date.strftime("%Y-%m-%d"), portfolio_value, new_portfolio_value, daily_log_return)

            # Record data for current day.
            records.append({
                "date": current_date,
                "portfolio_value": new_portfolio_value,
                "daily_log_return": daily_log_return,
                "transaction_cost": transaction_cost
            })

            # Update state for next day: set previous weights to the target weights.
            previous_weights = W_target.copy()
            portfolio_value = new_portfolio_value

        # Compile records into a pandas DataFrame.
        performance_df = pd.DataFrame(records)
        performance_df.set_index("date", inplace=True)
        performance_df.sort_index(inplace=True)
        logger.info("Backtest simulation completed over %d trading days.", len(performance_df))
        
        return performance_df
