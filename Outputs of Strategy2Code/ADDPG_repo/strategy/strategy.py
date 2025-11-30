## strategy/strategy.py

import numpy as np
import pandas as pd
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class Strategy:
    """Adaptive DDPG Strategy for generating trading signals.

    This class reads the strategy-specific configuration from a dictionary and, given a
    historical price DataFrame, produces daily portfolio weight adjustments (signals)
    that mimic an Adaptive DDPG trading mechanism with asymmetric exploration noise.
    
    The generated signals DataFrame has the same index (dates) and columns (tickers) as the price data.
    Each row is a valid portfolio allocation vector (non-negative weights that sum to 1).

    Attributes:
        name (str): Strategy name.
        strategy_type (str): Type of strategy.
        alpha_plus (float): Learning rate for positive prediction errors.
        alpha_minus (float): Learning rate for negative prediction errors.
        gamma (float): Discount factor.
        tau (float): Soft update parameter.
        actor_lr (float): Learning rate for the actor network.
        critic_lr (float): Learning rate for the critic network.
        noise_positive (str): Noise distribution type for positive daily returns.
        noise_negative (str): Noise distribution type for negative daily returns.
        scaling_factor (float): Constant scaling factor (k) used in signal adjustment.
    """

    def __init__(self, config: dict):
        """
        Initialize the Strategy with configuration parameters.

        Args:
            config (dict): A configuration dictionary containing the strategy section.
                           Expected structure:
                           {
                               "strategy": {
                                   "name": "Adaptive DDPG",
                                   "type": "adaptive_ddpg",
                                   "params": {
                                       "alpha_plus": float,
                                       "alpha_minus": float,
                                       "gamma": float,
                                       "tau": float,
                                       "actor_lr": float,
                                       "critic_lr": float,
                                       "exploration_noise": {
                                           "positive": str,  # e.g., "standard_normal"
                                           "negative": str   # e.g., "negative_only"
                                       }
                                   }
                               }
                           }
        Raises:
            ValueError: If required configuration items are missing.
        """
        if "strategy" not in config:
            raise ValueError("Configuration must contain a 'strategy' section.")

        strategy_config = config["strategy"]

        self.name = strategy_config.get("name", "Adaptive DDPG")
        self.strategy_type = strategy_config.get("type", "adaptive_ddpg")

        params = strategy_config.get("params", {})
        self.alpha_plus = float(params.get("alpha_plus", 1.0))
        self.alpha_minus = float(params.get("alpha_minus", 0.0))
        self.gamma = float(params.get("gamma", 0.99))
        self.tau = float(params.get("tau", 0.001))
        self.actor_lr = float(params.get("actor_lr", 0.0001))
        self.critic_lr = float(params.get("critic_lr", 0.001))
        
        exploration_noise = params.get("exploration_noise", {})
        self.noise_positive = exploration_noise.get("positive", "standard_normal")
        self.noise_negative = exploration_noise.get("negative", "negative_only")

        # Scaling factor for signal adjustment (can be adjusted if needed)
        self.scaling_factor = 0.1

        logger.info("Strategy '%s' of type '%s' initialized with parameters: alpha_plus=%s, alpha_minus=%s, "
                    "gamma=%s, tau=%s, actor_lr=%s, critic_lr=%s, noise_positive=%s, noise_negative=%s, scaling_factor=%s",
                    self.name, self.strategy_type, self.alpha_plus, self.alpha_minus, self.gamma, self.tau,
                    self.actor_lr, self.critic_lr, self.noise_positive, self.noise_negative, self.scaling_factor)

    def generate_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from historical price data.

        This method processes the provided price_data DataFrame, computes daily returns,
        applies asymmetric exploration noise based on recent price movements, and constructs new
        portfolio weight vectors. The signals are generated iteratively starting from an equal-weight
        allocation on the first trading day and adjusted on each subsequent day.

        Args:
            price_data (pd.DataFrame): DataFrame containing historical daily prices with:
                                       - Index: Date stamps (datetime)
                                       - Columns: Either ticker symbols or a MultiIndex (ticker, field)
                                                 where 'Close' is assumed to be the price used.

        Returns:
            pd.DataFrame: A DataFrame with the same index and columns as the input price data (tickers only).
                          Each row represents the portfolio allocation weights for that day (floats in [0, 1] that sum to 1).

        Raises:
            ValueError: If price_data is empty or does not contain valid price values.
        """
        if price_data.empty:
            raise ValueError("Price data provided to generate_signals is empty.")

        # Determine if the DataFrame has MultiIndex columns, and extract 'Close' prices if so.
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Close' not in price_data.columns.get_level_values(1):
                raise ValueError("MultiIndex price_data does not contain 'Close' column.")
            price_df = price_data.xs('Close', level=1, axis=1)
            logger.info("Extracted 'Close' prices from MultiIndex columns.")
        else:
            # Assume price_data columns are ticker names with price values.
            price_df = price_data.copy()

        # Ensure price_df is sorted by its index (date)
        price_df.sort_index(inplace=True)
        dates = price_df.index
        tickers = list(price_df.columns)
        n_assets = len(tickers)
        if n_assets == 0:
            raise ValueError("No tickers found in the price data.")

        # Initialize the signals DataFrame with the same structure as price_df.
        signals = pd.DataFrame(index=dates, columns=tickers, dtype=np.float64)

        # Set the first day signal to equal weights across all assets.
        init_weight = 1.0 / n_assets
        signals.iloc[0] = np.full(n_assets, init_weight)
        logger.info("Initial portfolio set to equal weights: %s", signals.iloc[0].tolist())

        # Iterate over each day from the second day onward to generate signals.
        for i in range(1, len(dates)):
            # Current and previous prices (as Series)
            prev_prices = price_df.iloc[i - 1]
            curr_prices = price_df.iloc[i]
            
            # Calculate daily returns: daily_return = (current / previous - 1)
            daily_return = (curr_prices / prev_prices) - 1

            # Get previous day weights from signals.
            prev_weights = signals.iloc[i - 1].values.astype(np.float64)

            # Initialize noise vector for each asset.
            noise_vector = np.zeros(n_assets, dtype=np.float64)
            # Vectorized noise generation:
            # For assets with non-negative daily return, sample from standard normal
            # For assets with negative daily return, sample so that noise is negative-only.
            positive_indices = daily_return >= 0
            negative_indices = daily_return < 0
            if positive_indices.sum() > 0:
                noise_pos = np.random.standard_normal(positive_indices.sum())
                noise_vector[positive_indices.values] = noise_pos
            if negative_indices.sum() > 0:
                noise_neg = -np.abs(np.random.standard_normal(negative_indices.sum()))
                noise_vector[negative_indices.values] = noise_neg

            # Compute raw new weights using the previous weights, daily returns, and noise.
            adjustment = self.scaling_factor * (daily_return.values + noise_vector)
            new_weights_raw = prev_weights + adjustment

            # Enforce non-negative weights.
            new_weights_clipped = np.clip(new_weights_raw, a_min=0, a_max=None)

            # Normalize the weights so that they sum to 1.
            total_weight = new_weights_clipped.sum()
            if total_weight > 0:
                new_weights_normalized = new_weights_clipped / total_weight
            else:
                # Fallback to equal weights if normalization is impossible.
                new_weights_normalized = np.full(n_assets, 1.0 / n_assets)
                logger.warning("Total weight sum is zero at index %s; reverting to equal weights.", dates[i])

            # Store the computed weights in the signals DataFrame.
            signals.iloc[i] = new_weights_normalized

        logger.info("Signal generation completed. Generated signals for %d days.", len(dates))
        return signals
