## data/loader_yahoo.py

import logging
from typing import Dict
import pandas as pd
import yfinance as yf

# Set up basic logging configuration for this module.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """DataLoader class for downloading and processing price data from Yahoo Finance.

    This class downloads daily OHLCV data for the configured tickers from Yahoo Finance,
    cleans the data by forward-filling any missing values, and resamples the data to monthly
    frequency (taking the last available price in each month) over a fixed backtesting period.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the DataLoader with configuration settings.

        Args:
            config (Dict): Configuration dictionary read from config.yaml.
        """
        data_config = config.get("data", {})
        self.source: str = data_config.get("source", "yahoo")
        self.tickers: list = data_config.get("tickers", ["SPY"])
        self.start_date: str = data_config.get("start_date", "2000-01-01")
        self.end_date: str = data_config.get("end_date", "2024-01-01")
        self.interval: str = data_config.get("interval", "1d")

    def load_price_data(self) -> pd.DataFrame:
        """Download daily price data, clean it, and resample to monthly frequency.

        Returns:
            pd.DataFrame: A DataFrame with monthly price data, where the index represents
            month-end dates and columns correspond to ticker symbols.

        Raises:
            ValueError: If the downloaded or resampled data is empty.
            Exception: If an error occurs during the download process.
        """
        logger.info("Starting download of price data for tickers: %s", self.tickers)
        try:
            raw_data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
        except Exception as e:
            logger.error("Error downloading data from Yahoo Finance: %s", e)
            raise Exception(f"Failed to download data: {e}")

        if raw_data.empty:
            error_msg = "Downloaded data is empty. Check tickers and date range in the configuration."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extract the most appropriate price column, preferring "Adj Close"
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multi-ticker: extract the 'Adj Close' prices for all tickers.
            if "Adj Close" in raw_data.columns.get_level_values(0):
                price_data = raw_data["Adj Close"]
            else:
                logger.warning("Adjusted Close data not found; defaulting to Close prices.")
                if "Close" in raw_data.columns.get_level_values(0):
                    price_data = raw_data["Close"]
                else:
                    # As a fallback, use the first available column.
                    price_data = raw_data.iloc[:, 0]
        else:
            # Single ticker: check if "Adj Close" exists.
            if "Adj Close" in raw_data.columns:
                price_data = raw_data["Adj Close"]
            else:
                logger.warning("Adjusted Close column not found; using the first column of data.")
                price_data = raw_data.iloc[:, 0]

        # Ensure the index is a DatetimeIndex and sorted.
        price_data.index = pd.to_datetime(price_data.index)
        price_data.sort_index(inplace=True)

        # Clean missing data: forward-fill and then drop any remaining NaN values.
        price_data.ffill(inplace=True)
        price_data.dropna(inplace=True)

        # Resample the cleaned daily data to monthly frequency using the last observed price.
        monthly_prices = price_data.resample("M").last()
        monthly_prices.dropna(inplace=True)

        if monthly_prices.empty:
            error_msg = "Monthly resampled price data is empty after processing."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # If the result is a Series (for a single ticker), convert it to a DataFrame.
        if isinstance(monthly_prices, pd.Series):
            monthly_prices = monthly_prices.to_frame(name=self.tickers[0])

        logger.info("Successfully loaded and processed monthly price data.")
        return monthly_prices
