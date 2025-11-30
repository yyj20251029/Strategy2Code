# data/loader_yahoo.py

import logging
from typing import Dict
import pandas as pd
import numpy as np
import yfinance as yf

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class DataLoader:
    """
    DataLoader for fetching historical daily OHLCV data from Yahoo Finance.
    
    This class reads configuration parameters from a config dictionary, downloads data 
    using the yfinance library, handles missing values, aligns data across tickers, and 
    returns a cleaned pandas DataFrame.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the DataLoader with configuration parameters.

        Args:
            config (Dict): Configuration dictionary containing data parameters.
                Expected keys in config['data']:
                    - source: Data source identifier, expected "yahoo".
                    - tickers: List of ticker strings.
                    - start_date: Start date string in "YYYY-MM-DD" format.
                    - end_date: End date string in "YYYY-MM-DD" format.
                    - interval: Data interval string (e.g., "1d" for daily data).
        """
        data_config = config.get('data', {})
        self.source: str = data_config.get('source', 'yahoo')
        self.tickers: list = data_config.get('tickers', ["000300.SS", "600519.SS", "601318.SS", "000651.SZ"])
        self.start_date: str = data_config.get('start_date', "2000-01-01")
        self.end_date: str = data_config.get('end_date', "2024-01-01")
        self.interval: str = data_config.get('interval', "1d")
        
        logger.info("DataLoader initialized with source: %s, tickers: %s, start_date: %s, end_date: %s, interval: %s",
                    self.source, self.tickers, self.start_date, self.end_date, self.interval)

    def load_price_data(self) -> pd.DataFrame:
        """
        Download and clean OHLCV price data from Yahoo Finance.
        
        This method uses the yfinance library to download daily data for the specified tickers 
        over the fixed date range provided in the configuration. It handles missing data by 
        forward filling and then fills any remaining missing values with zeros. The method also 
        ensures that the DataFrame's index is a sorted pandas DatetimeIndex.

        Returns:
            pd.DataFrame: Cleaned and aligned OHLCV data with a common date index.
        Raises:
            ValueError: If the downloaded data is empty.
        """
        try:
            logger.info("Downloading data for tickers: %s", self.tickers)
            price_df: pd.DataFrame = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                auto_adjust=False,
                group_by='ticker',
                threads=True
            )
        except Exception as e:
            logger.error("Error downloading data: %s", e)
            raise RuntimeError(f"Failed to download price data: {e}") from e

        if price_df.empty:
            logger.error("Downloaded price data is empty. Check ticker symbols and date range.")
            raise ValueError("Downloaded price data is empty. Check ticker symbols and date range.")

        logger.info("Data download complete. DataFrame shape before cleaning: %s", price_df.shape)

        # Ensure the index is a DatetimeIndex and sorted
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        price_df.sort_index(inplace=True)

        # Handle missing values: first forward-fill, then fill remaining NaNs with zeros
        missing_before: int = price_df.isnull().sum().sum()
        price_df = price_df.ffill().fillna(0)
        missing_after: int = price_df.isnull().sum().sum()
        logger.info("Missing values before cleaning: %d, after cleaning: %d", missing_before, missing_after)

        # If the DataFrame has multi-index columns due to multiple tickers,
        # ensure the data is properly aligned by reindexing with the union of all dates.
        all_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        price_df = price_df.reindex(all_dates, method='ffill').fillna(0)

        logger.info("DataFrame shape after reindexing: %s", price_df.shape)
        return price_df
