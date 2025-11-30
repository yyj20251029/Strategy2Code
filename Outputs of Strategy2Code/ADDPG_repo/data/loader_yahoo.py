## data/loader_yahoo.py

import pandas as pd
import yfinance as yf
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class DataLoader:
    """DataLoader downloads and cleans historical market data from Yahoo Finance.

    Attributes:
        tickers (list): List of ticker symbols.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        interval (str): Data interval (e.g., '1d' for daily data).
    """

    def __init__(self, config: dict):
        """
        Initialize DataLoader with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing data source parameters.
                           Expected keys:
                                - "data": {
                                      "source": "yahoo",
                                      "tickers": list of tickers,
                                      "start_date": str (e.g., "2000-01-01"),
                                      "end_date": str (e.g., "2024-01-01"),
                                      "interval": str (e.g., "1d")
                                  }
        Raises:
            ValueError: If required configuration keys are missing or invalid.
        """
        if "data" not in config:
            raise ValueError("Configuration must contain a 'data' section.")

        data_config = config["data"]

        # Validate tickers
        tickers = data_config.get("tickers")
        if not tickers or not isinstance(tickers, list):
            raise ValueError("Tickers must be provided as a non-empty list in config['data']['tickers'].")
        self.tickers = tickers

        # Validate date range
        self.start_date = data_config.get("start_date")
        self.end_date = data_config.get("end_date")
        if not self.start_date or not self.end_date:
            raise ValueError("Both 'start_date' and 'end_date' must be specified in config['data'].")
        
        # Validate data interval
        self.interval = data_config.get("interval", "1d")

        # Validate data source; only 'yahoo' is supported in this implementation.
        source = data_config.get("source", "yahoo")
        if source.lower() != "yahoo":
            raise ValueError("Only the 'yahoo' data source is supported. Provided source: {}".format(source))
        
        logger.info("DataLoader initialized with tickers: %s, date range: %s to %s, interval: %s",
                    self.tickers, self.start_date, self.end_date, self.interval)

    def load_price_data(self) -> pd.DataFrame:
        """
        Download historical OHLCV data from Yahoo Finance and clean the data.

        The method:
          - Downloads data for the specified tickers and date range using yfinance.
          - Ensures the index is a DateTimeIndex.
          - Reindexes the DataFrame to include a complete set of business days within the fixed date range.
          - Fills missing data using forward-fill and backfill methods.
          - Drops any rows that still contain missing values.

        Returns:
            pd.DataFrame: Cleaned DataFrame with dates as its index and OHLCV data in columns.
                        For multiple tickers, columns are organized in a MultiIndex (ticker, field).

        Raises:
            RuntimeError: If an error occurs during data download.
            ValueError: If the downloaded data is empty.
        """
        try:
            logger.info("Attempting to download data from Yahoo Finance for tickers: %s", self.tickers)
            # Download data using yfinance with group_by parameter for multi-ticker organization.
            data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                group_by="ticker",
                progress=False
            )
        except Exception as e:
            logger.error("Error downloading data from Yahoo Finance: %s", e)
            raise RuntimeError(f"Error downloading data from Yahoo Finance: {e}")

        if data.empty:
            error_msg = "Downloaded data is empty. Please check tickers and date range."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure the index is a DateTimeIndex.
        data.index = pd.to_datetime(data.index)
        
        # Create a complete date range using business days to align time series across tickers.
        full_index = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        logger.info("Reindexing data to cover full business day range from %s to %s", self.start_date, self.end_date)
        data = data.reindex(full_index)

        # Handle missing data: forward-fill then backfill if necessary.
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(inplace=True)

        logger.info("Data download and cleaning completed. Data shape: %s", data.shape)
        return data
