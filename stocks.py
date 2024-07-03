"""
This class sets up a financial asset and augments the price action data with technical and economic indicators
"""
import logging
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from Logger import CustomLogger

logs = CustomLogger()


class instrument:
    """
        This class object fetches xxxxx and xxx data
    """

    def __init__(self, ticker, start_date='1990-01-01', end_date='2026-12-31', frequency='1d'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency  # '15m', '90m' , '1h', '1d', '1wk', '1mo', '3mo'

    def download_price_volume(self):
        """

        :return: returns dataframe of asset
        """
        try:
            df = yf.download(self.ticker, self.start_date, self.end_date, self.frequency)
            logs.log("Successfully downloaded price-action data from yahoo finance API")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logs.log("Something went wrong while downloading price-action data from yahoo finance API", level='ERROR')

    def enrich_data_date(self):
        """

        :return:
        """
        try:
            df = self.download_price_volume()
            df = df.copy()
            df['Date'] = df.index
            df['Month'] = df.index.month
            df['Year'] = df.index.year

            if self.frequency == '1d':
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
                df["is_weekend"] = df.index.to_series().dt.dayofweek > 4
                df["is_week_start"] = df.index.to_series().dt.dayofweek == 0
                df['day_name'] = df.index.to_series().dt.day_name()

            elif self.frequency == '1wk':
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end

            elif self.frequency == '1mo':
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
            logs.log("Successfully enriched data with date fields")
            return df
        except Exception as e:
            raise ValueError(f"Error in enriching data with date fields: {e}")
            logs.log("Something went wrong while enriching the data with date fields", level='ERROR')

    def add_technical_indicators(self):
        """
        :return:
        """
        try:
            df = self.enrich_data_date()
            df = df.copy()
            df.ta.rsi(close="Close", append=True)
            df.ta.macd(close="Close", append=True)
            df.ta.atr(length=14, append=True)
            df.ta.bbands(append=True)
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['Volatility_10'] = df['Close'].rolling(window=10).std()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['Volatility_20'] = df['Close'].rolling(window=20).std()
            df['Return'] = df['Close'].pct_change()
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            # Calculate ADX, +DI, and -DI
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            # Append the ADX, +DI, and -DI to the original DataFrame
            df = df.join(adx)
            # Fetch VIX data from Yahoo Finance using yfinance
            vix_data = yf.download("^VIX",self.start_date, self.end_date, self.frequency)
            # Rename the 'Adj Close' column to 'VIX' for consistency
            vix_data = vix_data.rename(columns={'Adj Close': 'VIX'})
            # Merge the stock_data DataFrame with the VIX data
            df = pd.merge(df, vix_data['VIX'], how='left', left_index=True, right_index=True)
            logs.log("Successfully enriched data with technical indicator fields")
            return df
        except Exception as e:
            raise ValueError(f"Error in enriching data with technical indicator fields: {e}")
            logs.log("Something went wrong while enriching the data technical indicator fields", level='ERROR')

    def add_macro_indicators(self):
        """
        :return:
        """
        try:
            try:
                # Fetch Consumer Price Index (CPI) data from FRED
                cpi_data = web.DataReader("CPIAUCNS", "fred", self.start_date, self.end_date)
                # Assuming 'cpi_data' is your DataFrame and 'CPIAUCNS' is the column representing CPI
                cpi_data['InflationChange'] = cpi_data['CPIAUCNS'].pct_change() * 100
                cpi_data['Month'] = cpi_data.index.month
                cpi_data['Year'] = cpi_data.index.year
                logs.log("Successfully extracted CPI data fields")
            except Exception as e:
                raise ValueError(f"Error while extracting CPI data fields: {e}")
                logs.log("Something went wrong while extracting the CPI data fields", level='ERROR')

            try:
                # Fetch Federal Funds Rate data from FRED
                fed_funds_rate = web.DataReader("FEDFUNDS", "fred", self.start_date, self.end_date)

                fed_funds_rate['fedfundsChange'] = fed_funds_rate['FEDFUNDS'].pct_change() * 100
                fed_funds_rate['Month'] = fed_funds_rate.index.month
                fed_funds_rate['Year'] = fed_funds_rate.index.year
                logs.log("Successfully extracted the Fed fund rates data fields")
            except Exception as e:
                raise ValueError(f"Error while extracting the Fed Fund rates data fields: {e}")
                logs.log("Something went wrong while extracting the Fed Fund rates data fields", level='ERROR')

            try:
                # Fetch Non-farm Payrolls data from FRED using pandas_datareader
                nfp_data = web.DataReader("PAYEMS", "fred", self.start_date, self.end_date)
                # Rename the column to 'Nonfarm Payrolls' for consistency
                nfp_data = nfp_data.rename(columns={'PAYEMS': 'NonfarmPayrolls'})
                nfp_data['NonfarmPayrollsChange'] = nfp_data['NonfarmPayrolls'].pct_change() * 100
                nfp_data['Month'] = nfp_data.index.month
                nfp_data['Year'] = nfp_data.index.year
                logs.log("Successfully extracted the Non-farm Payrolls data fields")
            except Exception as e:
                raise ValueError(f"Error in extracting the Non-farm Payrolls data fields: {e}")
                logs.log("Something went wrong while extracting the Non-farm Payrolls data fields", level='ERROR')
            logs.log("Successfully generated the macro-economical dataframes")
            return cpi_data, fed_funds_rate, nfp_data
        except Exception as e:
            raise ValueError(f"Error in extracting the macro-economical datasets: {e}")
            logs.log("Something went wrong while extracting the macro-economical datasets", level='ERROR')



