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
from sklearn.impute import KNNImputer
from Logger import CustomLogger

# Create a KNNImputer object with a specified number of neighbors
knn_imputer = KNNImputer(n_neighbors=5)
logs = CustomLogger()


class instrument:
    """
        This class object fetches xxxxx and xxx data
    """

    def __init__(self, ticker, start_date='2023-01-01', end_date='2026-12-31', interval='1wk'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval  # '15m', '90m' , '1h', '1d', '1wk', '1mo', '3mo'

    def download_price_volume(self):
        """

        :return: returns dataframe of asset
        """
        try:
            df = yf.download(self.ticker, self.start_date, self.end_date, self.interval)
            if self.interval == '1d':
                df = df.copy()
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            elif self.interval == '1wk':
                df = df.copy()
                # Resample daily data to weekly frequency
                df = df.resample('W').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'})
                df = df.copy()
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

            elif self.interval == '1mo':
                df = df.copy()
                # Resample daily data to weekly frequency
                df = df.resample('ME').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'})
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            else:
                print("Wrong input!")

            logs.log("Successfully downloaded price-action data from yahoo finance API")
            return df[
                ['Open', 'High', 'Low', 'Close', 'Volume', 'open-close', 'low-high', 'MA_10', 'Volatility_10', 'MA_20',
                 'Volatility_20', 'Return', 'Target']]
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
            # df['Date'] = df.index
            df['Month'] = df.index.month
            df['Year'] = df.index.year

            if self.interval == '1d':
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
                df["is_weekend"] = df.index.to_series().dt.dayofweek > 4
                df["is_week_start"] = df.index.to_series().dt.dayofweek == 0
                df['day_name'] = df.index.to_series().dt.day_name()

            elif self.interval == '1wk':
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end

            elif self.interval == '1mo':
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
            # Calculate ADX, +DI, and -DI
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            # Append the ADX, +DI, and -DI to the original DataFrame
            df = df.join(adx)
            # Fetch VIX data from Yahoo Finance using yfinance
            vix_data = yf.download("^VIX", self.start_date, self.end_date, self.interval)
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
                logs.log("Successfully extracted CPI data fields")
            except Exception as e:
                raise ValueError(f"Error while extracting CPI data fields: {e}")
                logs.log("Something went wrong while extracting the CPI data fields", level='ERROR')

            try:
                # Fetch Federal Funds Rate data from FRED
                fed_funds_rate = web.DataReader("FEDFUNDS", "fred", self.start_date, self.end_date)
                logs.log("Successfully extracted the Fed fund rates data fields")
            except Exception as e:
                raise ValueError(f"Error while extracting the Fed Fund rates data fields: {e}")
                logs.log("Something went wrong while extracting the Fed Fund rates data fields", level='ERROR')

            try:
                # Fetch Non-farm Payrolls data from FRED using pandas_datareader
                nfp_data = web.DataReader("PAYEMS", "fred", self.start_date, self.end_date)
                # Rename the column to 'Nonfarm Payrolls' for consistency
                nfp_data = nfp_data.rename(columns={'PAYEMS': 'NonfarmPayrolls'})
                logs.log("Successfully extracted the Non-farm Payrolls data fields")
            except Exception as e:
                raise ValueError(f"Error in extracting the Non-farm Payrolls data fields: {e}")
                logs.log("Something went wrong while extracting the Non-farm Payrolls data fields", level='ERROR')
            logs.log("Successfully generated the macro-economical dataframes")
            return cpi_data, fed_funds_rate, nfp_data
        except Exception as e:
            raise ValueError(f"Error in extracting the macro-economical datasets: {e}")
            logs.log("Something went wrong while extracting the macro-economical datasets", level='ERROR')

    def join_technical_macro(self):
        """
        :return:
        """
        try:
            df1 = self.add_technical_indicators()
            df1 = df1.copy()
            cpi_data, fed_funds_rate, nfp_data = self.add_macro_indicators()
            cpi_data, fed_funds_rate, nfp_data = cpi_data.copy(), fed_funds_rate.copy(), nfp_data.copy()

            if self.interval == '1d':
                # Reindex df2 to have all the dates in the range of df
                all_dates = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='D')

                # CPI Data
                cpi_data = cpi_data.reindex(all_dates)
                # Apply the imputer to the CPI column
                cpi_data['CPIAUCNS'] = knn_imputer.fit_transform(cpi_data[['CPIAUCNS']])
                # Reset the index to have the date as a column again
                cpi_data.reset_index(inplace=True)
                cpi_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                df1.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df = pd.merge(df1, cpi_data, on='Date', how='left')

                # Fed funds rate
                fed_funds_rate = fed_funds_rate.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                fed_funds_rate['FEDFUNDS'] = knn_imputer.fit_transform(fed_funds_rate[['FEDFUNDS']])
                # Reset the index to have the date as a column again
                fed_funds_rate.reset_index(inplace=True)
                fed_funds_rate.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df2 = pd.merge(merged_df, fed_funds_rate, on='Date', how='left')

                # Non-farm Payrolls data
                nfp_data = nfp_data.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                nfp_data['NonfarmPayrolls'] = knn_imputer.fit_transform(nfp_data[['NonfarmPayrolls']])
                # Reset the index to have the date as a column again
                nfp_data.reset_index(inplace=True)
                nfp_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df2.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df3 = pd.merge(merged_df2, nfp_data, on='Date', how='left')

            elif self.interval == '1wk':
                # Reindex df2 to have all the dates in the range of df
                all_dates = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='W')
                cpi_data = cpi_data.reindex(all_dates)
                # Apply the imputer to the CPI column
                cpi_data['CPIAUCNS'] = knn_imputer.fit_transform(cpi_data[['CPIAUCNS']])
                # Reset the index to have the date as a column again
                cpi_data.reset_index(inplace=True)
                cpi_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                df1.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df = pd.merge(df1, cpi_data, on='Date', how='left')
                # Set the date as the index again
                merged_df.set_index('Date', inplace=True)

                # Fed funds rate
                fed_funds_rate = fed_funds_rate.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                fed_funds_rate['FEDFUNDS'] = knn_imputer.fit_transform(fed_funds_rate[['FEDFUNDS']])
                # Reset the index to have the date as a column again
                fed_funds_rate.reset_index(inplace=True)
                fed_funds_rate.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df2 = pd.merge(merged_df, fed_funds_rate, on='Date', how='left')
                # Set the date as the index again
                merged_df2.set_index('Date', inplace=True)

                # Non-farm Payrolls data
                nfp_data = nfp_data.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                nfp_data['NonfarmPayrolls'] = knn_imputer.fit_transform(nfp_data[['NonfarmPayrolls']])
                # Reset the index to have the date as a column again
                nfp_data.reset_index(inplace=True)
                nfp_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df2.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df3 = pd.merge(merged_df2, nfp_data, on='Date', how='left')
                # Set the date as the index again
                merged_df3.set_index('Date', inplace=True)

            elif self.interval == '1mo':
                # Reindex df2 to have all the dates in the range of df
                all_dates = pd.date_range(start=df1.index.min(), end=df1.index.max(), freq='M')
                cpi_data = cpi_data.reindex(all_dates)
                # Apply the imputer to the CPI column
                cpi_data['CPIAUCNS'] = knn_imputer.fit_transform(cpi_data[['CPIAUCNS']])
                # Reset the index to have the date as a column again
                cpi_data.reset_index(inplace=True)
                cpi_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                df1.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df = pd.merge(df1, cpi_data, on='Date', how='left')
                # Set the date as the index again
                merged_df.set_index('Date', inplace=True)

                # Fed funds rate
                fed_funds_rate = fed_funds_rate.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                fed_funds_rate['FEDFUNDS'] = knn_imputer.fit_transform(fed_funds_rate[['FEDFUNDS']])
                # Reset the index to have the date as a column again
                fed_funds_rate.reset_index(inplace=True)
                fed_funds_rate.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df2 = pd.merge(merged_df, fed_funds_rate, on='Date', how='left')
                # Set the date as the index again
                merged_df2.set_index('Date', inplace=True)

                # Non-farm Payrolls data
                nfp_data = nfp_data.reindex(all_dates)
                # Apply the imputer to the Fed funds rate column
                nfp_data['NonfarmPayrolls'] = knn_imputer.fit_transform(nfp_data[['NonfarmPayrolls']])
                # Reset the index to have the date as a column again
                nfp_data.reset_index(inplace=True)
                nfp_data.rename(columns={'index': 'Date'}, inplace=True)
                # Reset index for df to prepare for merge
                merged_df2.reset_index(inplace=True)
                # Merge the daily stock price data with the imputed CPI data
                merged_df3 = pd.merge(merged_df2, nfp_data, on='Date', how='left')
                # Set the date as the index again
                merged_df3.set_index('Date', inplace=True)

            column_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'open-close', 'low-high', 'RSI_14', 'ATRr_14',
                            'Return', 'VIX', 'Target', 'CPIAUCNS', 'FEDFUNDS', 'NonfarmPayrolls']
            logs.log("Successfully enriched data with date fields")
            return merged_df3[column_names]
        except Exception as e:
            raise ValueError(f"Error in combining the technical and macro-economical datasets: {e}")
            logs.log("Something went wrong while combining the technical and macro-economical datasets", level='ERROR')
