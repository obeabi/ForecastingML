"""
This class sets up a financial asset and augments the price action data with technical and economic indicators
"""
import logging
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_datareader.data as web
import numpy as np
import seaborn as sns
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

    def __init__(self, ticker, start_date='2022-01-01', end_date='2026-12-31', interval='1wk'):
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
            elif self.interval == '1wk':
                df = df.copy()
                # Resample daily data to weekly frequency
                df = df.resample('W').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'})
                df = df.copy()

            elif self.interval == '1mo':
                df = df.copy()
                # Resample daily data to weekly frequency
                df = df.resample('ME').agg(
                    {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Adj Close': 'last',
                     'Volume': 'sum'})
            else:
                print("Wrong input!")

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
            # df['Date'] = df.index
            df['Month'] = df.index.month
            df['Year'] = df.index.year

            if self.interval == '1d':
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
                df["is_weekend"] = df.index.to_series().dt.dayofweek > 4
                df["is_week_start"] = df.index.to_series().dt.dayofweek == 0
                df['day_name'] = df.index.to_series().dt.day_name()

            elif self.interval == '1wk':
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
                df['is_month_start'] = df.index.to_series().dt.is_month_start
                df['is_month_end'] = df.index.to_series().dt.is_month_end
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end

            elif self.interval == '1mo':
                df['open-close'] = df['Open'] - df['Close']
                df['low-high'] = df['Low'] - df['High']
                df['MA_10'] = df['Close'].rolling(window=10).mean()
                df['Volatility_10'] = df['Close'].rolling(window=10).std()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['Volatility_20'] = df['Close'].rolling(window=20).std()
                df['Return'] = df['Close'].pct_change()
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
                df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
            logs.log("Successfully enriched data with date fields")
            return df
        except Exception as e:
            raise ValueError(f"Error in enriching data with date fields: {e}")
            logs.log("Something went wrong while enriching the data with date fields", level='ERROR')

    def cursory_stockperformnace_analysis(self):
        """
        Quick analysis of price action at year-end and quarter-end
        :return:
        """
        try:
            df = self.download_price_volume().copy()
            df['year'] = df.index.year
            data_grouped = df.groupby('year').mean()
            plt.figure(figsize=(20, 10))
            for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
                plt.subplot(2, 2, i+1)
                data_grouped[col].plot.bar()
            plt.show()  # Show all distribution plots in one figure
            logs.log("Successfully performed quick analysis of stock performance")
        except Exception as e:
            raise ValueError(f"Error in performing quick analysis of stock performance: {e}")
            logs.log("Something went wrong while performing quick analysis of stock performance", level='ERROR')

    def end_of_quarter_stockperformance(self):
        """
        Quick end of quarter stock performance
        :return:
        """
        try:
            df = self.download_price_volume().copy()
            df['is_quarter_end'] = df.index.to_series().dt.is_quarter_end
            logs.log("Successfully performed end of quarter stock analysis")
            return df.groupby('is_quarter_end').mean()
        except Exception as e:
            raise ValueError(f"Error in performing quick end of quarter analysis of stock performance: {e}")
            logs.log("Something went wrong while performing quick end of quarter analysis of stock performance", level='ERROR')

    def add_technical_indicators(self):
        """
        :return:
        """
        try:
            df = self.enrich_data_date().copy()
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
            df1 = self.add_technical_indicators().copy()
            #columns_keep, _ = self.drop_correlated_features()
            #df1 = df1[columns_keep].copy()
            #df1 = df1.copy()
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
            return merged_df3
        except Exception as e:
            raise ValueError(f"Error in combining the technical and macro-economical datasets: {e}")
            logs.log("Something went wrong while combining the technical and macro-economical datasets", level='ERROR')

    def plot_price_volume_distribution(self):
        """
         Plot distribution of numerical features
        :return:
        """
        try:
            df = self.download_price_volume()
            df = df.copy()
            features = df.columns

            plt.figure(figsize=(20, 10))
            for i, col in enumerate(features):
                plt.subplot(2, 3, i+1)
                sns.distplot(df[col])
            plt.show()  # Show all distribution plots in one figure

            # Plot box plots
            plt.figure(figsize=(20, 10))
            for i, col in enumerate(features):
                plt.subplot(2, 3, i+1)
                sns.boxplot(df[col])
            plt.show()  # Show all box plots in one figure

            logs.log("Successfully rendered distribution plots of price-volume data")
        except Exception as e:
            raise ValueError(f"Error in rendering distribution plots of price-volume data: {e}")
            logs.log("Something went wrong while rendering distribution plots of price-volume data", level='ERROR')

    def plot_target_distribution(self):
        """
        Plots the distribution of the target variable.
        Returns:
        - None
        """
        # Plot count distribution
        try:
            df = self.enrich_data_date().copy()
            plt.figure(figsize=(12, 6))
            plt.pie(df['Target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
            plt.title('Pie Plot : Class Distribution of Target Variable')
            plt.ylabel('')
            plt.tight_layout()
            plt.show()
            logs.log("Successfully rendered distribution plots of target variable")
        except Exception as e:
            raise ValueError(f"Error while rendering distribution plots of target variable : {e}")
            logs.log("An error was raised while rendering distribution plots of target variable", level='ERROR')

    def find_correlated_features(self):
        """
        Plots the correlation heat-map for all features of the stock data
        :return:
        """
        try:
            df = self.add_technical_indicators().copy()
            numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
            plt.figure(figsize=(12, 12))
            sns.heatmap(df[numeric_columns].corr() > 0.8, annot=True)
            plt.show()
            logs.log("Successfully rendered correlation plot for stock dataset")
        except Exception as e:
            raise ValueError(f"Error while rendering correlation plot for stock dataset : {e}")
            logs.log("An error was raised while rendering correlation plot for stock dataset  ", level='ERROR')

    def drop_correlated_features(self):
        """
        Plots the correlation heat-map for all features of the stock data
        :return:
        """
        try:
            df = self.add_technical_indicators().copy()
            numeric_columns = df.select_dtypes(include=[float, int]).columns.tolist()
            x_num = df[numeric_columns].dropna()
            x_num.drop(columns=['Target'], inplace=True)
            correlation_matrix = x_num.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_correlation_pairs = [(column, row) for row in upper_triangle.index for column in upper_triangle.columns if upper_triangle.loc[row, column] > 0.8]
            columns_to_drop = {column for column, row in high_correlation_pairs}
            df_reduced = df.drop(columns=columns_to_drop)
            logs.log("Successfully found the numerical columns to drop from  stock dataset")
            return df_reduced.columns, columns_to_drop
        except Exception as e:
            raise ValueError(f"Error while found the numerical columns to drop from  stock dataset : {e}")
            logs.log("An error was raised while finding the numerical columns to drop from  stock dataset ", level='ERROR')


