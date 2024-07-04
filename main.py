from stocks import instrument
import pandas as pd
from ExploratoryAnalysis import extensive_eda

asset = instrument(ticker='AAPL', interval='1d')

data = asset.download_price_volume()
#cpi_data, _, nfp_data = asset.add_macro_indicators()
# Perform EDA using extensive_eda class
eda = extensive_eda()
eda.save_eda_html(data)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = asset.add_technical_indicators()
    df = df.dropna()
    #df = asset.join_technical_macro()
    #cpi_data, fed_funds_rate, nfp_data = asset.add_macro_indicators()
    print(df.columns)
    #print(df.head())
    # print(cpi_data.head())
    # print()
    # print(fed_funds_rate.head())
    #print()
    #print(nfp_data.head())
