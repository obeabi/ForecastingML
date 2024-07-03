from stocks import instrument
import pandas as pd


asset = instrument(ticker='AAPL',  frequency='1d')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = asset.add_technical_indicators()
    cpi_data, fed_funds_rate, nfp_data = asset.add_macro_indicators()
    print(df.columns)
    print()
    print(cpi_data.columns)
    print()
    print(fed_funds_rate.columns)
    print()
    print(nfp_data.columns)


