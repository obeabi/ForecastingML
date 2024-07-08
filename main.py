from stocks import instrument
import pandas as pd
import matplotlib.pyplot as plt
from ExploratoryAnalysis import extensive_eda
from Classifier import MLClassifier

# Object of class stocks
asset = instrument(ticker='TSLA', interval='1d')
# Object of class MLClassifier
model = MLClassifier()

data = asset.download_price_volume()
#cpi_data, _, nfp_data = asset.add_macro_indicators()
# Perform EDA using extensive_eda class
#eda = extensive_eda()
#eda.save_eda_html(data)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = asset.add_technical_indicators()   # use join_technical_macro method when macro-economic data is fixed
    #cpi_data, fed_funds_rate, nfp_data = asset.add_macro_indicators()
    #asset.cursory_stockperformnace_analysis()
    columns_keep, _ = asset.drop_correlated_features()
    df = df[columns_keep].dropna()
    X_train, X_test, y_train, y_test = model.split_train_test(df, test_size=0.01)
    x_train_sc, x_test_sc = model.pre_process(X_train, X_test)
    X = pd.concat([pd.DataFrame(x_train_sc), pd.DataFrame(x_test_sc)], axis=0)
    model.fit_models(x_train_sc, y_train)
    best_model_name, best_model = model.select_best_model(x_train_sc, x_test_sc, y_train, y_test)
    print("\nThe best model to put in production based on test-set is: ", best_model_name)
    print("\nThe evaluation metrics from the best model based on train set are :", model.evaluation_results[best_model_name]["Train"])
    print("\nThe evaluation metrics from the best model based on test set are :", model.evaluation_results[best_model_name]["Test"])
    #model.save_best_model()
    model.plot_confusion_matrix(x_test_sc, y_test)
    df['Predicted_Signal'], _ = model.prediction(X, model.models[best_model_name])
    # Calculate daily returns
    df['Close'] = df['Open'] - df['open-close']
    df['Return'] = df.Close.pct_change()
    # Calculate strategy returns
    df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
    # Calculate Cumulative returns
    df['Cum_Ret'] = df['Return'].cumsum()
    # Plot Strategy Cumulative returns
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    # Create a new figure
    plt.figure(figsize=(12, 6))
    # Plot cumulative returns
    plt.plot(df['Cum_Ret'], color='red', label='Cumulative Return')
    # Plot cumulative strategy returns
    plt.plot(df['Cum_Strategy'], color='blue', label='Cumulative Strategy')
    # Add a title to the plot
    plt.title('Cumulative Returns and Strategy Returns')
    # Add a label to the x-axis
    plt.xlabel('Time')
    # Add a label to the y-axis
    plt.ylabel('Cumulative Return')
    # Add a legend to the plot
    plt.legend()
    # Show the plot
    plt.show()
    print(df.head())
    # print()
    # print(fed_funds_rate.head())
    #print()
    #print(nfp_data.head())
