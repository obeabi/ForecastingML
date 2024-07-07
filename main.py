from stocks import instrument
import pandas as pd
from ExploratoryAnalysis import extensive_eda
from Classifier import MLClassifier

# Object of class stocks
asset = instrument(ticker='AAPL', interval='1d')
# Object of class MLClassifier
model = MLClassifier()

#data = asset.download_price_volume()
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
    X_train, X_test, y_train, y_test = model.split_train_test(df, test_size=0.2)
    x_train_sc, x_test_sc = model.pre_process(X_train, X_test)
    model.fit_models(x_train_sc, y_train)
    best_model_name, best_model = model.select_best_model(x_train_sc, x_test_sc, y_train, y_test)
    print("\nThe best model to put in production based on test-set is: ", best_model_name)
    print("\nThe evaluation metrics from the best model based on train set are :", model.evaluation_results[best_model_name]["Train"])
    print("\nThe evaluation metrics from the best model based on test set are :", model.evaluation_results[best_model_name]["Test"])
    model.save_best_model()
    model.plot_confusion_matrix(x_test_sc, y_test)
    #df = asset.join_technical_macro()

    # print(cpi_data.head())
    # print()
    # print(fed_funds_rate.head())
    #print()
    #print(nfp_data.head())
