from stocks import instrument
import pandas as pd
import matplotlib.pyplot as plt
from ExploratoryAnalysis import extensive_eda
from Classifier import MLClassifier
from FeatureEngineering import clean

# Object of class stocks
asset = instrument(ticker='GOOGL', interval='1d')
# Object of class MLClassifier
model = MLClassifier()

df = asset.add_technical_indicators()
X = df.drop(columns=['Target'])
y = df['Target']
# Perform EDA using extensive_eda class
# eda = extensive_eda()
# eda.save_eda_html(data)


# # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fe = clean()
    print(fe)
    # best_columns, _ = fe.correlation_multicollinearity(X)
    best_columns, columns_to_drop = model.correlation_multicollinearity(X)
    print(f"\nImportant columns after performing the mult-collinearity test using VIF approach are :{best_columns}\n")
    numerical_cols, category_cols = model.find_numericategory_columns(X[best_columns])
    print("\nThe important numeric columns are :", numerical_cols)
    print("\nThe important categorical columns are :", category_cols)
    # label_encode_cols = ['day_name']
    # one_hot_encode_cols = ['is_month_start', 'is_month_end', 'is_quarter_end', 'is_week_start']
    X_best = X[best_columns].copy()
    X_train, X_test, y_train, y_test = fe.split_train_test(X_best, y, test_size=0.01)
    print("\nThe length of test set is : ", len(y_test))
    model.preprocessor_fit(X_train, one_hot_encode_cols=category_cols, label_encode_cols=None)
    # fe.preprocessor_fit(X_train, one_hot_encode_cols=None, label_encode_cols=category_cols)
    # Transform X_train
    X_train_transformed = model.preprocessor_transform(X_train).values
    print(X_train_transformed)

    # Train the Knowledge model
    model.fit_all_models(X_train_transformed, y_train)

    # Prediction
    X_test_transformed = model.preprocessor_transform(X_test).values
    best_model_name, best_model = model.select_best_model(X_train_transformed, X_test_transformed, y_train, y_test)
    print("\nThe best model to put in production based on test-set is: ", best_model_name)
    print(f"\nThe evaluation metrics from the {best_model_name} based on train set are :",
          model.evaluation_results[best_model_name]["Train"])
    print(f"\nThe evaluation metrics from the {best_model_name} based on test set are :",
          model.evaluation_results[best_model_name]["Test"])

    # Perform GridSearchCV for best model (Optional)
    # gridsearch_best_model, best_params, best_score = model.tune_parameters(best_model_name, X_train_transformed, y_train, cv=5)
    # print(gridsearch_best_model)
    # print("\nThe best parameters are :", best_params)
    # print("\nThe best score is :", best_score)

    # # Perform KFold Cross validation (Optional)
    # #accuracy_score, roc_auc_score = model.cross_validate_models(best_model_name, X_train_transformed, y_train)
    # #print("\nThe accuracy score is :", accuracy_score)
    # #print("\nThe roc auc score is :", roc_auc_score)

    # Final outputs
    model.save_best_model()
    model.plot_confusion_matrix(X_test_transformed, y_test)
    X_best_transformed = model.preprocessor_transform(X_best).values
    df['Predicted_Signal'], _ = model.prediction(X_best_transformed, model.models[best_model_name])
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
    plt.savefig(f'Cumulative Returns and Strategy Returns_{best_model_name}.png')

    # Predict Next day price direction
    X_nd = asset.generate_next_day_data(X_best)
    X_nd_transformed = model.preprocessor_transform(X_nd).values
    print(X_nd_transformed)
    # Predict the direction
    y_pred, y_pred_proba = model.prediction(X_nd_transformed, model.models[best_model_name])
    print(y_pred)
    #
    # Interpret the prediction
    if y_pred[0] == 1:
        print("The model predicts that the stock price will go up tomorrow.")
    else:
        print("The model predicts that the stock price will go down tomorrow.")

    # Print prediction probabilities
    print(f"Probability of going up: {y_pred_proba[0]:.2f}")
