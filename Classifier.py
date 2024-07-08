import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Logger import CustomLogger
import warnings

warnings.filterwarnings("ignore")

# Example usage:
logger = CustomLogger()


class MLClassifier:
    """
    Classifier model for ML tasks uses Logistic regression, random forest and XGBoost
    """

    def __init__(self, learning_rate=0.01, max_iter=100, n_estimators=100, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trained_models = {}
        self.preprocessor = None
        self.test_best_model = None
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.best_model = None
        self.results = {}
        self.evaluation_results = {}
        self.best_model_name = None

        # # Initialize models with class_weight or scale_pos_weight to handle imbalanced datasets
        self.models = {
            "LogisticClassifier": LogisticRegression(C=1 / self.learning_rate, max_iter=self.max_iter,
                                                     class_weight='balanced', random_state=self.random_state),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=self.n_estimators,
                                                             random_state=self.random_state, class_weight='balanced'),
            "SupportVectorClassifier": SVC(C=1 / self.learning_rate, kernel='rbf', probability=True, cache_size=300,
                                           class_weight='balanced', verbose=True, max_iter=self.max_iter,
                                           random_state=self.random_state),
            "CatBoostClassifier": CatBoostClassifier(iterations=self.max_iter, learning_rate=self.learning_rate,
                                                     depth=6, loss_function='Logloss', eval_metric='AUC', verbose=10,
                                                     auto_class_weights='Balanced'),
            "XGBClassifier": xgb.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                               random_state=self.random_state, scale_pos_weight=1)
        }

    def drop_target_column(self, df):
        """
        Drop the target column from the DataFrame.

        :param df: pandas DataFrame from which to drop the target column
        :return: DataFrame with the target column dropped
        """
        try:
            df = df.drop(columns=['Target'])
            logger.log("Target column dropped successfully.")
            return df
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the target column to drop: {e}")
            logger.log("Something went wrong while finding the the target column to drop", level='ERROR')

    def find_numericategory_columns(self, X):
        """
        Find numerical and categorical columns from dataset
        :param X:
        :return: numeric and categorical column names
        """
        try:
            numeric_columns = X.select_dtypes(include=[float, int]).columns.tolist()
            categorical_columns = X.select_dtypes(exclude=[float, int]).columns.tolist()
            logger.log('Successfully found numeric and categorical columns from the dataset')
            return numeric_columns, categorical_columns
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the numeric and categorical columns: {e}")
            logger.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def minmax_scale_features(self, X):
        """
        Applies MinMax Scaler normalization

        Parameters
        ----------
        X : features

        Returns
        -------
        normalized features

        """
        try:
            return self.minmax_scaler.fit_transform(X)
            logger.log('Successfully performed Min-Max standardization of dataset')
        except Exception as e:
            raise ValueError(f"Error in normailizing the data using the Min-Max scaler : {e}")
            logger.log("An error was raised while attempting to perform standardization using Min-Max ", level='ERROR')

    def standard_scale_features(self, X):
        """
        Applies Standard Scaler normalization

        Parameters
        ----------
        X : features

        Returns
        -------
        normalized features

        """
        try:
            return self.standard_scaler.fit_transform(X)
            logger.log('Successfully performed Standard normalization of dataset')
        except Exception as e:
            raise ValueError(f"Error in normailizing the data using the Standard scaler method : {e}")
            logger.log("An error was raised while attempting to perform standardization using Standard scaler method ",
                       level='ERROR')

    def vif_multicollinearity(self, X, threshold=10.0):
        """
        Checks for multi-collinearity between features doesn't work well
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features, _ = self.find_numericategory_columns(X)
            x_num = self.pre_process(X[numeric_features])
            vif_data = pd.DataFrame()
            vif_data["feature"] = X[numeric_features].columns
            vif_data["VIF"] = [variance_inflation_factor(x_num, i) for i in range(X[numeric_features].shape[1])]

            # Drop columns with VIF above the threshold
            high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
            x_dropped = X.drop(columns=high_vif_features)
            logger.log("Successfully performed the multi-collinearity check step!")

            return x_dropped.columns, high_vif_features, vif_data
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logger.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    def pre_process(self, X_train, X_test):
        """
        Create data pre-processing pipeline using min-max scaler and one-hot encoding
        :param X_train:
        :param X_test:
        :return:
        """
        try:
            numeric_features, categorical_features = self.find_numericategory_columns(X_train)
            numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            logger.log("Successfully performed the pre-processing step!")
            return X_train_processed, X_test_processed
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logger.log("Something went wrong while pre-processing the dataset", level='ERROR')

    def pre_process_le(self, X_train, X_test):
        """
        Create data pre-processing pipeline using min-max scaler and label encoding
        :param X_train:
        :param X_test:
        :return:
        """
        try:
            numeric_features, categorical_features = self.find_numericategory_columns(X_train)
            numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
            # Define a function to apply LabelEncoder to categorical columns
            def label_encode_column(column):
                le = LabelEncoder()
                return le.fit_transform(column)

            # Create a transformer for categorical features using a custom FunctionTransformer
            categorical_transformer = Pipeline(steps=[('label_encoder', FunctionTransformer(func=lambda col: col.apply(label_encode_column), validate=False))])
            # Create the column transformer
            self.preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

            # Fit and transform the training data, and transform the test data
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            # Convert the transformed arrays back to DataFrames with proper column names
            X_train_processed = pd.DataFrame(X_train_processed, columns=numeric_features + categorical_features)
            X_test_processed = pd.DataFrame(X_test_processed, columns=numeric_features + categorical_features)
            logger.log("Successfully performed the pre-processing step!")
            return X_train_processed, X_test_processed
        except Exception as e:
            raise ValueError(f"Error in preprocessing data: {e}")
            logger.log("Something went wrong while pre-processing the dataset", level='ERROR')

    def fit_models(self, X, y):
        """
         Perform  regression on train-set using specified model
        :param X:
        :param y
        :return: trained model
        """
        trained_models = {}
        try:
            for name, model in self.models.items():
                if name == "CatBoostClassifier":
                    train_pool = Pool(data=X, label=y)
                    model.fit(train_pool)
                else:
                    print(f"Training {name}...")
                    model.fit(X, y)
                print("Success training of model sucessfull!")
                trained_models[name] = model
            logger.log("Successfully trained the models")
            self.trained_models = trained_models
        except Exception as e:
            raise ValueError(f"Error in fitting model: {e}")
            logger.log("Something went wrong while training models", level='ERROR')

    def prediction(self, X, model):
        """
         Perform  prediction on trained model
        :param X:
        :param model
        :return: prediction
        """
        try:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            logger.log("Successfully made predictions from the models")
            return y_pred, y_pred_proba
        except Exception as e:
            raise ValueError(f"Error in making predictions: {e}")
            logger.log("Something went wrong while making predictions", level='ERROR')

    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """
        Evaluate all trained models
        :param y_true:
        :param y_pred:
        :return: roc_auc score
        """
        try:

            roc_auc_scores = roc_auc_score(y_true, y_pred_proba)
            accuracy = accuracy_score(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            logger.log('Successfully saved the evaluation result as a dictionary object')
            return {
                "accuracy": accuracy,
                "roc_auc": roc_auc_scores
                #"confusion_matrix": conf_matrix,
                #"Classification_report": class_report,
                #"fpr": fpr,
                #"tpr": tpr,
                #"thresholds": thresholds

            }
        except Exception as e:
            raise ValueError(f"Error in evaluating model: {e}")
            logger.log("Something went wrong while evaluating model", level='ERROR')

    def select_best_model(self, X_train, X_test, y_train, y_test):
        """
        Select the best ML model based on adjusted roc_auc score
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        try:
            best_score = -np.inf
            best_model_name = None
            best_model = None
            for model_name, model in self.trained_models.items():
                y_train_pred, y_train_pred_proba = self.prediction(X_train, model)
                y_test_pred, y_test_pred_proba = self.prediction(X_test, model)
                train_scores = self.evaluate_model(y_train, y_train_pred, y_train_pred_proba)
                test_scores = self.evaluate_model(y_test, y_test_pred, y_test_pred_proba)

                self.evaluation_results[model_name] = {"Train": train_scores, "Test": test_scores}
                if test_scores["roc_auc"] > best_score:
                    best_score = test_scores["roc_auc"]
                    best_model_name = model_name
                    best_model = model
            self.best_model = best_model
            self.best_model_name = best_model_name
            logger.log("Successfully selected the best model")
            return best_model_name, best_model
        except Exception as e:
            raise ValueError(f"Error in selecting the best model: {e}")
            logger.log("Something went wrong while selecting the best model", level='ERROR')

    def save_best_model(self, file_path='model.pickle'):
        """
        Save the trained ML model as a pickle or sav file.
        :param file_path:
        :return:
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.best_model, file)
            print("\nModel saved as a pickle file successfully!")
            logger.log("Successfully saved the best model")
        except Exception as e:
            raise ValueError(f"Error in saving the best model: {e}")
            logger.log("Something went wrong while saving the best model", level='ERROR')

    def find_optimal_clusters(self, X, max_k=5):
        """
        :param X:
        :param max_k:
        :return:
        """
        try:
            X_scaled = self.scale_features(X)
            self.wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, init="k-means++", n_init=12, random_state=self.random_state)
                kmeans.fit(X_scaled)
                self.wcss.append(kmeans.inertia_)

            diffs = np.diff(self.wcss)
            diffs_ratio = diffs[:-1] / diffs[1:]
            optimal_k = np.argmin(diffs_ratio) + 2  # +2 because of zero-based indexing and the diff shifts results by 1
            logger.log('Successfully found the optimal count of clusters')
            logger.log(optimal_k)
            return optimal_k
        except Exception as e:
            raise ValueError(f"Error while finding the optimal number of clusters : {e}")
            logger.log("An error was raised while finding the optimal number of clusters", level='ERROR')

    def plot_elbow_curve(self, max_k=5):
        """

        :param max_k:
        :return:
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, max_k + 1), self.wcss, marker='o')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.xticks(range(1, max_k + 1))
            plt.grid(True)
            plt.show(block=False)
            logger.log('Successfully plotted the elbow curve  the optimal count of clusters')
        except Exception as e:
            raise ValueError(f"Error while rendering the Elbow plot : {e}")
            logger.log("An error was raised while rendering the Elbow plot", level='ERROR')

    def find_clusters(self, X, n_clusters=4):
        """

        :param X:
        :param n_clusters:
        :return:
        """
        try:
            X_scaled = self.scale_features(X)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=12, random_state=self.random_state)
            logger.log('Successfully trained cluster algorthim with the optimal count of clusters')
            return kmeans.fit_predict(X_scaled)
        except Exception as e:
            raise ValueError(f"Error while finding the number of clusters : {e}")
            logger.log("An error was raised while finding the number of clusters", level='ERROR')

    def split_train_test(self, data, test_size=0.01):
        """

        :param data:
        :param test_size:
        :return:
        """
        try:
            # Ensure the data is sorted by date
            df = data.sort_index()
            # Split the data
            #train_df = df.iloc[:-1]
            #test_df = df.iloc[-1:]
            # Calculate the number of test samples
            n_test = int(len(df) * test_size)
            # Split the data
            train_df = df[:-n_test]
            test_df = df[-n_test:]
            # Separate features and target
            X_train = train_df.drop(columns=['Target'])
            y_train = train_df['Target']
            X_test = test_df.drop(columns=['Target'])
            y_test = test_df['Target']
            logger.log('Successfully splitted the dataset into train-test sets')
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
            return X_train, X_test, y_train, y_test
            # return
        except Exception as e:
            raise ValueError(f"Error while splitting the dataset to train-test sets : {e}")
            logger.log("An error was raised while splitting the dataset to train-test sets", level='ERROR')

    def plot_confusion_matrix(self, X, y):
        try:

            if self.best_model is None:
                raise ValueError("No model has been trained yet. Please train the model before plotting the confusion matrix.")
            y_pred = self.best_model.predict(X)
            cm = confusion_matrix(y, y_pred)

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {self.best_model_name}')
            plt.show(block=False)
        except Exception as e:
            raise ValueError(f"Error while plotting the confusion matrix : {e}")
            logger.log("An error was raised while plotting the confusion matrix", level='ERROR')

    def plot_roc_curve(self, fpr, tpr):
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.show(block=False)
        except Exception as e:
            raise ValueError(f"Error while plotting the Receiver Operating Characteristic (ROC) Curve : {e}")
            logger.log("An error was raised while plotting the Receiver Operating Characteristic (ROC) Curve",
                       level='ERROR')

    def plot_class_distribution(self, df, target):
        """
        Plots the distribution of the target variable.

        Parameters:
        - df: DataFrame, the input dataframe containing the target variable.
        - target: str, the name of the target variable column.

        Returns:
        - None
        """
        # Plot count distribution
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.countplot(x=target, data=df)
            plt.title('Count Plot: Class Distribution of Target Variable')
            plt.xlabel('Class')
            plt.ylabel('Count')

            # Plot pie chart
            plt.subplot(1, 2, 2)
            df[target].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
            plt.title('Pie Plot : Class Distribution of Target Variable')
            plt.ylabel('')

            plt.tight_layout()
            plt.show(block=False)
            logger.log("Successfully rendered distribution plots of target variable")
        except Exception as e:
            raise ValueError(f"Error while rendering distribution plots of target variable : {e}")
            logger.log("An error was raised while rendering distribution plots of target variable", level='ERROR')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hi PyCharm')
