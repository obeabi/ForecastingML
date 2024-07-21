import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, \
    FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
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
                                                     depth=6, loss_function='Logloss', eval_metric='Accuracy',
                                                     verbose=10,
                                                     auto_class_weights='Balanced'),
            "XGBClassifier": xgb.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                               random_state=self.random_state, scale_pos_weight=1)
        }

    def correlation_multicollinearity(self, X, threshold=0.9):
        """
        Checks for multi-collinearity between features using pearson correlation
        :param X:
        :param threshold:
        :return: non-collinear features
        """
        try:
            numeric_features = X.select_dtypes(include=[float, int]).columns.tolist()
            x_num = X[numeric_features].dropna()
            correlation_matrix = x_num.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_correlation_pairs = [(column, row) for row in upper_triangle.index for column in upper_triangle.columns
                                      if upper_triangle.loc[row, column] > threshold]
            columns_to_drop = {column for column, row in high_correlation_pairs}
            df_reduced = X.drop(columns=columns_to_drop)
            logger.log("Successfully dropped mult-collinear columns!")
            return df_reduced.columns, columns_to_drop
        except Exception as e:
            raise ValueError(f"Error in checking multi-collinearity: {e}")
            logger.log("Something went wrong while checking multi-collinearity:", level='ERROR')

    def preprocessor_fit(self, X, one_hot_encode_cols=None, label_encode_cols=None):
        """
        Fit the preprocessor on the data.

        Args:
            X (pd.DataFrame): Input data containing both numerical and categorical columns.
            one_hot_encode_cols (list): List of categorical columns to one-hot encode.
            label_encode_cols (list): List of categorical columns to label encode.
        """
        try:
            self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            transformers = []

            if self.numerical_cols:
                num_pipeline = Pipeline([
                    ('num_imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('num', num_pipeline, self.numerical_cols))

            if self.categorical_cols:
                if one_hot_encode_cols:
                    cat_pipeline = Pipeline([
                        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])
                    transformers.append(('cat_onehot', cat_pipeline, one_hot_encode_cols))

                if label_encode_cols:
                    for col in label_encode_cols:
                        transformers.append((f'{col}_label', FunctionTransformer(self.label_encode), [col]))

            self.preprocessing_pipeline = ColumnTransformer(transformers, remainder='passthrough')
            self.preprocessing_pipeline.fit(X)
            self.fit_status = True
            self.feature_names_out = self.get_feature_names_out()
            logger.log("Successfully fitted the pre-processing pipeline!")
        except Exception as e:
            logger.log(f"Error during fit: {str(e)}")

    def preprocessor_transform(self, X):
        """
        Transform the input data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Input data to transform.

        Returns:
            pd.DataFrame: Transformed data with original column names.
        """
        try:
            if not self.fit_status:
                raise ValueError("Preprocessor must be fit on data before transforming.")
            transformed_data = self.preprocessing_pipeline.transform(X)
            transformed_df = pd.DataFrame(transformed_data, columns=self.feature_names_out)
            logger.log("Successfully transformed the dataset using the pre-processing pipeline")
            return transformed_df
        except Exception as e:
            logger.log(f"Error during transform: {str(e)}")

    def get_feature_names_out(self):
        """
        Get feature names after transformation.

        Returns:
            list: List of feature names after transformation.
        """
        try:
            if self.preprocessing_pipeline is None:
                return []

            feature_names_out = []
            for name, trans, column_names in self.preprocessing_pipeline.transformers_:
                if trans == 'drop' or trans == 'passthrough':
                    continue
                if isinstance(trans, Pipeline):
                    if name.startswith('cat_onehot') and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                        feature_names_out.extend(trans.named_steps['onehot'].get_feature_names_out())
                    else:
                        feature_names_out.extend(column_names)
                elif isinstance(trans, FunctionTransformer):
                    feature_names_out.extend(column_names)
                else:
                    feature_names_out.extend(column_names)
            logger.log("Successfully retrieved features name!")
            return feature_names_out

        except Exception as e:
            logger.log(f"Error during get_feature_names_out: {str(e)}")

    def label_encode(self, X):
        """
        Apply label encoding to the input data.

        Args:
            X (pd.Series or pd.DataFrame): Input data to encode.

        Returns:
            np.ndarray: Label encoded data reshaped to 2D.
        """
        try:
            le = LabelEncoder()
            logger.log("Successfully performed label encoding!")
            return le.fit_transform(X.squeeze()).reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Something went wrong while performing label encoding: {e}")
            logger.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def find_numericategory_columns(self, X):
        """
        Find numerical and categorical columns from dataset
        :param X:
        :return: numeric and categorical column names
        """
        try:
            numeric_cols = X.select_dtypes(include=[float, int]).columns.tolist()
            categoric_cols = X.select_dtypes(exclude=[float, int]).columns.tolist()
            logger.log("Successfully extracted numerical and categorical columns!")

            return numeric_cols, categoric_cols
        except Exception as e:
            raise ValueError(f"Something went wrong while finding the numeric and categorical columns: {e}")
            logger.log("Something went wrong while finding the numeric and categorical columns", level='ERROR')

    def fit_all_models(self, X, y):
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
                print("Success training of model successfull!")
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
            }
        except Exception as e:
            raise ValueError(f"Error in evaluating model: {e}")
            logger.log("Something went wrong while evaluating model", level='ERROR')

    def select_best_model(self, X_train, X_test, y_train, y_test):
        """
        Select the best ML model based on accuracy score
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
                if test_scores["accuracy"] > best_score:
                    best_score = test_scores["accuracy"]
                    best_model_name = model_name
                    best_model = model
            self.best_model = best_model
            self.best_model_name = best_model_name
            logger.log("Successfully selected the best model")
            return best_model_name, best_model
        except Exception as e:
            raise ValueError(f"Error in selecting the best model: {e}")
            logger.log("Something went wrong while selecting the best model", level='ERROR')

    def tune_parameters(self, model_name, X, y, cv=5, scoring='accuracy'):
        """
        Perform parameter tuning of parameters using Grid Search CV
        :param model_name:
        :param X:
        :param y:
        :param cv:
        :param scoring: ''accuracy', 'roc_auc'
        :return:

        """
        try:
            # Define models and their parameter grids
            models = {'LogisticClassifier': (LogisticRegression(max_iter=self.max_iter,
                                                                class_weight='balanced',
                                                                random_state=self.random_state), {
                                                 'C': [0.01, 0.1, 1, 10, 100, 200, 500],
                                                 'solver': ['newton-cg', 'lbfgs', 'liblinear']
                                             }),

                      'RandomForestClassifier': (
                          RandomForestClassifier(random_state=self.random_state, class_weight='balanced'), {
                              'n_estimators': [100, 200, 500, 1000],
                              'max_depth': [None, 10, 20, 30],
                              'min_samples_split': [2, 5, 10]}),

                      'SupportVectorClassifier': (SVC(probability=True, cache_size=300,
                                                      class_weight='balanced', verbose=True, max_iter=self.max_iter,
                                                      random_state=self.random_state), {
                                                      'C': [0.1, 1, 10, 100, 200],
                                                      'gamma': ['scale', 'auto'],
                                                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}),

                      'XGBClassifier': (
                      xgb.XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                        random_state=self.random_state, scale_pos_weight=1), {
                          'max_depth': [3, 4, 5, 7,10]}),

                      'CatBoostClassifier': (
                      CatBoostClassifier(loss_function='Logloss', eval_metric='Accuracy', verbose=10,
                                         auto_class_weights='Balanced'), {
                          'iterations': [10, 50, 100, 200, 500],
                          'learning_rate': [0.001, 0.01, 0.1, 0.2],
                          'depth': [3, 4, 5, 7]})
                      }

            # Select the model and parameter grid
            model, param_grid = models[model_name]
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=cv)
            grid_search.fit(X, y)
            # Get the best parameters and the best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            logger.log("Successfully tuned the parameters using GridSearchCV")
            return grid_search.best_estimator_, best_params, best_score
        except Exception as e:
            raise ValueError(f"Error in tuning parameters using GridSearchCV: {e}")
            logger.log("Something went wrong while tuning parameters using GridSearchCV", level='ERROR')

    def cross_validate_models(self, model_name, X, y, n_splits=5):
        """
        Perform cross validation of specified model
        :param X:
        :param y:
        :param n_splits:
        :param model_name:
        :return:
        """
        try:
            # Perform grid search to get the best model
            best_model, best_params, best_score = self.tune_parameters(model_name, X, y)
            # Initialize KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            accuracy_scores = cross_val_score(best_model, X, y, cv=kf, scoring='accuracy')
            roc_auc_scores = cross_val_score(best_model, X, y, cv=kf, scoring='roc_auc')
            logger.log("Successfully performed cross-validation")
            return np.mean(accuracy_scores), np.std(roc_auc_scores)
        except Exception as e:
            raise ValueError(f"Error in cross-validating models: {e}")
            logger.log("Something went wrong while performing KFold cross validation", level='ERROR')

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

    def plot_confusion_matrix(self, X, y):
        try:

            if self.best_model is None:
                raise ValueError(
                    "No model has been trained yet. Please train the model before plotting the confusion matrix.")
            y_pred = self.best_model.predict(X)
            cm = confusion_matrix(y, y_pred)

            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix for {self.best_model_name}')
            plt.savefig(f'confusion_matrix_{self.best_model_name}.png')
            #plt.show(block=False)
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
