import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle


class AutoMLKit:
    def __init__(self):
        self.data = None
        self.target = None
        self.train_data = None
        self.train_target = None
        self.test_data = None
        self.test_target = None
        self.features = None
        self.selected_features = None
        self.pipe = None
        self.best_model = None

    def load_data(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def preprocess_data(self):
        X = self.data.drop(columns=['target'])
        y = self.data['target']

        # Train-test split
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Data Cleaning
        imputer = SimpleImputer(strategy='mean')
        self.train_data = imputer.fit_transform(self.train_data)
        self.test_data = imputer.transform(self.test_data)

        # Feature Engineering and Selection
        encoder = OneHotEncoder()
        scaler = StandardScaler()
        selector = SelectKBest(score_func=f_classif, k=10)

        encoded_data = encoder.fit_transform(self.train_data)
        scaled_data = scaler.fit_transform(encoded_data.toarray())
        self.train_data = selector.fit_transform(
            scaled_data, self.train_target)

        encoded_data = encoder.transform(self.test_data)
        scaled_data = scaler.transform(encoded_data.toarray())
        self.test_data = selector.transform(scaled_data)

        self.features = selector.get_support(indices=True)

    def select_model(self):
        # Model Selection and Hyperparameter Optimization
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
        grid_search.fit(self.train_data, self.train_target)

        self.best_model = grid_search.best_estimator_

    def train_evaluate_model(self):
        # Model Training and Evaluation
        self.best_model.fit(self.train_data, self.train_target)
        predictions = self.best_model.predict(self.test_data)

        accuracy = accuracy_score(self.test_target, predictions)
        precision = precision_score(self.test_target, predictions)
        recall = recall_score(self.test_target, predictions)
        f1 = f1_score(self.test_target, predictions)

        print("Performance Metrics:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

    def save_model(self, model_file):
        # Save the best model
        with open(model_file, 'wb') as f:
            pickle.dump(self.best_model, f)

    def load_model(self, model_file):
        # Load the saved model
        with open(model_file, 'rb') as f:
            self.best_model = pickle.load(f)

    def deploy_model(self, features):
        # Model Deployment and Prediction
        def predict(data_row):
            transformed_data = data_row[features].values.reshape(1, -1)
            prediction = self.best_model.predict(transformed_data)
            return prediction[0]

        return predict


# Example usage
automl = AutoMLKit()
automl.load_data("data.csv")
automl.preprocess_data()
automl.select_model()
automl.train_evaluate_model()
automl.save_model("best_model.pkl")
automl.load_model("best_model.pkl")

sample_data = pd.DataFrame()  # Add sample data for prediction
prediction_func = automl.deploy_model(automl.features)
prediction = prediction_func(sample_data)
print("Prediction:", prediction)
