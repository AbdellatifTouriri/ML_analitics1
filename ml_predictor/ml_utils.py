from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train_model(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Example: Train a RandomForestClassifier
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)

    X = df.drop('target_column', axis=1)
    y = df['target_column']

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    return accuracy
