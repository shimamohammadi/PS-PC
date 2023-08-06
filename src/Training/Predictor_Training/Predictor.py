import pandas as pd
from scipy.stats.mstats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import pickle
from src.data_collection.shared_func import merging


def normalize_features(X_train, X_test):
    """A function to normalize the input featuers
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled


def train_predictor(X_train_scaled, y_train):
    print("Training .....")
    predictor = SVR(kernel='rbf')
    predictor.fit(X_train_scaled, y_train)

    # Save the predictor
    filename = 'Predictor_model.sav'
    pickle.dump(predictor, open(filename, 'wb'))

    return predictor


def main():

    X_test, y_test = pd.read_csv(
        './src/dataset/JPEGAI-quality-metrics-on-IQA.csv', header=0, sep=",")

    # Read train set
    X_train, y_train = pd.read_csv(
        './src/dataset/JPEGAI-quality-metrics-on-IQA.csv', header=0, sep=",")

    # Noramalization of the features
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    y_train = y_train.values
    y_test = y_test.values

    # Train predictor
    predictor = train_predictor(X_train_scaled, y_train)

    # Test predictor with test set
    prediction = predictor.predict(X_test_scaled)
    plcc, _ = pearsonr(prediction, y_test)
    print(plcc)


if __name__ == '__main__':
    main()
