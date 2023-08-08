import pandas as pd
from scipy.stats.mstats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import pickle
import numpy as np
import csv


def write_to_csv(file_name, prob_pcm, ref_name):
    """ Write each possible pair of image with its probability of preference to a file
    """
    CONDITIONS = 16
    with open(file_name+'.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Pair_a', 'Pair_b', 'Probability'])
        for row in range(CONDITIONS):
            for col in range(row+1, CONDITIONS):
                writer.writerow(["ref_"+ref_name+"_Img"+str(row+1), "ref_" +
                                ref_name+"_Img"+str(col+1), prob_pcm[row, col]])


def get_probability(pcm):
    """ A function to convert pcm to probability pcm
    """
    CONDITIONS = 16
    prob_pcm = np.zeros([CONDITIONS, CONDITIONS])
    for i in range(CONDITIONS):
        for j in range(CONDITIONS):
            if(pcm[i, j] == 0):
                prob_pcm[i, j] = 0
            else:
                prob_pcm[i, j] = (pcm[i, j] / (pcm[i, j] + pcm[j, i]))
    return prob_pcm


def normalize_features(X_train):
    """A function to normalize the input featuers
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled


def train_predictor(X_train_scaled, y_train):
    print("Training .....")
    predictor = SVR(kernel='rbf')
    predictor.fit(X_train_scaled, y_train)
    return predictor


def merge(features, pairs_with_probability):
    """This function recieves two dataframe and merge them based on the common features

    Args:
        features: A dataframe of features extreacted from PC-IQA database
        pairs_with_probability: A dataframe of every possible pair of images (pair_a, pair_b) 
        in the training set along with probability of peference.

    Returns:
        dataframes: two dataframs (X_train and y_train) 
    """

    # Merge labeld_pairs, features with the common feature (pair_a)
    first_df = pd.merge(pairs_with_probability, features)
    features.rename({'Pair_a': 'Pair_b'}, axis='columns', inplace=True)
    second_df = pd.merge(pairs_with_probability, features)
    train_set = pd.merge(first_df, second_df, on=[
                         'Pair_a', 'Pair_b'], suffixes=['_a', '_b'])
    train_set.drop(['Probability_b', 'Pair_a', 'Pair_b'], axis=1, inplace=True)
    train_set.rename({'Probability_a': 'Probability'},
                     axis='columns', inplace=True)
    X_train = train_set.drop(['Probability'], axis=1).copy()
    y_train = train_set['Probability'].copy()
    return X_train, y_train


def main():

    # Read train set
    features = pd.read_csv(
        './src/Feature_extraction/JPEGAI-quality-metrics-on-IQA.csv', header=0, sep=",")

    pairs_with_probability = pd.read_csv(
        './src/Training/Predictor_training/All_probability_of_preference.csv', header=0, sep=",")

    # Associate features to each pair
    X_train, y_train = merge(features, pairs_with_probability)

    # Noramalization of the features
    X_train_scaled = normalize_features(X_train)
    y_train = y_train.values

    # Train predictor
    predictor = train_predictor(X_train_scaled, y_train)

    # Save predictor
    filename = 'predictor_test.sav'
    pickle.dump(predictor, open(filename, 'wb'))


if __name__ == '__main__':
    main()
