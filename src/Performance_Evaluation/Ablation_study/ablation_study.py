import csv
import pickle
from itertools import product
from random import sample

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.io
from scipy import linalg
from scipy.stats.mstats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale

from src.tools import get_pcm, read_matlab_data, run_modeling_Bradley_Terry


def frmwrk_with_clf_only(gth_prob_pcm, gth_judgments_pcm, X_test_transformed):
    """The pcm is filled from the classifers' output. 
    The groundruth data is used for defer pairs, and 0.5 is used for predict pairs 
    """
    clf_only_pcm, trial_num = clf_only(
        gth_prob_pcm, gth_judgments_pcm, X_test_transformed)

    scores_model = infer_scores(clf_only_pcm)

    return scores_model, trial_num


def frmwrk_w_predictor_only(X_test_transformed):

    prediction_pcm = predictor_only(X_test_transformed)
    scores_model = infer_scores(prediction_pcm)
    return scores_model


def frmwrk_with_rnd_clf_with_predictor(gth_prob_pcm, gth_judgments_pcm, X_test_transformed, num_defer_pairs):
    """ Random classifier and predictor
    """
    CONDITIONS = 16
    num_defer_pairs = 25

    # Apply predictor
    prediction_pcm = predictor_only(X_test_transformed)

    # Apply random classifeir
    marked_matrix = random_clf(
        gth_prob_pcm, gth_judgments_pcm, prediction_pcm, num_defer_pairs)

    # Update the prediction pcm
    for row in range(CONDITIONS):
        for col in range(CONDITIONS):
            if (marked_matrix[row, col] == False):
                prediction_pcm[row, col] = gth_prob_pcm[row, col]

    scores_model = infer_scores(prediction_pcm)
    return scores_model


def frmwrk_ps_pc(gth_prob_pcm, gth_judgments_pcm, X_test_transformed):
    """ Framwwork = Predictor + classiifer
    """

    CONDITIONS = 16

    # Apply predictor
    prediction_pcm = predictor_only(X_test_transformed)

    # Apply classifer
    clf_only_pcm, trial_num = clf_only(
        gth_prob_pcm, gth_judgments_pcm,  X_test_transformed)

    # Update the prediction pcm using the classifer output
    for row in range(CONDITIONS):
        for col in range(CONDITIONS):
            if (clf_only_pcm[row, col] == False):
                prediction_pcm[row, col] = gth_prob_pcm[row, col]

    scores_model = infer_scores(prediction_pcm)
    return scores_model


def read_gth(ref_name):

    CONDITIONS = 16
    raw_data = read_matlab_data('IQA', ref_name)
    temp_pcm = get_pcm(CONDITIONS, raw_data)

    gth_pcm = np.zeros((CONDITIONS, CONDITIONS))
    for i in range(CONDITIONS):
        for j in range(CONDITIONS):
            if (temp_pcm[i, j] == 0):
                gth_pcm[i, j] = 0
            else:
                gth_pcm[i, j] = (temp_pcm[i, j] /
                                 (temp_pcm[i, j] + temp_pcm[j, i]))

    # Get scores for the gth_pcm
    scores_gth = infer_scores(gth_pcm)

    return gth_pcm, scores_gth


def predictor_only(X_test_transformed):
    """ Apply predictor, and return a pcm
    """
    CONDITIONS = 16
    # Apply predictor
    # Predictor output. A vector not a pcm
    predictions = predict(X_test_transformed)
    # Get pcm
    prediction_pcm = np.zeros([CONDITIONS, CONDITIONS])

    # Fill the matrix from predictor output
    ind_upper = np.triu_indices(CONDITIONS, 1)
    predictor_results = np.squeeze(predictor_results)
    prediction_pcm[ind_upper] = predictions
    for row in range(CONDITIONS):
        for col in range(row+1, CONDITIONS):
            prediction_pcm[col, row] = 1 - prediction_pcm[row, col]
    return prediction_pcm


def clf_only(gth_prob_pcm, gth_judgments_pcm, X_test_transformed):
    """The pcm is filled from the classifers' output. 
    The groundruth data is used for defer pairs, and 0.5 is used for predict pairs 
    """
    CONDITIONS = 16
    trial_num = 0
    clf_only_pcm = np.zeros([CONDITIONS, CONDITIONS])
    clf_output = np.zeros([CONDITIONS, CONDITIONS])

    clf_only_pcm[clf_only_pcm == 0] = 0.5
    for i in range(CONDITIONS):
        for j in range(CONDITIONS):
            if (i == j):
                clf_only_pcm[i, j] = 0

    # Fill the matrix from Classifeir
    clf = pickle.load(open('./clf_model.sav', 'rb'))
    cls_results = clf.predict(X_test_transformed)
    ind_upper = np.triu_indices(CONDITIONS, 1)
    clf_output[ind_upper] = cls_results
    ind_lower = np.tril_indices(CONDITIONS, 0)
    clf_output[ind_lower] = np.nan

    # Copy
    for row in range(CONDITIONS):
        for col in range(row+1, CONDITIONS):
            if (clf_output[row, col] == False):
                clf_only_pcm[row, col] = gth_prob_pcm[row, col]
                clf_only_pcm[col, row] = gth_prob_pcm[col, row]
                trial_num += gth_judgments_pcm[row,
                                               col] + gth_judgments_pcm[col, row]

    return clf_only_pcm, trial_num


def infer_scores(pcm):
    [scores_full, _, _, prob_full, _, _] = run_modeling_Bradley_Terry(pcm)
    return prob_full


def random_clf(gth_pcm, num_defer_pairs):
    """Input: The pcm of the predictor output
    output: randomly select some pairs to defer and update the prediction_pcm of the selected pairs from the groundtruth data
    """

    CONDITIONS = 16
    marked_matrix = np.eye(CONDITIONS, dtype=bool)
    marked_matrix[marked_matrix == False] = True

    # Random classifier
    count_defer = ((np.count_nonzero(marked_matrix)) - CONDITIONS)
    while (count_defer != num_defer_pairs):
        # Generate two random number
        pairs_to_remove = sample(
            list(product(range(CONDITIONS), repeat=2)), k=1)

        # Check if the random pair was not selected previously, and it's not on the diagonal matrix
        if (marked_matrix[pairs_to_remove[0][0], pairs_to_remove[0][1]] == True and
                pairs_to_remove[0][0] != pairs_to_remove[0][1]):

            # Mark the random pair as selected
            marked_matrix[pairs_to_remove[0][0],
                          pairs_to_remove[0][1]] = False
            marked_matrix[pairs_to_remove[0][1],
                          pairs_to_remove[0][0]] = False
            count_defer = 120 - \
                ((np.count_nonzero(marked_matrix)) - CONDITIONS) / 2
        else:
            pairs_to_remove = sample(
                list(product(range(CONDITIONS), repeat=2)), k=1)

    return marked_matrix


def normalize_features(data):
    """A function to normalize data
    """
    sc = MinMaxScaler()
    data_transformed = pd.DataFrame(sc.fit_transform(data))
    return data_transformed


def predict(X_test_transformed):
    predictor = pickle.load(open('./predictor_model.sav', 'rb'))
    predictions = predictor.predict(X_test_transformed)

    # normalize predictions
    predictions = normalize_features(predictions)

    return predictions


def calculate_correlations(scores_gth, scores_model):
    """Calculate the corelations between the groundtruth scores and the model's scores
    """
    plcc, _ = pearsonr(scores_gth, scores_model)
    srocc, _ = spearmanr(scores_gth, scores_model)

    return plcc, srocc


def run_frmwrk_clf_only():
    """classifer output is used
    predictor is not used
    """
    references = [df_test, df_test, df_test, df_test, df_test]
    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    all_ref_trial_num = 0
    for ref in references:
        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth('data1')

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores, trial_num = frmwrk_with_clf_only(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)
        all_ref_trial_num += trial_num

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    df_test = pd.read_csv('./src/ref_data1.csv', header=0, sep=",")
    return plcc, srocc, all_ref_trial_num


def run_frmwrk_predictor_only():
    """classifer is not used
    predictor is used
    """
    references = [df_test, df_test, df_test, df_test, df_test]
    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    for ref in references:
        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth('data1')

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores = frmwrk_w_predictor_only(X_test_transformed)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    df_test = pd.read_csv('./src/ref_data1.csv', header=0, sep=",")
    return plcc, srocc


def run_frmwrk_with_rnd_clf_with_predictor():
    """classifer is not used
    predictor is used
    """
    num_defer_pairs = 25
    references = [df_test, df_test, df_test, df_test, df_test]
    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    for ref in references:
        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth('data1')

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores = frmwrk_with_rnd_clf_with_predictor(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed, num_defer_pairs)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    df_test = pd.read_csv('./src/ref_data1.csv', header=0, sep=",")
    return plcc, srocc


def run_frmwrk_ps_pc():
    """classifer is not used
    predictor is used
    """
    num_defer_pairs = 25
    references = [df_test, df_test, df_test, df_test, df_test]
    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    for ref in references:
        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth('data1')

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores = frmwrk_ps_pc(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    df_test = pd.read_csv('./src/ref_data1.csv', header=0, sep=",")
    return plcc, srocc


def main():

    plcc, srocc, trials = run_frmwrk_clf_only()
    print(f"plcc is: {plcc}, srocc is: {srocc}")

    plcc, srocc = run_frmwrk_predictor_only()
    print(f"plcc is: {plcc}, srocc is: {srocc}")

    plcc_vector = []
    srocc_vector = []
    for repeat in range(25):
        plcc, srocc = run_frmwrk_with_rnd_clf_with_predictor()
        plcc_vector.append(plcc)
        srocc_vector.append(srocc_vector)
        print(f"plcc is: {plcc}, srocc is: {srocc}")

    plcc, srocc = run_frmwrk_ps_pc()
    print(f"plcc is: {plcc}, srocc is: {srocc}")


if __name__ == '__main__':
    main()
