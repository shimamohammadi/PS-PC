import pickle
from itertools import product
from random import sample
import numpy as np
import pandas as pd
from scipy.stats.mstats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

from src.tools import get_pcm, read_matlab_data, run_modeling_Bradley_Terry


def frmwrk_with_clf_only(gth_prob_pcm, gth_judgments_pcm, X_test_transformed, clf_name):
    """This function is used to run the framework without predictor and with the clf for ONE reference
    The pcm is filled from the classifers' output only. 
    The groundruth data is used for defer pairs, and 0.5 is used for predict pairs 
    """

    CONDITIONS = 16
    trial_num = 0
    clf_only_pcm = np.zeros((CONDITIONS, CONDITIONS))

    clf_only_pcm[clf_only_pcm == 0] = 0.5
    for i in range(CONDITIONS):
        for j in range(CONDITIONS):
            if (i == j):
                clf_only_pcm[i, j] = 0

    # Apply classifier
    clf_output = classify(X_test_transformed, clf_name)

    # Update the prediction pcm using the classifer output
    for row in range(CONDITIONS):
        for col in range(row+1, CONDITIONS):
            if (clf_output[row, col] == False):
                clf_only_pcm[row, col] = gth_prob_pcm[row, col]
                clf_only_pcm[col, row] = gth_prob_pcm[col, row]
                trial_num += gth_judgments_pcm[row,
                                               col] + gth_judgments_pcm[col, row]

    scores_model = infer_scores(clf_only_pcm)

    return scores_model, trial_num


def frmwrk_w_predictor_only(X_test_transformed, predictor_name):
    """This function is used to run the framework without clf and with the predictor for ONE reference
    """

    prediction_pcm = predictor_only(X_test_transformed, predictor_name)
    scores_model = infer_scores(prediction_pcm)
    return scores_model


def frmwrk_with_rnd_clf_with_predictor(gth_prob_pcm, gth_judgments_pcm, X_test_transformed, num_defer_pairs, predictor_name):
    """This function is used to run the framework with predictor and with random classifier for ONE reference 
    Random classifier and predictor
    """
    CONDITIONS = 16
    # num_defer_pairs = 25
    trial_num = 0

    # Apply predictor
    prediction_pcm = predictor_only(X_test_transformed, predictor_name)

    # Apply random classifeir
    marked_matrix = random_clf(num_defer_pairs)

    # Update the prediction pcm
    for row in range(CONDITIONS):
        for col in range(CONDITIONS):
            if (marked_matrix[row, col] == False):
                prediction_pcm[row, col] = gth_prob_pcm[row, col]
                trial_num += gth_judgments_pcm[row,
                                               col] + gth_judgments_pcm[col, row]

    scores_model = infer_scores(prediction_pcm)
    return scores_model, trial_num


def frmwrk_ps_pc(gth_prob_pcm, gth_judgments_pcm, X_test_transformed, predictor_name, clf_name):
    """ This function is used to run the framework for ONE reference
    Framwwork = Predictor and  classiifer
    """

    CONDITIONS = 16
    trial_num = 0

    # Apply predictor
    prediction_pcm = predictor_only(X_test_transformed, predictor_name)

    # Apply classifer
    clf_output = classify(X_test_transformed, clf_name)

    # Update the prediction pcm using the classifer output
    for row in range(CONDITIONS):
        for col in range(row+1, CONDITIONS):
            if (clf_output[row, col] == False):
                prediction_pcm[row, col] = gth_prob_pcm[row, col]
                prediction_pcm[col, row] = gth_prob_pcm[col, row]
                trial_num += gth_judgments_pcm[row,
                                               col] + gth_judgments_pcm[col, row]

    scores_model = infer_scores(prediction_pcm)
    return scores_model, trial_num


def read_gth(ref_name):

    CONDITIONS = 16
    raw_data = read_matlab_data(ref_name)
    gth_judgments_pcm = get_pcm(CONDITIONS, raw_data)

    gth_prob_pcm = np.zeros((CONDITIONS, CONDITIONS))
    for i in range(CONDITIONS):
        for j in range(CONDITIONS):
            if (gth_judgments_pcm[i, j] == 0):
                gth_prob_pcm[i, j] = 0
            else:
                gth_prob_pcm[i, j] = (gth_judgments_pcm[i, j] /
                                      (gth_judgments_pcm[i, j] + gth_judgments_pcm[j, i]))

    # Get scores for the gth_pcm
    gth_scores = infer_scores(gth_prob_pcm)

    return gth_prob_pcm, gth_judgments_pcm, gth_scores


def predictor_only(X_test_transformed, predictor_name):
    """ Apply predictor, and return a pcm
    """
    CONDITIONS = 16
    # Apply predictor
    # Predictor output. A vector not a pcm
    predictions = predict(X_test_transformed, predictor_name)
    # Get pcm
    prediction_pcm = np.zeros([CONDITIONS, CONDITIONS])

    # Fill the matrix from predictor output
    ind_upper = np.triu_indices(CONDITIONS, 1)
    predictions = np.squeeze(predictions)
    prediction_pcm[ind_upper] = predictions
    for row in range(CONDITIONS):
        for col in range(row+1, CONDITIONS):
            prediction_pcm[col, row] = 1 - prediction_pcm[row, col]
    return prediction_pcm


def classify(X_test_transformed, clf_name):
    """The pcm is filled from the classifers' output. 
    """
    CONDITIONS = 16
    clf_output = np.zeros((CONDITIONS, CONDITIONS))

    # Fill the matrix from Classifeir
    clf = pickle.load(
        open('./src/data/Trained models/CLF/XGBOOST/0.98/'+clf_name, 'rb'))
    cls_results = clf.predict(X_test_transformed)
    ind_upper = np.triu_indices(CONDITIONS, 1)
    clf_output[ind_upper] = cls_results
    ind_lower = np.tril_indices(CONDITIONS, 0)
    clf_output[ind_lower] = np.nan

    return clf_output


def infer_scores(pcm):
    [scores_full, _, _, prob_full, _, _] = run_modeling_Bradley_Terry(pcm)
    return prob_full


def random_clf(num_defer_pairs):
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


def predict(X_test_transformed, predictor_name):
    predictor = pickle.load(
        open('./src/data/Trained models/Predictor/'+predictor_name, 'rb'))
    predictions = predictor.predict(X_test_transformed)

    predictions = pd.DataFrame(predictions)

    # normalize predictions
    predictions = normalize_features(predictions)

    return predictions


def calculate_correlations(scores_gth, scores_model):
    """Calculate the corelations between the groundtruth scores and the model's scores
    """
    plcc, _ = pearsonr(scores_gth, scores_model)
    srocc, _ = spearmanr(scores_gth, scores_model)

    return plcc, srocc


def run_frmwrk_clf_only(references, ref_names):
    """This function is used to run the framework without predictor and with the clf for ALL references
    classifer output is used
    predictor is not used
    """
    clf_models = ['clf_123_model.sav', 'clf_456_model.sav', 'clf_789_model.sav',
                  'clf_101112_model.sav', 'clf_131415_model.sav']
    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    all_ref_trial_num = 0
    for index, ref in enumerate(references):

        # Select a classifier model
        q = index//3
        clf_name = clf_models[q]

        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth(
            ref_names[index])

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores, trial_num = frmwrk_with_clf_only(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed, clf_name)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)
        all_ref_trial_num += trial_num

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    return plcc, srocc, all_ref_trial_num


def run_frmwrk_predictor_only(references, ref_names):
    """This function is used to run the framework with predictor and without clf for ALL references
    classifer is not used
    predictor is used
    """
    predictor_models = ['predictor_123_model.sav', 'predictor_456_model.sav', 'predictor_789_model.sav',
                        'predictor_101112_model.sav', 'predictor_131415_model.sav']

    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    for index, ref in enumerate(references):
        # Select a predictor model
        q = index//3
        predictor_name = predictor_models[q]

        # Read groundtruth data
        _, _, gth_scores = read_gth(ref_names[index])

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores = frmwrk_w_predictor_only(
            X_test_transformed, predictor_name)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    return plcc, srocc


def run_frmwrk_with_rnd_clf_with_predictor(references, ref_names, num_defer_pairs):
    """This function is used to run the framework with predictor and with random clf for ALL references
    classifer is not used
    predictor is used
    """

    predictor_models = ['predictor_123_model.sav', 'predictor_456_model.sav', 'predictor_789_model.sav',
                        'predictor_101112_model.sav', 'predictor_131415_model.sav']

    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    all_ref_trial_num = 0
    for index, ref in enumerate(references):

        # Select a predictor model
        q = index//3
        predictor_name = predictor_models[q]

        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth(
            ref_names[index])

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores, trial_num = frmwrk_with_rnd_clf_with_predictor(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed, num_defer_pairs, predictor_name)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)
        all_ref_trial_num += trial_num

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)

    return plcc, srocc, all_ref_trial_num


def run_frmwrk_ps_pc(references, ref_names):
    """This function is used to run the framework for ALL references
    classifer is not used
    predictor is used
    """

    predictor_models = ['predictor_123_model.sav', 'predictor_456_model.sav', 'predictor_789_model.sav',
                        'predictor_101112_model.sav', 'predictor_131415_model.sav']

    clf_models = ['clf_123_model.sav', 'clf_456_model.sav', 'clf_789_model.sav',
                  'clf_101112_model.sav', 'clf_131415_model.sav']

    all_ref_gth_scores = []
    all_ref_frmwrk_scores = []
    all_ref_trial_num = 0
    for index, ref in enumerate(references):

        # Select a classifier and predictor model
        q = index//3
        clf_name = clf_models[q]
        predictor_name = predictor_models[q]

        # Read groundtruth data
        gth_prob_pcm, gth_judgments_pcm, gth_scores = read_gth(
            ref_names[index])

        # Read and normalize
        X_test_transformed = normalize_features(ref)

        frmwork_scores, trial_num = frmwrk_ps_pc(
            gth_prob_pcm, gth_judgments_pcm, X_test_transformed, predictor_name, clf_name)

        all_ref_gth_scores.append(gth_scores)
        all_ref_frmwrk_scores.append(frmwork_scores)
        all_ref_trial_num += trial_num

    plcc, srocc = calculate_correlations(
        all_ref_gth_scores, all_ref_frmwrk_scores)
    # print(all_ref_frmwrk_scores)

    return plcc, srocc, all_ref_trial_num


def main():
    df_data1 = pd.read_csv(
        './src/data/IQA features/reference_1_features.csv', header=0, sep=",")
    df_data2 = pd.read_csv(
        './src/data/IQA features/reference_2_features.csv', header=0, sep=",")
    df_data3 = pd.read_csv(
        './src/data/IQA features/reference_3_features.csv', header=0, sep=",")
    df_data4 = pd.read_csv(
        './src/data/IQA features/reference_4_features.csv', header=0, sep=",")
    df_data5 = pd.read_csv(
        './src/data/IQA features/reference_5_features.csv', header=0, sep=",")
    df_data6 = pd.read_csv(
        './src/data/IQA features/reference_6_features.csv', header=0, sep=",")
    df_data7 = pd.read_csv(
        './src/data/IQA features/reference_7_features.csv', header=0, sep=",")
    df_data8 = pd.read_csv(
        './src/data/IQA features/reference_8_features.csv', header=0, sep=",")
    df_data9 = pd.read_csv(
        './src/data/IQA features/reference_9_features.csv', header=0, sep=",")
    df_data10 = pd.read_csv(
        './src/data/IQA features/reference_10_features.csv', header=0, sep=",")
    df_data11 = pd.read_csv(
        './src/data/IQA features/reference_11_features.csv', header=0, sep=",")
    df_data12 = pd.read_csv(
        './src/data/IQA features/reference_12_features.csv', header=0, sep=",")
    df_data13 = pd.read_csv(
        './src/data/IQA features/reference_13_features.csv', header=0, sep=",")
    df_data14 = pd.read_csv(
        './src/data/IQA features/reference_14_features.csv', header=0, sep=",")
    df_data15 = pd.read_csv(
        './src/data/IQA features/reference_15_features.csv', header=0, sep=",")

    references = [df_data1, df_data2, df_data3, df_data4, df_data5, df_data6, df_data7,
                  df_data8, df_data9, df_data10, df_data11, df_data12, df_data13, df_data14, df_data15]
    ref_names = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7',
                 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14', 'data15']

    # Classifier only
    plcc, srocc, trials = run_frmwrk_clf_only(references, ref_names)
    print(f"plcc is: {plcc}, srocc is: {srocc}, and trial is{trials}")

    # Predictor only
    plcc, srocc = run_frmwrk_predictor_only(references, ref_names)
    print(f"plcc is: {plcc}, srocc is: {srocc}")

    # RndClass+predict
    plcc_vector = []
    srocc_vector = []
    trial_vector = []
    # try this numbers => 10, 13, 17, 20, 25
    num_defer_pairs = 25
    for repeat in range(25):
        plcc, srocc, trials = run_frmwrk_with_rnd_clf_with_predictor(
            references, ref_names, num_defer_pairs)
        plcc_vector.append(plcc)
        srocc_vector.append(srocc)
        trial_vector.append(trials)
        print(f"plcc is: {plcc}, srocc is: {srocc}, and trial is{trials}")
    print(
        f" Average plcc is: {sum(plcc_vector)/len(plcc_vector)}, average srocc is: {sum(srocc_vector)/len(srocc_vector)}, and average trail is:{sum(trial_vector)/len(trial_vector)}")

    # PS-PC
    plcc, srocc, trials = run_frmwrk_ps_pc(references, ref_names)
    print(f"plcc is: {plcc}, srocc is: {srocc}, and trails is {trials}")


if __name__ == '__main__':
    main()
