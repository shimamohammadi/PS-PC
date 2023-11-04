from src.tools import read_matlab_data, run_modeling_Bradley_Terry, get_pcm
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd


class ParentClass:
    """This is the class definintion for each reference content. 
    It contains the same features and functions for kld, entropy, and random class
    """

    def __init__(self, df_test_features, ref_name):

        # Number of degraded images per reference content
        self.CONDITIONS = 16

        # A matrix for tracking the selection of pairs for evaluation
        self.marked_pairs = np.eye(self.CONDITIONS, dtype=bool)

        # A matrix to lable each pair as 'predic' or 'defer'
        self.lbl_pairs = np.eye(self.CONDITIONS, dtype=bool)

        # The ground truh pcm, and the corresponding inferred scores and standard deviation. This is FIXED during each iteration.
        self.gth_pcm = self.read_gth_data(ref_name)
        self.gth_prob, self.gth_std = self.get_scores(self.gth_pcm)

        # Initially, the current PCM and scores are identical to the ground truth. However, they may undergo changes during each iteration.
        self.current_pcm = np.copy(self.gth_pcm)
        self.current_prob, self.current_std = self.gth_prob, self.gth_std

        # The prediction pcm containing the predictor output, with each entry representing the probability of preferring the first image over the second one.
        self.prediction_pcm = self.apply_predictor(df_test_features)

    def load_predicor(self, predictor_name):
        """Load the predictor module
        """
        predictor = pickle.load(
            open('./src/data/Trained models/Predictor/predictor_123_model.sav', 'rb'))
        return predictor

    def read_gth_data(self, ref_name):
        """Read the ground truth data
        """
        gth_pcm = np.zeros((self.CONDITIONS, self.CONDITIONS))
        raw_data = read_matlab_data(ref_name)

        # Get probability of preference
        pcm_temp = get_pcm(self.CONDITIONS, raw_data)
        for i in range(self.CONDITIONS):
            for j in range(self.CONDITIONS):
                if (pcm_temp[i, j] == 0):
                    gth_pcm[i, j] = 0
                else:
                    gth_pcm[i, j] = (
                        pcm_temp[i, j] / (pcm_temp[i, j] + pcm_temp[j, i]))
        return gth_pcm

    def apply_predictor(self, df_test_features):
        """Apply predictor on the test features, and get predictions
        """

        predictor = self.load_predicor("predictor_name")
        prediction_pcm = np.zeros((self.CONDITIONS, self.CONDITIONS))

        # Transform features
        scaler_x = MinMaxScaler()
        X_test_transformed = pd.DataFrame(
            scaler_x.fit_transform(df_test_features))
        predictions_res = predictor.predict(X_test_transformed)

        # Transform output (y)
        scaler_y = MinMaxScaler()
        predictions_res = pd.DataFrame(predictions_res)
        predictions_res = scaler_y.fit_transform(predictions_res)

        # Fill the matrix from predictor output
        ind_upper = np.triu_indices(self.CONDITIONS, 1)
        predictions_res = np.squeeze(predictions_res)
        prediction_pcm[ind_upper] = predictions_res
        for row in range(self.CONDITIONS):
            for col in range(row+1, self.CONDITIONS):
                prediction_pcm[col, row] = 1 - \
                    prediction_pcm[row, col]
        return prediction_pcm

    def get_scores(self, pcm):
        """Infer scores from a PCM
        """
        [scores_pcm, std_pcm, _, prob_pcm, pstd_pcm,
            pcov_pcm] = run_modeling_Bradley_Terry(pcm)
        return prob_pcm, pstd_pcm

    def mark(self, matrix, pair_i, pair_j, boolean):
        """Mark the matrix as true or false
        """
        if (matrix == 'marked_pairs'):
            self.marked_pairs[pair_i, pair_j] = boolean
            self.marked_pairs[pair_j, pair_i] = boolean
        else:
            self.lbl_pairs[pair_i, pair_j] = boolean
            self.lbl_pairs[pair_j, pair_i] = boolean
