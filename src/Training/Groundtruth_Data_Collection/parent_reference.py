from src.tools import read_matlab_data, run_modeling_Bradley_Terry, get_pcm
import numpy.ma as ma
import random
import cmath
import math
import numpy.polynomial.hermite as herm
import scipy.io
import csv
import sys
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import pickle

import numpy as np
from scipy import linalg
from scipy.stats.mstats import pearsonr, spearmanr
import time
from itertools import product
from random import sample
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import statistics


class Reference:
    """This is the class definintion for each reference that has degraded images
    """

    def __init__(self, df_test_features, ref_name):

        # Number of degraded images per reference
        self.CONDITIONS = 16

        # A matrix to mark every pair that has been chosen for evaluation
        self.marked_pairs = np.eye(self.CONDITIONS, dtype=bool)

        # A matrix to lable each pair as 'predic' or 'defer'
        self.lbl_pairs = np.eye(self.CONDITIONS, dtype=bool)

        # The PCM after removing some pairs, and the corresponding scores and standard deviation
        self.current_pcm = np.zeros((self.CONDITIONS, self.CONDITIONS))
        self.current_prob, self.current_std = self.get_scores(self.current_pcm)

        # The ground truh pcm, and the corresponding scores and standard deviation
        self.gth_pcm = self.read_gth_data(self, ref_name)
        self.gth_prob, self.gth_std = self.get_scores(self.gth_pcm)

        # The prediction pcm containing the predictor output
        self.prediction_pcm = self.apply_predictor(self, df_test_features)

    def load_predicor(self, predictor_name):
        """Load the predictor
        """
        predictor = pickle.load(open('./Predictor_131415_model.sav', 'rb'))
        return predictor

    def read_gth_data(self, ref_name):
        """Read the ground truth data
        """
        gth_pcm = np.zeros((self.CONDITIONS, self.CONDITIONS))
        raw_data = read_matlab_data('IQA', ref_name)

        # Get probability of preference
        pcm_temp = get_pcm(self.CONDITIONS, raw_data)
        for i in range(self.CONDITIONS):
            for j in range(self.CONDITIONS):
                if (pcm_temp[i, j] == 0):
                    self.gth_pcm[i, j] = 0
                else:
                    self.gth_pcm[i, j] = (
                        pcm_temp[i, j] / (pcm_temp[i, j] + pcm_temp[j, i]))
        return gth_pcm

    def apply_predictor(self, df_test_features):
        """Apply predictor on the test features, and get predictions
        """

        predictor = self.load_predicor(self, "predictor_name")
        prediction_pcm = np.zeros((self.CONDITIONS, self.CONDITIONS))

        # Transform features
        scaler_x = MinMaxScaler()
        X_test_transformed = pd.DataFrame(
            scaler_x.fit_transform(df_test_features))
        predictions_res = predictor.predict(X_test_transformed)

        # transform output (y)
        scaler_y = MinMaxScaler()
        predictions_res = pd.DataFrame(predictions_res)
        predictions_res = scaler_y.fit_transform(predictions_res)

        # Fill the matrix from predictor output
        ind_upper = np.triu_indices(len(prediction_pcm), 1)
        predictions_res = np.squeeze(predictions_res)
        prediction_pcm[ind_upper] = predictions_res
        for row in range(len(prediction_pcm)):
            for col in range(row+1, len(prediction_pcm)):
                prediction_pcm[col, row] = 1 - \
                    prediction_pcm[row, col]

    def get_scores(self, pcm):
        """Infer scores from a PCM
        """
        [scores_pcm, std_pcm, _, prob_pcm, pstd_pcm,
            pcov_pcm] = run_modeling_Bradley_Terry(pcm)
        return prob_pcm, pstd_pcm

    def mark(self, matrix, pair_i, pair_j, boolean):
        """Mark the matrix as true or false
        """
        self.marked_pairs[pair_i, pair_j] = boolean
        self.marked_pairs[pair_j, pair_i] = boolean
