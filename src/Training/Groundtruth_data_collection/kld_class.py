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
from src.Training.Groundtruth_data_collection.parent_class import ParentClass


class KLDClass(ParentClass):
    """This is the class definintion for each reference that has degraded images
    """

    def __init__(self, df_test_features, ref_name):
        super().__init__(df_test_features, ref_name)

        # The kld matrix is updated every time the reference object is selected
        self.kld_matrix = self.create_kld_matrix()

    def create_kld_matrix(self):
        """This function removes a pair in each iteration from the current_pcm, and calculate the kld
        """
        kld_matrix = np.zeros((self.CONDITIONS, self.CONDITIONS))
        for row in range(self.CONDITIONS):
            for col in range(row+1, self.CONDITIONS):
                # Storing values of the current_pcm before updating their values
                temp_1 = self.current_pcm[row, col]
                temp_2 = self.current_pcm[col, row]

                self.current_pcm[row, col] = self.prediction_pcm[row, col]
                self.current_pcm[col, row] = self.prediction_pcm[col, row]

                # Infer scores from the updated current_pcm
                [p_tmp, pstd_tmp] = self.get_scores(
                    self.current_pcm)

                # Calculate KLD between prior(groundtruth) and posterior(current_pcm)
                kld_res = kl_divergence_approx(
                    self.gth_prob, self.gth_std, p_tmp, pstd_tmp)

                # Recovering the current_pcm
                self.current_pcm[row, col] = temp_1
                self.current_pcm[col, row] = temp_2

                kld_matrix[row, col] = kld_res - 16
                kld_matrix[col, row] = kld_res - 16
        return kld_matrix

    def select_min_value_from_KLD_matrix(self):
        """This function outputs the minimum value in the KLD matrix and the coresponding indexes 
        """
        kld = np.ma.array(self.kld_matrix, mask=self.marked_pairs)
        i, j = np.unravel_index(kld.argmin(), kld.shape)
        minimum = kld[i, j]
        return minimum, i, j


def kl_divergence_approx(mean_1, var_1, mean_2, var_2):
    '''
    Aproximation of the multivariate normal KL divergence: 
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    '''
    total = np.sum(np.log(var_2)) - np.sum(np.log(var_1)) + \
        sum(var_1/var_2)+np.dot(1/var_2, (mean_1-mean_2)**2)
    return total
