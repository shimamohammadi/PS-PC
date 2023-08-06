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
from src.labeling.parent_reference import Reference


class EntropyClass(Reference):
    """This is the class definintion for each reference that has degraded images
    """

    def __init__(self, df_test_features, ref_name):
        super().__init__(df_test_features, ref_name)

        # The kld matrix is updated every time the reference object is selected
        self.entropy_matrix = np.zeros((self.CONDITIONS, self.CONDITIONS))
        self.entropy_matrix = self.entropy_estimation()

    def select_max_entropy(self):
        """This function returns the maximum value in the entropy matrix and its coresponding indexs
        """
        entropy = np.ma.array(self.entropy_matrix, self.marked_pairs)
        i, j = np.unravel_index(entropy.argmax(), entropy.shape)
        maximum = entropy[i, j]
        return maximum, i, j

    def entropy_estimation(self, pcm):
        """This function claculate entroy for the pc matrix
        """
        temp = np.subtract(1, pcm)

        self.entropy_matrix = - np.multiply(pcm, np.log2(pcm)) - \
            np.multiply(temp, np.log2(temp))
        self.entropy_matrix[np.isnan(self.entropy_matrix)] = 0
