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

from src.labeling.parent_reference import Reference


class RandomClass(Reference):
    """This is the class definintion for each reference that has degraded images
    """

    def __init__(self, df_test_features, ref_name):
        super().__init__(df_test_features, ref_name)

        # The kld matrix is updated every time the reference object is selected
        self.finished = False

    def select_randomly(self):
        pairs_to_remove = sample(
            list(product(range(self.CONDITIONS), repeat=2)), k=1)
        while (self.marked_pairs[pairs_to_remove[0][0], pairs_to_remove[0][1]] == True):
            pairs_to_remove = sample(
                list(product(range(self.CONDITIONS), repeat=2)), k=1)
        pair_i = pairs_to_remove[0][0]
        pair_j = pairs_to_remove[0][1]
        
        return pair_i, pair_j
