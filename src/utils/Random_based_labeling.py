import scipy.io
import numpy.polynomial.hermite as herm
import math
import cmath
import csv
import statistics
import sys
import time
from itertools import product
from random import sample

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import rel_entr
from scipy.stats.mstats import pearsonr, spearmanr

from utils.shared_func import read_matlab_data, get_pcm, mark, run_modeling_Bradley_Terry
from src.data import DATA_DIR

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


def run(conditions, pcm_current, scores_full, final_result, marked_matrix):
    iteration = ((conditions * (conditions - 1)) / 2)
    plcc_vector = []
    count = 0
    pairs_to_remove = sample(list(product(range(16), repeat=2)), k=1)

    while (iteration != 1):
        while (marked_matrix[pairs_to_remove[0][0], pairs_to_remove[0][1]] == True):
            pairs_to_remove = sample(list(product(range(16), repeat=2)), k=1)

        pair_i = pairs_to_remove[0][0]
        pair_j = pairs_to_remove[0][1]

        temp_ij = pcm_current[pair_i, pair_j]
        temp_ji = pcm_current[pair_j, pair_i]

        pcm_current[pair_i, pair_j] = 0.5
        pcm_current[pair_j, pair_i] = 0.5

        # Measure the performance change
        [scores_curr, _, _, _, _, _] = run_modeling_Bradley_Terry(pcm_current)
        plcc, _ = pearsonr(scores_curr, scores_full)

        mark(marked_matrix, pair_i, pair_j, True)
        mark(final_result, pair_i, pair_j, True)
        if(plcc < 0.99):
            pcm_current[pair_i, pair_j] = temp_ij
            pcm_current[pair_j, pair_i] = temp_ji
            mark(final_result, pair_i, pair_j, False)
            break

        plcc_vector.append(plcc)
        iteration = iteration - 1
        count = count + 1


def main():

    CONDITIONS = 16
    raw_data = read_matlab_data('VQA', 'data1')
    pcm_full = get_pcm(CONDITIONS, raw_data)

    [scores_full, _, _, prob_full, pstd_full,
        _] = run_modeling_Bradley_Terry(pcm_full)
    pcm_current = np.copy(pcm_full)
    THERESHOLD = 0.02
    marked_matrix = np.eye(16, dtype=bool)
    final_result = np.eye(16, dtype=bool)

    run(CONDITIONS, pcm_current,
        scores_full, final_result, marked_matrix)


if __name__ == '__main__':
    main()
