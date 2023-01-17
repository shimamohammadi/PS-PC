import sys
from itertools import product
from random import sample

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats.mstats import pearsonr

from src.data import DATA_DIR
from data_collection.shared_func import (get_pcm, mark, read_matlab_data,
                                         run_modeling_Bradley_Terry)

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


def run(conditions, pcm_current, scores_full, final_result, marked_matrix, pcm_full):
    plcc_100_rnd = np.zeros((100, 119))
    for repeat in range(100):
        iteration = ((conditions * (conditions - 1)) / 2)
        marked_matrix = np.eye(16, dtype=bool)
        final_result = np.eye(16, dtype=bool)
        pcm_current = np.copy(pcm_full)
        iteration = ((conditions * (conditions - 1)) / 2)
        count = 0
        pairs_to_remove = sample(list(product(range(16), repeat=2)), k=1)

        while (iteration != 1):
            while (marked_matrix[pairs_to_remove[0][0], pairs_to_remove[0][1]] == True):
                pairs_to_remove = sample(
                    list(product(range(16), repeat=2)), k=1)

            pair_i = pairs_to_remove[0][0]
            pair_j = pairs_to_remove[0][1]

            temp_ij = pcm_current[pair_i, pair_j]
            temp_ji = pcm_current[pair_j, pair_i]

            pcm_current[pair_i, pair_j] = 0.5
            pcm_current[pair_j, pair_i] = 0.5

            # Measure the performance change
            [scores_curr, _, _, _, _, _] = run_modeling_Bradley_Terry(
                pcm_current)
            plcc, _ = pearsonr(scores_curr, scores_full)

            mark(marked_matrix, pair_i, pair_j, True)
            mark(final_result, pair_i, pair_j, True)
            if(plcc < 0.99):
                pcm_current[pair_i, pair_j] = temp_ij
                pcm_current[pair_j, pair_i] = temp_ji
                mark(final_result, pair_i, pair_j, False)
                break

            iteration = iteration - 1
            plcc_100_rnd[repeat, count] = plcc
            count = count + 1
    plcc_100_rnd = np.where(np.isnan(plcc_100_rnd), ma.array(
        plcc_100_rnd, mask=np.isnan(plcc_100_rnd)).mean(axis=0), plcc_100_rnd)
    plcc_random = []
    plcc_random = np.mean(plcc_100_rnd, axis=0)
    return plcc_random


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

    plcc_random = run(CONDITIONS, pcm_current,
                      scores_full, final_result, marked_matrix, pcm_full)


if __name__ == '__main__':
    main()
