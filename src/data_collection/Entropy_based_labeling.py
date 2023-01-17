import sys
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


def select_min_value_from_entropy_result(entropy_result, marked_matrix):
    entropy_result = np.ma.array(entropy_result, mask=marked_matrix)
    i, j = np.unravel_index(entropy_result.argmin(), entropy_result.shape)

    return i, j


def entropy_estimation(prior_pcm, num_images):

    # Entropy = -plog(p)-(1-p)log(1-p)
    entropy_result = np.zeros([num_images, num_images])
    p = np.zeros([num_images, num_images])
    for i in range(num_images):
        for j in range(num_images):
            if(prior_pcm[i, j] == 0):
                p[i, j] = 0
            else:
                p[i, j] = (prior_pcm[i, j] /
                           (prior_pcm[i, j] + prior_pcm[j, i]))
    temp = np.subtract(1, p)

    entropy_result = - np.multiply(p, np.log2(p)) - \
        np.multiply(temp, np.log2(temp))
    entropy_result[np.isnan(entropy_result)] = 0
    return entropy_result


def run(conditions, pcm_current, scores_full, final_result, marked_matrix, entropy_result):
    iteration = ((conditions * (conditions - 1)) / 2)
    plcc_vector = []
    count = 0
    while (iteration != 1):

        pair_i, pair_j = select_min_value_from_entropy_result(
            entropy_result, marked_matrix)

        temp_ij = pcm_current[pair_i, pair_j]
        temp_ji = pcm_current[pair_j, pair_i]

        pcm_current[pair_i, pair_j] = 0.5
        pcm_current[pair_j, pair_i] = 0.5

        # Measure the performance change

        [scores_curr, _, _, _, _, _] = run_modeling_Bradley_Terry(pcm_current)
        plcc, _ = pearsonr(scores_curr, scores_full)

        mark(marked_matrix, pair_i, pair_j, True)
        mark(final_result, pair_i, pair_j, True)
        if(plcc <= 0.98):
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
    entropy_result = entropy_estimation(pcm_full, CONDITIONS)

    run(CONDITIONS, pcm_current,
        scores_full, final_result, marked_matrix, entropy_result)


if __name__ == '__main__':
    main()
