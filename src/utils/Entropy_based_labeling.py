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

from src.data import DATA_DIR

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


def read_matlab_data(dataset_name, mat_file):
    raw_data = scipy.io.loadmat(
        "./dataset/" + dataset_name + "/" + mat_file)['data_ref']
    return raw_data


def get_pcm(n, raw_data):
    pcm = np.zeros((n, n))
    for row in raw_data:
        pcm[row[0]-1, row[1]-1] = pcm[row[0]-1, row[1]-1]+1
    return pcm


def mark(marked_matrix, pair_i, pair_j, boolean):
    marked_matrix[pair_i, pair_j] = boolean
    marked_matrix[pair_j, pair_i] = boolean


def select_min_value_from_entropy_result(entropy_result, marked_matrix):
    entropy_result = np.ma.array(entropy_result, mask=marked_matrix)
    i, j = np.unravel_index(entropy_result.argmin(), entropy_result.shape)

    return i, j


def run_modeling_Bradley_Terry(alpha):
    """this code is from zhi li, sureal package"""

    # alpha = np.array(
    #     [[0, 3, 2, 7],
    #      [1, 0, 6, 3],
    #      [4, 3, 0, 0],
    #      [1, 2, 5, 0]]
    #     )

    M, M_ = alpha.shape
    assert M == M_

    iteration = 0
    p = 1.0 / M * np.ones(M)
    change = sys.float_info.max

    DELTA_THR = 1e-8

    while change > DELTA_THR:
        iteration += 1
        p_prev = p
        n = alpha + alpha.T
        pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
        p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1)

        p = p / np.sum(p)

        change = linalg.norm(p - p_prev)

    # lambda_ii = sum_j -alpha_ij / p_i^2 + n_ij / (p_i + p_j)^2
    # lambda_ij = n_ij / (p_i + p_j)^2, i != j
    # H = [lambda_ij]
    # C = [[-H, 1], [1', 0]]^-1 of (M + 1) x (M + 1)
    # variance of p_i is then diag(C)[i].

    pp = np.tile(p, (M, 1)).T + np.tile(p, (M, 1))
    lbda_ii = np.sum(-alpha / np.tile(p, (M, 1)).T**2 + n /
                     pp**2, axis=1)  # summing over axis=1 marginalizes j
    lbda_ij = n / pp*2
    lbda = lbda_ij + np.diag(lbda_ii)
    cova_p = np.linalg.pinv(
        np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
    vari_p = np.diagonal(cova_p)[:-1]
    stdv_p = np.sqrt(vari_p)
    cova_p = cova_p[:-1, :-1]

    cova_v = cova_p / (np.expand_dims(p, axis=1) *
                       (np.expand_dims(p, axis=1).T))
    v = np.log(p)
    stdv_v = stdv_p / p  # y = log(x) -> dy = 1/x * dx

    return v, stdv_v, cova_v, p, stdv_p, cova_p


def entropy_estimation(prior_pcm, num_images):
    entropy_result = np.zeros([num_images, num_images])

    temp = np.subtract(32, prior_pcm)

    entropy_result = - \
        np.multiply(prior_pcm, np.log2(prior_pcm)) - \
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
    entropy_result = entropy_estimation(pcm_full, CONDITIONS)

    run(CONDITIONS, pcm_current,
        scores_full, final_result, marked_matrix, entropy_result)


if __name__ == '__main__':
    main()
