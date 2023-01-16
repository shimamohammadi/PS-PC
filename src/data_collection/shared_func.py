import sys
import csv

import numpy as np
import pandas as pd
import scipy.io
from scipy import linalg

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


def write_to_csv(file_name, results_matrix):
    with open(file_name+'.csv', 'w') as file:
        writer = csv.writer(file)
        for row in range(len(results_matrix)):
            for col in range(row+1, len(results_matrix)):
                writer.writerow([row, col, results_matrix[row, col]])
