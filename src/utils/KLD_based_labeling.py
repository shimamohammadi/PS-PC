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
from utils.shared_func import read_matlab_data, get_pcm, mark, run_modeling_Bradley_Terry, write_to_csv
from src.data import DATA_DIR

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


def select_min_value_from_KLD_matrix(kld_result, marked_pairs):
    kld_result = np.ma.array(kld_result, mask=marked_pairs)

    i, j = np.unravel_index(kld_result.argmin(), kld_result.shape)

    return i, j


def kl_divergence_approx(mean_1, var_1, mean_2, var_2):
    '''
    Aproximation of the multivariate normal KL divergence: 
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    '''
    total = np.sum(np.log(var_2)) - np.sum(np.log(var_1)) + \
        sum(var_1/var_2)+np.dot(1/var_2, (mean_1-mean_2)**2)
    return total


def compute_kldiv(mu_f, cov_f, cov_f_inv, mu_g, cov_g):
    a = np.trace(np.matmul(cov_f_inv, cov_g))
    b = np.matmul((mu_f - mu_g).T, np.matmul(cov_f_inv, (mu_f-mu_g)))[0, 0]
    c = mu_f.shape[0]
    t1 = np.linalg.det(cov_f)
    t2 = np.linalg.det(cov_g)
    t3 = t1/t2
    t4 = np.log(t3)
    d = np.log(np.linalg.det(cov_f) / np.linalg.det(cov_g))

    kldiv = (a + b - c + d) / (2 * np.log(2))

    return kldiv


def create_kld_matrix(pcm_current, num_pairs, prob_full, pstdv_full):
    """
    A function to remove one pair each time form the PCM and calculate the KL divergence 
    """
    kld_result = np.zeros([num_pairs, num_pairs])
    for row in range(num_pairs):
        for col in range(row+1, num_pairs):
            # Storing values of pcm_current before converting them to zeros
            temp_1 = pcm_current[row, col]
            temp_2 = pcm_current[col, row]

            pcm_current[row, col] = 0.5
            pcm_current[col, row] = 0.5

            [_, _, _, prob_curr, pstd_curr,
                _] = run_modeling_Bradley_Terry(pcm_current)

            # Recovering the pcm_current
            pcm_current[row, col] = temp_1
            pcm_current[col, row] = temp_2

            kld_output = kl_divergence_approx(
                prob_full, pstdv_full, prob_curr, pstd_curr)
            kld_result[row, col] = kld_output
            kld_result[col, row] = kld_output
    return kld_result


def run(conditions, pcm_current, prob_full, pstd_full, scores_full, final_result, marked_matrix):
    iteration = ((conditions * (conditions - 1)) / 2)
    plcc_vector = []
    count = 0
    while (iteration != 1):

        kld_result = create_kld_matrix(
            pcm_current, conditions, prob_full, pstd_full)

        pair_i, pair_j = select_min_value_from_KLD_matrix(
            kld_result, marked_matrix)

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

        plcc_vector.append(plcc)
        iteration = iteration - 1
        count = count + 1


def main():

    CONDITIONS = 16
    ref_name = 'data1'
    raw_data = read_matlab_data('VQA', ref_name)
    pcm_full = get_pcm(CONDITIONS, raw_data)

    [scores_full, _, _, prob_full, pstd_full,
        _] = run_modeling_Bradley_Terry(pcm_full)
    pcm_current = np.copy(pcm_full)
    THERESHOLD = 0.02
    marked_matrix = np.eye(16, dtype=bool)
    final_result = np.eye(16, dtype=bool)

    run(CONDITIONS, pcm_current, prob_full, pstd_full,
        scores_full, final_result, marked_matrix)
    write_to_csv(ref_name, final_result)


if __name__ == '__main__':
    main()
