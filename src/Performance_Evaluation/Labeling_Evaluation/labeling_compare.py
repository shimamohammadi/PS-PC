import numpy as np
from scipy import linalg
from scipy.stats.mstats import pearsonr, spearmanr
from src.Performance_Evaluation.Labeling_Evaluation.entrpy_reference import EntropyClass
from src.Training.Groundtruth_data_collection.kld_class import KLDClass
from src.Performance_Evaluation.Labeling_Evaluation.random_reference import RandomClass
from random import sample
import random
import matplotlib.pyplot as plt
import pandas as pd


def run_entropy_labeling():
    """Entropy-based labeing function
    """
    df_data1 = pd.read_csv(
        './src/data/IQA features/reference_1_features.csv', header=0, sep=",")
    df_data2 = pd.read_csv(
        './src/data/IQA features/reference_2_features.csv', header=0, sep=",")
    df_data3 = pd.read_csv(
        './src/data/IQA features/reference_3_features.csv', header=0, sep=",")
    df_data4 = pd.read_csv(
        './src/data/IQA features/reference_4_features.csv', header=0, sep=",")
    df_data5 = pd.read_csv(
        './src/data/IQA features/reference_5_features.csv', header=0, sep=",")
    df_data6 = pd.read_csv(
        './src/data/IQA features/reference_6_features.csv', header=0, sep=",")
    df_data7 = pd.read_csv(
        './src/data/IQA features/reference_7_features.csv', header=0, sep=",")
    df_data8 = pd.read_csv(
        './src/data/IQA features/reference_8_features.csv', header=0, sep=",")
    df_data9 = pd.read_csv(
        './src/data/IQA features/reference_9_features.csv', header=0, sep=",")
    df_data10 = pd.read_csv(
        './src/data/IQA features/reference_10_features.csv', header=0, sep=",")
    df_data11 = pd.read_csv(
        './src/data/IQA features/reference_11_features.csv', header=0, sep=",")
    df_data12 = pd.read_csv(
        './src/data/IQA features/reference_12_features.csv', header=0, sep=",")
    df_data13 = pd.read_csv(
        './src/data/IQA features/reference_13_features.csv', header=0, sep=",")
    df_data14 = pd.read_csv(
        './src/data/IQA features/reference_14_features.csv', header=0, sep=",")
    df_data15 = pd.read_csv(
        './src/data/IQA features/reference_15_features.csv', header=0, sep=",")

    REF = 15
    CONDITIONS = 16
    plcc_entropy = []
    srocc_entropy = []
    ref_1 = EntropyClass(df_data1, 'data1')
    ref_2 = EntropyClass(df_data2, 'data2')
    ref_3 = EntropyClass(df_data3, 'data3')
    ref_4 = EntropyClass(df_data4, 'data4')
    ref_5 = EntropyClass(df_data5, 'data5')
    ref_6 = EntropyClass(df_data6, 'data6')
    ref_7 = EntropyClass(df_data7, 'data7')
    ref_8 = EntropyClass(df_data8, 'data8')
    ref_9 = EntropyClass(df_data9, 'data9')
    ref_10 = EntropyClass(df_data10, 'data10')
    ref_11 = EntropyClass(df_data11, 'data11')
    ref_12 = EntropyClass(df_data12, 'data12')
    ref_13 = EntropyClass(df_data13, 'data13')
    ref_14 = EntropyClass(df_data14, 'data14')
    ref_15 = EntropyClass(df_data15, 'data15')
    gth_probs_arr = np.array([
        ref_1.gth_prob, ref_2.gth_prob, ref_3.gth_prob,
        ref_4.gth_prob, ref_5.gth_prob, ref_6.gth_prob,
        ref_7.gth_prob, ref_8.gth_prob, ref_9.gth_prob,
        ref_10.gth_prob, ref_11.gth_prob, ref_12.gth_prob,
        ref_13.gth_prob, ref_14.gth_prob, ref_15.gth_prob])

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5, ref_6, ref_7,
                     ref_8, ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15]
    iteration = 1

    # Total number of pairs equalls (n(n-1)/2)*15 = 1800 pairs
    max_iter = REF * ((CONDITIONS * (CONDITIONS-1))/2)

    # Loop over every pair in the IQA dataset
    while (iteration < max_iter):
        selected_ref_idx = 0
        selected_max = -50
        selected_idx_i = 0
        selected_idx_j = 0
        for index, ref in enumerate(reference_obj):

            max_ent, idx_i, idx_j = reference_obj[index].select_max_entropy()
            # Find the minimum kld between all the references
            if (max_ent > selected_max):
                selected_ref_idx = index
                selected_max = max_ent
                selected_idx_i = idx_i
                selected_idx_j = idx_j

        # Update mark matrix, and lbl_pairs
        reference_obj[selected_ref_idx].mark(
            'marked_pairs', selected_idx_i, selected_idx_j, True)
        reference_obj[selected_ref_idx].mark(
            'lbl_pairs', selected_idx_i, selected_idx_j, True)

        # Update the selected current_pcm with the 0.5
        reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                    selected_idx_j] = 0.5
        reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                    selected_idx_i] = 0.5

        # Update BT scores for the selected reference
        reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
            reference_obj[selected_ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array([
            ref_1.current_prob, ref_2.current_prob, ref_3.current_prob,
            ref_4.current_prob, ref_5.current_prob, ref_6.current_prob,
            ref_7.current_prob, ref_8.current_prob, ref_9.current_prob,
            ref_10.current_prob, ref_11.current_prob, ref_12.current_prob,
            ref_13.current_prob, ref_14.current_prob, ref_15.current_prob])

        # Calculate PLCC between all the current_pcm scores and the groundtruth scores
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)

        iteration = iteration + 1
        plcc_entropy.append(plcc)
        srocc_entropy.append(srocc)

    return plcc_entropy, srocc_entropy


def run_random_labeling():

    df_data1 = pd.read_csv(
        './src/data/IQA features/reference_1_features.csv', header=0, sep=",")
    df_data2 = pd.read_csv(
        './src/data/IQA features/reference_2_features.csv', header=0, sep=",")
    df_data3 = pd.read_csv(
        './src/data/IQA features/reference_3_features.csv', header=0, sep=",")
    df_data4 = pd.read_csv(
        './src/data/IQA features/reference_4_features.csv', header=0, sep=",")
    df_data5 = pd.read_csv(
        './src/data/IQA features/reference_5_features.csv', header=0, sep=",")
    df_data6 = pd.read_csv(
        './src/data/IQA features/reference_6_features.csv', header=0, sep=",")
    df_data7 = pd.read_csv(
        './src/data/IQA features/reference_7_features.csv', header=0, sep=",")
    df_data8 = pd.read_csv(
        './src/data/IQA features/reference_8_features.csv', header=0, sep=",")
    df_data9 = pd.read_csv(
        './src/data/IQA features/reference_9_features.csv', header=0, sep=",")
    df_data10 = pd.read_csv(
        './src/data/IQA features/reference_10_features.csv', header=0, sep=",")
    df_data11 = pd.read_csv(
        './src/data/IQA features/reference_11_features.csv', header=0, sep=",")
    df_data12 = pd.read_csv(
        './src/data/IQA features/reference_12_features.csv', header=0, sep=",")
    df_data13 = pd.read_csv(
        './src/data/IQA features/reference_13_features.csv', header=0, sep=",")
    df_data14 = pd.read_csv(
        './src/data/IQA features/reference_14_features.csv', header=0, sep=",")
    df_data15 = pd.read_csv(
        './src/data/IQA features/reference_15_features.csv', header=0, sep=",")

    REF = 15
    CONDITIONS = 16
    plcc_random = []
    srocc_random = []
    ref_1 = RandomClass(df_data1, 'data1')
    ref_2 = RandomClass(df_data2, 'data2')
    ref_3 = RandomClass(df_data3, 'data3')
    ref_4 = RandomClass(df_data4, 'data4')
    ref_5 = RandomClass(df_data5, 'data5')
    ref_6 = RandomClass(df_data6, 'data6')
    ref_7 = RandomClass(df_data7, 'data7')
    ref_8 = RandomClass(df_data8, 'data8')
    ref_9 = RandomClass(df_data9, 'data9')
    ref_10 = RandomClass(df_data10, 'data10')
    ref_11 = RandomClass(df_data11, 'data11')
    ref_12 = RandomClass(df_data12, 'data12')
    ref_13 = RandomClass(df_data13, 'data13')
    ref_14 = RandomClass(df_data14, 'data14')
    ref_15 = RandomClass(df_data15, 'data15')
    gth_probs_arr = np.array([
        ref_1.gth_prob, ref_2.gth_prob, ref_3.gth_prob,
        ref_4.gth_prob, ref_5.gth_prob, ref_6.gth_prob,
        ref_7.gth_prob, ref_8.gth_prob, ref_9.gth_prob,
        ref_10.gth_prob, ref_11.gth_prob, ref_12.gth_prob,
        ref_13.gth_prob, ref_14.gth_prob, ref_15.gth_prob])

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5, ref_6, ref_7,
                     ref_8, ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15]
    iteration = 1

    # Total number of pairs equalls (n(n-1)/2)*15 = 1800 pairs
    max_iter = REF * ((CONDITIONS * (CONDITIONS-1))/2)

    while (iteration < max_iter):
        # Select one reference randomly
        ref_idx = random.randint(0, 14)

        while (reference_obj[ref_idx].finished == True):
            ref_idx = random.randint(0, 14)
        # Select one pair from the reference randomly
        idx_i, idx_j = reference_obj[ref_idx].select_randomly()

        # Update mark and current matrix
        reference_obj[ref_idx].mark(
            'marked_pairs', idx_i, idx_j, True)
        reference_obj[ref_idx].mark(
            'lbl_pairs', idx_i, idx_j, True)

        reference_obj[ref_idx].current_pcm[idx_i, idx_j] = 0.5
        reference_obj[ref_idx].current_pcm[idx_j, idx_i] = 0.5

        # Update BT scores for the selected reference
        reference_obj[ref_idx].current_prob, _ = reference_obj[ref_idx].get_scores(
            reference_obj[ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array([
            ref_1.current_prob, ref_2.current_prob, ref_3.current_prob,
            ref_4.current_prob, ref_5.current_prob, ref_6.current_prob,
            ref_7.current_prob, ref_8.current_prob, ref_9.current_prob,
            ref_10.current_prob, ref_11.current_prob, ref_12.current_prob,
            ref_13.current_prob, ref_14.current_prob, ref_15.current_prob])

        # Calculate PLCC
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)

        plcc_random.append(plcc)
        srocc_random.append(srocc)

        # If every pair of the chosen reference is selected, mark it as finish
        if (np.all(reference_obj[ref_idx].marked_pairs == True)):
            reference_obj[ref_idx].finished = True
        iteration = iteration + 1
    return plcc_random, srocc_random


def repeat_rand_labling():
    plcc_repeat_rnd = np.zeros((10, 1799))
    srocc_repeat_rnd = np.zeros((10, 1799))
    for repeat in range(10):
        plcc_random, srocc_random = run_random_labeling()
        plcc_repeat_rnd[repeat,] = plcc_random
        srocc_repeat_rnd[repeat,] = srocc_random
    avg_plcc_random = []
    avg_plcc_random = np.mean(plcc_repeat_rnd, axis=0)

    avg_srocc_random = []
    avg_srocc_random = np.mean(srocc_repeat_rnd, axis=0)

    return avg_plcc_random, avg_srocc_random


def plot_labeling_approaches(plcc_entropy, srocc_entropy, plcc_kld, srocc_kld, avg_plcc_random, avg_srocc_random):
    num_entropy = [i for i in range(len(plcc_entropy))]
    num_kld = [i for i in range(len(plcc_kld))]
    num_rnd = [i for i in range(len(avg_plcc_random))]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].plot(num_entropy, plcc_entropy, 'g-', label='Entropy-based labeling')
    ax[0].plot(num_kld, plcc_kld, 'm-', label='KLD-based labeling')
    ax[0].plot(num_rnd, avg_plcc_random, 'c-', label='Random-based labeling')

    ax[0].legend(fontsize=13)
    ax[0].set_xlabel('Number of $\it{predict}$ pairs', fontsize=14)
    ax[0].set_ylabel('PLCC', fontsize=14)
    ax[0].grid()

    ax[1].plot(num_entropy, srocc_entropy, 'g-',
               label='Entropy-based labeling')
    ax[1].plot(num_kld, srocc_kld, 'm-', label='KLD-based labeling')
    ax[1].plot(num_rnd, avg_srocc_random, 'c-', label='Random-based labeling')

    ax[1].legend(fontsize=13)
    ax[1].set_xlabel('Number of $\it{predict}$ pairs', fontsize=14)
    ax[1].set_ylabel('SROCC', fontsize=14)
    ax[1].grid()
    plt.show()
    plt.savefig("labeling.pdf", pad_inches=0.01, bbox_inches="tight")


def run_kld_labeling():
    df_data1 = pd.read_csv(
        './src/data/IQA features/reference_1_features.csv', header=0, sep=",")
    df_data2 = pd.read_csv(
        './src/data/IQA features/reference_2_features.csv', header=0, sep=",")
    df_data3 = pd.read_csv(
        './src/data/IQA features/reference_3_features.csv', header=0, sep=",")
    df_data4 = pd.read_csv(
        './src/data/IQA features/reference_4_features.csv', header=0, sep=",")
    df_data5 = pd.read_csv(
        './src/data/IQA features/reference_5_features.csv', header=0, sep=",")
    df_data6 = pd.read_csv(
        './src/data/IQA features/reference_6_features.csv', header=0, sep=",")
    df_data7 = pd.read_csv(
        './src/data/IQA features/reference_7_features.csv', header=0, sep=",")
    df_data8 = pd.read_csv(
        './src/data/IQA features/reference_8_features.csv', header=0, sep=",")
    df_data9 = pd.read_csv(
        './src/data/IQA features/reference_9_features.csv', header=0, sep=",")
    df_data10 = pd.read_csv(
        './src/data/IQA features/reference_10_features.csv', header=0, sep=",")
    df_data11 = pd.read_csv(
        './src/data/IQA features/reference_11_features.csv', header=0, sep=",")
    df_data12 = pd.read_csv(
        './src/data/IQA features/reference_12_features.csv', header=0, sep=",")
    df_data13 = pd.read_csv(
        './src/data/IQA features/reference_13_features.csv', header=0, sep=",")
    df_data14 = pd.read_csv(
        './src/data/IQA features/reference_14_features.csv', header=0, sep=",")
    df_data15 = pd.read_csv(
        './src/data/IQA features/reference_15_features.csv', header=0, sep=",")

    REF = 15
    CONDITIONS = 16
    plcc_kld = []
    srocc_kld = []
    stage = 'labeling_compare'
    ref_1 = KLDClass(df_data1, 'data1', stage)
    ref_2 = KLDClass(df_data2, 'data2', stage)
    ref_3 = KLDClass(df_data3, 'data3', stage)
    ref_4 = KLDClass(df_data4, 'data4', stage)
    ref_5 = KLDClass(df_data5, 'data5', stage)
    ref_6 = KLDClass(df_data6, 'data6', stage)
    ref_7 = KLDClass(df_data7, 'data7', stage)
    ref_8 = KLDClass(df_data8, 'data8', stage)
    ref_9 = KLDClass(df_data9, 'data9', stage)
    ref_10 = KLDClass(df_data10, 'data10', stage)
    ref_11 = KLDClass(df_data11, 'data11', stage)
    ref_12 = KLDClass(df_data12, 'data12', stage)
    ref_13 = KLDClass(df_data13, 'data13', stage)
    ref_14 = KLDClass(df_data14, 'data14', stage)
    ref_15 = KLDClass(df_data15, 'data15', stage)
    gth_probs_arr = np.array([
        ref_1.gth_prob, ref_2.gth_prob, ref_3.gth_prob,
        ref_4.gth_prob, ref_5.gth_prob, ref_6.gth_prob,
        ref_7.gth_prob, ref_8.gth_prob, ref_9.gth_prob,
        ref_10.gth_prob, ref_11.gth_prob, ref_12.gth_prob,
        ref_13.gth_prob, ref_14.gth_prob, ref_15.gth_prob])

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5, ref_6, ref_7,
                     ref_8, ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15]
    iteration = 1

    # Taken out three reference as test set, (n(n-1)/2)*12 = 1400 pairs remains in the dataset
    max_iter = REF * ((CONDITIONS * (CONDITIONS-1))/2)

    # Loop over every pair in the IQA dataset
    while (iteration < max_iter):
        selected_ref_idx = 0
        selected_min = 1000
        selected_idx_i = 0
        selected_idx_j = 0

        # (Find a pair as "Predict")
        for index, ref in enumerate(reference_obj):

            # Select the minimum kld for the ref
            min_kld, idx_i, idx_j = reference_obj[index].select_min_value_from_KLD_matrix(
            )
            # Find the minimum kld between all the references
            if (min_kld < selected_min):
                selected_ref_idx = index
                selected_min = min_kld
                selected_idx_i = idx_i
                selected_idx_j = idx_j

        # (Label the selected pair as "Predict")
        reference_obj[selected_ref_idx].mark(
            'marked_pairs', selected_idx_i, selected_idx_j, True)

        reference_obj[selected_ref_idx].mark(
            'lbl_pairs', selected_idx_i, selected_idx_j, True)

        # (Get the result from the prediction output for the selected "Predict" pair)
        reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                    selected_idx_j] = 0.5
        reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                    selected_idx_i] = 0.5

        # (Inferr scores for the PCM of the selected pair after lebeling)
        reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
            reference_obj[selected_ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array([
            ref_1.current_prob, ref_2.current_prob, ref_3.current_prob,
            ref_4.current_prob, ref_5.current_prob, ref_6.current_prob,
            ref_7.current_prob, ref_8.current_prob, ref_9.current_prob,
            ref_10.current_prob, ref_11.current_prob, ref_12.current_prob,
            ref_13.current_prob, ref_14.current_prob, ref_15.current_prob])

        # Calculate PLCC between all the current_pcm scores and the groundtruth scores
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)

        # update kld
        reference_obj[selected_ref_idx].create_kld_matrix()
        plcc_kld.append(plcc)
        srocc_kld.append(srocc)
        iteration = iteration + 1
        print('kld', iteration)

    return plcc_kld, srocc_kld


def main():
    plcc_entropy, srocc_entropy = run_entropy_labeling()
    plcc_kld, srocc_kld = run_kld_labeling()
    avg_plcc_random, avg_srocc_random = repeat_rand_labling()
    num_rnd = [i for i in range(len(avg_plcc_random))]

    plot_labeling_approaches(plcc_entropy, srocc_entropy,
                             plcc_kld, srocc_kld, avg_plcc_random, avg_srocc_random)


if __name__ == '__main__':
    main()
