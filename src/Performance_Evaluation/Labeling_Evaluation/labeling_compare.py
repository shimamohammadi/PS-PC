import csv
import numpy as np
from scipy import linalg
from scipy.stats.mstats import pearsonr, spearmanr
from src.labeling.kld_reference import KLDClass
from src.labeling.entrpy_reference import EntropyClass
from src.labeling.labeling import run_kld_labeling
from src.labeling.random_reference import RandomClass
from random import sample
import random
import matplotlib.pyplot as plt


def run_entropy_labeling():
    """Entropy-based labeing function
    """

    REF = 5
    CONDITIONS = 16
    plcc_entropy = []
    srocc_entropy = []
    ref_1 = EntropyClass('data1')
    ref_2 = EntropyClass('data2')
    ref_3 = EntropyClass('data3')
    ref_4 = EntropyClass('data4')
    ref_5 = EntropyClass('data5')
    gth_probs_arr = np.array(ref_1.gth_prob, ref_2.gth_prob,
                             ref_3.gth_prob, ref_4.gth_prob, ref_5.gth_prob)

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5]
    iteration = 0

    # Taken out three reference as test set, (n(n-1)/2)*12 = 1400 pairs remains in the dataset
    max_iter = 1400

    # Loop over every pair in the IQA dataset
    while (iteration < max_iter):
        selected_ref_idx = 0
        selected_max = 1000
        selected_idx_i = 0
        selected_idx_j = 0
        for ref in range(reference_obj):

            max_ent, idx_i, idx_j = reference_obj[ref].select_max_entropy()
            # FInd the minimum kld between all the references
            if (max_ent < selected_max):
                selected_ref_idx = ref
                selected_max = max_ent
                selected_idx_i = idx_i
                selected_idx_j = idx_j

        # Backup from the current_pcm
        temp_ij = reference_obj[selected_ref_idx].current_pcm[selected_idx_i, selected_idx_j]
        temp_ji = reference_obj[selected_ref_idx].current_pcm[selected_idx_j, selected_idx_i]

        # Update mark matrix, and lbl_pairs
        reference_obj[selected_ref_idx].mark(
            reference_obj[selected_ref_idx].marked_pairs, selected_idx_i, selected_idx_j, True)
        reference_obj[selected_ref_idx].mark(
            reference_obj[selected_ref_idx].lbl_pairs, selected_idx_i, selected_idx_j, True)

        # Update the selected current_pcm with the prediction_pcm
        reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                    selected_idx_j] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_i, selected_idx_j]
        reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                    selected_idx_i] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_j, selected_idx_i]

        # Update BT scores for the selected reference
        reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
            reference_obj[selected_ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array(ref_1.current_prob, ref_2.current_prob,
                                     ref_3.current_prob, ref_4.current_prob, ref_5.current_prob)

        # Calculate PLCC between all the current_pcm scores and the groundtruth scores
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)
        if (plcc < 0.995):
            reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                        selected_idx_j] = temp_ij
            reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                        selected_idx_i] = temp_ji
            reference_obj[selected_ref_idx].mark(
                reference_obj[selected_ref_idx].lbl_pairs, selected_idx_i, selected_idx_j, True)
            reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
                reference_obj[selected_ref_idx].current_pcm)
            print("Reverting labling")

        iteration = iteration + 1
        plcc_entropy.append(plcc)
        srocc_entropy.append(srocc)

        return plcc_entropy, srocc_entropy


def run_random_labeling():

    REF = 5
    CONDITIONS = 16
    plcc_random = []
    srocc_random = []
    ref_1 = RandomClass('data1')
    ref_2 = RandomClass('data2')
    ref_3 = RandomClass('data3')
    ref_4 = RandomClass('data4')
    ref_5 = RandomClass('data5')
    gth_probs_arr = np.array(ref_1.gth_prob, ref_2.gth_prob,
                             ref_3.gth_prob, ref_4.gth_prob, ref_5.gth_prob)

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5]
    iteration = 0

    # Taken out three reference as test set, (n(n-1)/2)*12 = 1400 pairs remains in the dataset
    max_iter = 1400

    while (iteration < max_iter):
        # Select one reference randomly
        ref_idx = random.randint(1, 15)
        while (reference_obj[ref_idx].finished == True):
            ref_idx = random.randint(1, 15)

        # Select one pair from the reference randomly
        idx_i, idx_j = reference_obj[ref_idx].select_randomly()

        # Update mark and current matrix
        reference_obj[ref_idx].mark(
            reference_obj[ref_idx].marked_pairs, idx_i, idx_j, True)
        reference_obj[ref_idx].mark(
            reference_obj[ref_idx].lbl_pairs, idx_i, idx_j, True)

        reference_obj[ref_idx].current_pcm[idx_i, idx_j] = 0.5
        reference_obj[ref_idx].current_pcm[idx_j, idx_i] = 0.5

        # Update BT scores for the selected reference
        reference_obj[ref_idx].current_prob, _ = reference_obj[ref_idx].get_scores(
            reference_obj[ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array(ref_1.current_prob, ref_2.current_prob,
                                     ref_3.current_prob, ref_4.current_prob, ref_5.current_prob)

        # Calculate PLCC
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)

        plcc_random.append(plcc)
        srocc_random.append(srocc)

        # If every pair of the choden reference is selected, mark it as finish
        if (np.all(reference_obj[ref_idx].marked_pairs == True)):
            reference_obj[ref_idx].finished = True
        iteration = iteration + 1
    return plcc_random, srocc_random

def repeat_rand_labling():
    plcc_repeat_rnd = np.zeros((50,1799))
    srocc_repeat_rnd = np.zeros((50,1799))
    for repeat in range(50):
        plcc_random, srocc_random =  run_random_labeling()
        plcc_repeat_rnd[repeat: ] = plcc_random
        srocc_repeat_rnd[repeat: ] = srocc_random
    avg_plcc_random = []
    avg_plcc_random = np.mean(plcc_repeat_rnd, axis=0)  

    avg_srocc_random = []
    avg_srocc_random = np.mean(srocc_repeat_rnd, axis=0) 

    return avg_plcc_random, avg_srocc_random

def plot_labeling_approaches(plcc_entropy, srocc_entropy, plcc_kld, srocc_kld, avg_plcc_random, avg_srocc_random):
    num_entropy = [i for i in range(len(plcc_entropy))]
    num_kld = [i for i in range(len(plcc_kld))]
    num_rnd = [i for i in range(len(avg_plcc_random))]


    fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4))

    ax[0].plot(num_entropy, plcc_entropy, 'g-', label='Entropy-based labeling')
    ax[0].plot(num_kld, plcc_kld, 'm-', label='KLD-based labeling')
    ax[0].plot(num_rnd, avg_plcc_random, 'c-', label='Random-based labeling')



    ax[0].legend(fontsize=13)
    ax[0].set_xlabel('Number of $\it{predict}$ pairs', fontsize=14)
    ax[0].set_ylabel('PLCC', fontsize=14)
    ax[0].grid()


    ax[1].plot(num_entropy, srocc_entropy, 'g-', label='Entropy-based labeling')
    ax[1].plot(num_kld, srocc_kld, 'm-', label='KLD-based labeling')
    ax[1].plot(num_rnd, avg_srocc_random, 'c-', label='Random-based labeling')



    ax[1].legend(fontsize=13)
    ax[1].set_xlabel('Number of $\it{predict}$ pairs', fontsize=14)
    ax[1].set_ylabel('SROCC', fontsize=14)
    ax[1].grid()
    plt.savefig("labeling.pdf", pad_inches=0.01, bbox_inches="tight")

def main():
    plcc_entropy, srocc_entropy = run_entropy_labeling()
    plcc_kld, srocc_kld = run_kld_labeling()
    avg_plcc_random, avg_srocc_random = repeat_rand_labling()

    plot_labeling_approaches(plcc_entropy, srocc_entropy, plcc_kld, srocc_kld, avg_plcc_random, avg_srocc_random)

if __name__ == '__main__':
    main()
