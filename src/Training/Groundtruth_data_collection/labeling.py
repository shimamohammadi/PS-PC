import csv
import numpy as np
from scipy import linalg
import pandas as pd
from scipy.stats.mstats import pearsonr, spearmanr

from src.Training.Groundtruth_data_collection.kld_class import KLDClass


def write_to_csv(labling_res, writer, ref_name):
    for row in range(len(labling_res)):
        for col in range(row+1, len(labling_res)):
            writer.writerow(
                ["ref_"+ref_name+"_Img"+str(row+1), "ref_1_Img"+str(col+1), labling_res[row, col]])


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
    REF = 5
    CONDITIONS = 16
    plcc_kld = []
    srocc_kld = []
    ref_1 = KLDClass(df_data1, 'data1')
    ref_2 = KLDClass(df_data2, 'data2')
    ref_3 = KLDClass(df_data3, 'data3')
    ref_4 = KLDClass(df_data4, 'data4')
    ref_5 = KLDClass(df_data5, 'data5')
    gth_probs_arr = np.array([ref_1.gth_prob, ref_2.gth_prob,
                             ref_3.gth_prob, ref_4.gth_prob, ref_5.gth_prob])

    # available reference objects
    reference_obj = [ref_1, ref_2, ref_3, ref_4, ref_5]
    iteration = 0

    # Taken out three reference as test set, (n(n-1)/2)*12 = 1400 pairs remains in the dataset
    max_iter = REF * CONDITIONS

    # Loop over every pair in the IQA dataset
    while (iteration < max_iter):
        selected_ref_idx = 0
        selected_min = 1000
        selected_idx_i = 0
        selected_idx_j = 0
        for index, ref in enumerate(reference_obj):

            min_kld, idx_i, idx_j = reference_obj[index].select_min_value_from_KLD_matrix(
            )
            # FInd the minimum kld between all the references
            if (min_kld < selected_min):
                selected_ref_idx = index
                selected_min = min_kld
                selected_idx_i = idx_i
                selected_idx_j = idx_j

        # update kld
        reference_obj[selected_ref_idx].create_kld_matrix()

        # Backup from the current_pcm
        temp_ij = reference_obj[selected_ref_idx].current_pcm[selected_idx_i, selected_idx_j]
        temp_ji = reference_obj[selected_ref_idx].current_pcm[selected_idx_j, selected_idx_i]

        # Update mark matrix, and lbl_pairs
        reference_obj[selected_ref_idx].mark(
            'marked_pairs', selected_idx_i, selected_idx_j, True)

        reference_obj[selected_ref_idx].mark(
            'lbl_pairs', selected_idx_i, selected_idx_j, True)

        # Update the selected current_pcm with the prediction_pcm
        reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                    selected_idx_j] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_i, selected_idx_j]
        reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                    selected_idx_i] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_j, selected_idx_i]

        # Update BT scores for the selected reference
        reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
            reference_obj[selected_ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array([ref_1.current_prob, ref_2.current_prob,
                                     ref_3.current_prob, ref_4.current_prob, ref_5.current_prob])

        # Calculate PLCC between all the current_pcm scores and the groundtruth scores
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)
        if (plcc < 0.995):
            reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                        selected_idx_j] = temp_ij
            reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                        selected_idx_i] = temp_ji
            reference_obj[selected_ref_idx].mark(
                'lbl_pairs', selected_idx_i, selected_idx_j, True)
            reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
                reference_obj[selected_ref_idx].current_pcm)
            print("Reverting labling")

        iteration = iteration + 1

    return reference_obj, plcc_kld, srocc_kld


def main():

    reference_obj, plcc_kld, srocc_kld = run_kld_labeling()
    with open('kld_labeling.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Pair_a', 'Pair_b', 'Defer'])

        write_to_csv(reference_obj[0].lbl_pairs, writer, "1")
        write_to_csv(reference_obj[1].lbl_pairs, writer, "2")
        write_to_csv(reference_obj[2].lbl_pairs, writer, "3")
        write_to_csv(reference_obj[3].lbl_pairs, writer, "4")
        write_to_csv(reference_obj[4].lbl_pairs, writer, "5")


if __name__ == '__main__':
    main()
