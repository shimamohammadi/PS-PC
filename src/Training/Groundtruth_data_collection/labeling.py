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
                ["ref_"+ref_name+"_Img"+str(row+1), "ref_"+ref_name+"_Img"+str(col+1), labling_res[row, col]])


def run_kld_labeling():

    # There are 15 reference in PC-IQA database. Each time three of them is kept for evaluation
    # df_data1 = pd.read_csv(
    #     './src/data/IQA features/reference_1_features.csv', header=0, sep=",")
    # df_data2 = pd.read_csv(
    #     './src/data/IQA features/reference_2_features.csv', header=0, sep=",")
    # df_data3 = pd.read_csv(
    #     './src/data/IQA features/reference_3_features.csv', header=0, sep=",")
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

    REF = 12
    CONDITIONS = 16
    plcc_kld = []
    srocc_kld = []
    stage = 'gth_labeling'
    # ref1,2,3 are kept for evaluation
    # ref_1 = KLDClass(df_data1, 'data1', stage)
    # ref_2 = KLDClass(df_data2, 'data2', stage)
    # ref_3 = KLDClass(df_data3, 'data3', stage)
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
        ref_4.gth_prob, ref_5.gth_prob, ref_6.gth_prob,
        ref_7.gth_prob, ref_8.gth_prob, ref_9.gth_prob,
        ref_10.gth_prob, ref_11.gth_prob, ref_12.gth_prob,
        ref_13.gth_prob, ref_14.gth_prob, ref_15.gth_prob])

    # available reference objects
    reference_obj = [ref_4, ref_5, ref_6, ref_7,
                     ref_8, ref_9, ref_10, ref_11, ref_12, ref_13, ref_14, ref_15]
    iteration = 0

    # Taken out three reference as test set, (n(n-1)/2)*12 = 1440 pairs remains in the dataset
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

        # (Label the selected pair as "Predict")
        reference_obj[selected_ref_idx].mark(
            'marked_pairs', selected_idx_i, selected_idx_j, True)

        reference_obj[selected_ref_idx].mark(
            'lbl_pairs', selected_idx_i, selected_idx_j, True)

        # (Get the result from the prediction output for the selected "Predict" pair)
        reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                    selected_idx_j] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_i, selected_idx_j]
        reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                    selected_idx_i] = reference_obj[selected_ref_idx].prediction_pcm[selected_idx_j, selected_idx_i]

        # (Inferr scores for the PCM of the selected pair after lebeling)
        reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
            reference_obj[selected_ref_idx].current_pcm)

        # An array of all the scores for all the references
        current_probs_arr = np.array([
            ref_4.current_prob, ref_5.current_prob, ref_6.current_prob,
            ref_7.current_prob, ref_8.current_prob, ref_9.current_prob,
            ref_10.current_prob, ref_11.current_prob, ref_12.current_prob,
            ref_13.current_prob, ref_14.current_prob, ref_15.current_prob])

        # Calculate PLCC between all the current_pcm scores and the groundtruth scores
        plcc, _ = pearsonr(current_probs_arr, gth_probs_arr)
        srocc, _ = spearmanr(current_probs_arr, gth_probs_arr)

        # The 0.995 is the eta. By changing the eta diferrent results is obtaind.
        if (plcc < 0.995):
            reference_obj[selected_ref_idx].current_pcm[selected_idx_i,
                                                        selected_idx_j] = temp_ij
            reference_obj[selected_ref_idx].current_pcm[selected_idx_j,
                                                        selected_idx_i] = temp_ji
            reference_obj[selected_ref_idx].mark(
                'lbl_pairs', selected_idx_i, selected_idx_j, False)
            reference_obj[selected_ref_idx].current_prob, reference_obj[selected_ref_idx].current_std = reference_obj[selected_ref_idx].get_scores(
                reference_obj[selected_ref_idx].current_pcm)
            print("Reverting labling")

        # Update kld
        # reference_obj[selected_ref_idx].create_kld_matrix()
        print(iteration, selected_ref_idx, selected_idx_i, selected_idx_j)
        iteration = iteration + 1

    return reference_obj, plcc_kld, srocc_kld


def main():

    reference_obj, plcc_kld, srocc_kld = run_kld_labeling()
    with open('kld_labeling.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Pair_a', 'Pair_b', 'Defer'])

        # write_to_csv(reference_obj[0].lbl_pairs, writer, "1")
        # write_to_csv(reference_obj[1].lbl_pairs, writer, "2")
        # write_to_csv(reference_obj[2].lbl_pairs, writer, "3")
        write_to_csv(reference_obj[0].lbl_pairs, writer, "4")
        write_to_csv(reference_obj[1].lbl_pairs, writer, "5")
        write_to_csv(reference_obj[2].lbl_pairs, writer, "6")
        write_to_csv(reference_obj[3].lbl_pairs, writer, "7")
        write_to_csv(reference_obj[4].lbl_pairs, writer, "8")
        write_to_csv(reference_obj[5].lbl_pairs, writer, "9")
        write_to_csv(reference_obj[6].lbl_pairs, writer, "10")
        write_to_csv(reference_obj[7].lbl_pairs, writer, "11")
        write_to_csv(reference_obj[8].lbl_pairs, writer, "12")
        write_to_csv(reference_obj[9].lbl_pairs, writer, "13")
        write_to_csv(reference_obj[10].lbl_pairs, writer, "14")
        write_to_csv(reference_obj[11].lbl_pairs, writer, "15")


if __name__ == '__main__':
    main()
