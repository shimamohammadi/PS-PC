import numpy as np
from src.Training.Groundtruth_data_collection.parent_class import ParentClass


class KLDClass(ParentClass):
    """This is the class definintion for each reference content.
    """

    def __init__(self, df_test_features, ref_name, stage):
        self.stage = stage
        super().__init__(df_test_features, ref_name)

        # The kld matrix is updated each time the reference object is chosen
        self.kld_matrix = self.create_kld_matrix()

    def create_kld_matrix(self):
        """This function temporary removes a pair in each iteration from the current_pcm, and calculate the kld
        """
        kld_matrix = np.zeros((self.CONDITIONS, self.CONDITIONS))
        for row in range(self.CONDITIONS):
            for col in range(row+1, self.CONDITIONS):

                # Storing values of the current_pcm before updating their values
                temp_1 = self.current_pcm[row, col]
                temp_2 = self.current_pcm[col, row]

                if (self.stage == 'labeling_compare'):
                    self.current_pcm[row, col] = 0.5
                    self.current_pcm[col, row] = 0.5
                else:
                    self.current_pcm[row, col] = self.prediction_pcm[row, col]
                    self.current_pcm[col, row] = self.prediction_pcm[col, row]

                # Infer scores from the updated current_pcm
                [p_tmp, pstd_tmp] = self.get_scores(
                    self.current_pcm)

                # Calculate KLD between prior(groundtruth) and posterior(current_pcm)
                kld_res = kl_divergence_approx(
                    self.gth_prob, self.gth_std, p_tmp, pstd_tmp)

                # Recovering the current_pcm
                self.current_pcm[row, col] = temp_1
                self.current_pcm[col, row] = temp_2

                kld_matrix[row, col] = kld_res - 16
                kld_matrix[col, row] = kld_res - 16
        return kld_matrix

    def select_min_value_from_KLD_matrix(self):
        """This function outputs the minimum value in the KLD matrix and the coresponding indexes 
        """
        kld = np.ma.array(self.kld_matrix, mask=self.marked_pairs)
        i, j = np.unravel_index(kld.argmin(), kld.shape)
        minimum = kld[i, j]
        return minimum, i, j


def kl_divergence_approx(mean_1, var_1, mean_2, var_2):
    '''
    Aproximation of the multivariate normal KL divergence: 
    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    '''
    total = np.sum(np.log(var_2)) - np.sum(np.log(var_1)) + \
        sum(var_1/var_2)+np.dot(1/var_2, (mean_1-mean_2)**2)
    return total
