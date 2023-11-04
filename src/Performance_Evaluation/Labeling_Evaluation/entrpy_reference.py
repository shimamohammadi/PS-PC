import numpy as np

from src.Training.Groundtruth_data_collection.parent_class import ParentClass


class EntropyClass(ParentClass):
    """This is the class definintion for each reference content
    """

    def __init__(self, df_test_features, ref_name):
        super().__init__(df_test_features, ref_name)

        self.entropy_matrix = self.entropy_estimation()

    def select_max_entropy(self):
        """This function returns the maximum value in the entropy matrix and its coresponding indexs
        """

        entropy = np.ma.MaskedArray(self.entropy_matrix, self.marked_pairs)
        i, j = np.unravel_index(entropy.argmax(), entropy.shape)
        maximum = entropy[i, j]
        return maximum, i, j

    def entropy_estimation(self):
        """This function claculate entroy for the pc matrix
        """
        entropy_matrix = np.zeros((self.CONDITIONS, self.CONDITIONS))
        temp = np.subtract(1, self.gth_pcm)

        entropy_matrix = - \
            np.multiply(self.gth_pcm, np.log2(self.gth_pcm)) - \
            np.multiply(temp, np.log2(temp))
        entropy_matrix[np.isnan(entropy_matrix)] = 0

        return entropy_matrix
