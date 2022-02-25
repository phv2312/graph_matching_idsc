import numpy as np

from libs.interfaces import Node, Character


class SimilarityMeasure:
    MAX_DISTANCE = 30  # maximum distance for edge to be taken into account

    @staticmethod
    def calculate_for_node(node_i: Node, node_i_hat:Node, sigma1: float):
        """
        Calculate the similarity between two nodes
        :param node_i:
        :param node_i_hat:
        :param sigma1: float
        """

        from scipy.spatial.distance import cosine
        distance = cosine(node_i.feature.flatten(), node_i_hat.feature.flatten())
        sim = np.exp(-(distance / sigma1) ** 2)
        return sim

    @staticmethod
    def calculate_for_edge(node_i: Node, node_j: Node, node_i_hat: Node, node_j_hat: Node,
                           sigma2: float, sigma3: float):
        """
        Calculate the similarity between edge (i, j) in reference & (i_hat, j_hat) in target.
        where,
        - edge k: from node_i to node_j
        - edge l: from node_i_hat to node_j_hat
        :param node_i
        :param node_j
        :param node_i_hat
        :param node_j_hat:
        :param sigma2:
        :param sigma3:
        """

        d_k = np.linalg.norm(node_i.centroid - node_j.centroid, ord=2)
        d_l = np.linalg.norm(node_i_hat.centroid - node_j_hat.centroid, ord=2)
        d_k_l = (d_k - d_l) / max(d_k, d_l)

        distance = np.abs(d_k - d_l)
        if distance > SimilarityMeasure.MAX_DISTANCE:
            return 0.

        a_k_l = (node_i.area * node_j_hat.area - node_i_hat.area * node_j.area) / max(node_i.area * node_j_hat.area,
                                                                                      node_i_hat.area * node_j.area)
        sim = np.exp(- d_k_l**2 / (2 * sigma2**2) - a_k_l**2 / (2 * sigma3**2))

        return sim


class SMSolver:
    def __init__(self, reference: Character, target: Character):
        self.reference = reference
        self.target = target

    def setup_affinity_mat(self):
        """
        Initialize the matrix M, aim to find variable x which satisfy x'Mx
        """

        n_reference = len(self.reference)
        n_target = len(self.target)

        # get all the permutation
        range_reference = np.arange(0, n_reference)
        range_target = np.arange(0, n_target)
        qs, ps = np.meshgrid(range_target, range_reference)
        candidates = [(p, q) for p, q in zip(ps.flatten(), qs.flatten())]
        n_candidate = len(candidates)

        # calculate Affinity Matrix
        M = np.zeros(shape=(n_candidate, n_candidate), dtype=np.float32)
        for a in range(n_candidate):
            for b in range(n_candidate):
                i, i_hat = candidates[a]
                j, j_hat = candidates[b]

                if i == j and i_hat != j_hat:
                    continue

                if i != j and i_hat == j_hat:
                    continue

                if a == b:
                    # Node similarity
                    node_i = self.reference[i]
                    node_i_hat = self.target[i_hat]
                    M[a, a] = SimilarityMeasure.calculate_for_node(node_i, node_i_hat,
                                                                   sigma1=1.4)
                else:
                    # Edge similarity
                    node_i = self.reference[i]
                    node_i_hat = self.target[i_hat]
                    node_j = self.reference[j]
                    node_j_hat = self.target[j_hat]

                    M[a, b] = SimilarityMeasure.calculate_for_edge(node_i, node_j, node_i_hat, node_j_hat,
                                                                   sigma2=0.9, sigma3=0.45)

        return M, candidates

    def solve_using_spectral_matching(self, M, candidates, n_reference, n_target):
        """
        Finding the best cluster which maximize x'Mx
        :param M: (n_candidate, n_candidate)
        :param candidates: all the correct assignments
        :param n_reference: number of reference nodes
        :param n_target: number of target nodes
        """
        n_candidate = len(candidates)

        #
        x = np.zeros(shape=(n_candidate,))
        _, x_star = np.linalg.eig(M)
        x_star = x_star[:, 0]  # -> the principal eigenvector

        # make double stochastic
        x_star0 = x_star.copy().reshape((n_reference, n_target))
        x_star1 = x_star.copy().reshape((n_reference, n_target))

        for _ in range(5):
            x_star0 = x_star0 / (1e-6 + np.sum(x_star0, axis=1)[:, None])
            x_star1 = x_star1 / (1e-6 + np.sum(x_star1, axis=0)[None, :])

        x_star = (x_star0.flatten() + x_star1.flatten()) / 2.
        x_star = x_star / np.linalg.norm(x_star, ord=2)

        best_candidates = []
        is_end = False

        # greedy algorithm
        still_valid_ids = np.arange(n_candidate)
        while not is_end:
            #
            a_star = still_valid_ids[int(np.argmax(x_star[still_valid_ids]))]
            if x_star[a_star] == 0:
                # stop and return the x
                is_end = True
            else:
                # remove from candidates all the conflict assignments with a_star
                i, i_hat = candidates[a_star]

                # one-to-many constraint
                conflicted_assignments_ids = [i for i, a in enumerate(candidates) if a[1] == i_hat]

                #
                x[a_star] = 1.
                x[conflicted_assignments_ids] = 1.
                best_candidates += [(i, i_hat)]  # tuple(index_reference, index_target)

            #
            still_valid_ids = np.where(x != 1)[0]
            if len(still_valid_ids) < 1:
                is_end = True

        return best_candidates

