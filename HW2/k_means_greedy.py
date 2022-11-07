import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl

mpl.use('macosx')


class KmeansClustering:

    def __init__(self, input_data: np.ndarray, clusters: int):

        self.k = clusters
        self.data = self.min_max_scale(input_data)
        self.centroids = self.assign_initial_centroids()
        self.labels = []
        self.assign_cluster()
        self.distance_criterion = 0.01

    @staticmethod
    def normalize_data(input_data):
        """
        Normalization.
        """
        [n, m] = np.shape(input_data)

        for index in range(m):
            data_min, data_max = np.min(input_data[:, index]), np.max(input_data[:, index])
            input_data[:, index] = (input_data[:, index] - data_min * np.ones(n) / (data_max - data_min))
        return input_data

    @staticmethod
    def min_max_scale(input_data):
        """
        Min max scaler.
        """
        [_, m] = np.shape(input_data)

        for index in range(m):
            input_data[:, index] = input_data[:, index] / np.max(np.abs(input_data[:, index]))
        return input_data

    @staticmethod
    def euclidean_distance(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
        """
        Compute euclidean distance.
        """
        return np.sum((a_mat[:, None, :] - b_mat[None, :, :]) ** 2, axis=2) ** 0.5

    def assign_cluster(self):
        """
        Assign data to certain cluster based on min distance.
        """
        self.labels = np.argmin(self.euclidean_distance(self.data, self.centroids), axis=1)
        return None

    def assign_initial_centroids(self) -> np.array:
        """
        Assign random centroids.
        """
        samples = np.random.choice(range(self.data.shape[0]), self.k, replace=False)
        return np.array([self.data[index] for index in samples])

    def stop_criterion(self, old_centroids: np.array) -> bool:
        """
        Stopping criterion based on centroid movement.
        """
        return np.max(self.euclidean_distance(self.centroids, old_centroids).diagonal()) <= self.distance_criterion

    def objective_function(self):
        """
        Cost Function.
        """
        cost = 0
        for cluster in range(self.k):
            distances = self.euclidean_distance(np.array(self.data[self.labels == cluster, :]),
                                                np.array(self.centroids))
            cost += np.var(distances[:, cluster])
        return cost

    def determine_centroids(self):
        """
        Mean value of each cluster.
        """
        self.centroids = np.array([np.mean(self.data[self.labels == key], axis=0) for key in range(self.k)])
        return None

    def run(self):
        """
        Perform single iteration of greedy algorithm.
        """
        self.determine_centroids()
        self.assign_cluster()
        return None

    def visualize_clusters_3d(self):  # only for data with 3 features
        """
        Visualization of data. Plot clusters.
        """
        ax = plt.axes(projection="3d")
        for i in range(self.k):

            cluster = self.data[self.labels == i]
            x_points = cluster[:, 0]
            y_points = cluster[:, 1]
            z_points = cluster[:, 2]
            ax.scatter3D(x_points, y_points, z_points, s=80, linewidths=5.0)
            ax.set_xlabel('Strength [MPa]', fontsize=12)
            ax.set_ylabel('Failure [%]', fontsize=12)
            ax.set_zlabel('Young\'s Modulus[GPa]', fontsize=12)
            ax.grid(False)

        plt.show()
        return None


if __name__ == '__main__':

    data = np.load("data.npy")
    strength = data[:, 0]           # strength [MPa]
    strain_failure = data[:, 1]     # [%]
    elastic_modulus = data[:, 2]    # [GPa]
    k_size = 2
    max_iterations = 100

    k_means = KmeansClustering(data, k_size)
    objective = []

    for _ in range(max_iterations):
        k_means.run()
        objective.append(k_means.objective_function())

    k_means.visualize_clusters_3d()
    plt.plot(objective)
    plt.show()
