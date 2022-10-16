import matplotlib.pyplot as plt
import numpy as np
from k_means_greedy import KmeansClustering
from sklearn.datasets import make_blobs
from tqdm import tqdm

if __name__ == "__main__":

    data = np.load("data.npy")
    strength = data[:, 0]               # strength [MPa]
    strain_failure = data[:, 1]         # [%]
    elastic_modulus = data[:, 2]        # [GPa]
    k_size = 3
    max_iterations = 500
    [n, m] = np.shape(data)

    # data, _ = make_blobs(n_samples=200, n_features=m, centers=k_size)
    k_means = KmeansClustering(data, k_size)
    states = np.arange(k_size)
    hamiltonian = k_means.objective_function()
    energy = []
    energy.append(hamiltonian)
    temperature = 2

    for _ in tqdm(range(max_iterations)):

        sample_element = np.random.choice(range(n), 1, replace=False)
        previous_state = k_means.labels[sample_element]
        new_state = np.random.choice(states[states != previous_state], 1)
        k_means.labels[sample_element] = int(new_state)
        new_hamiltonian = k_means.objective_function()
        de = new_hamiltonian - hamiltonian
        temperature *= 0.93

        if de <= 0 or np.exp(- de / temperature) > np.random.random():
            hamiltonian = new_hamiltonian
        else:
            k_means.labels[sample_element] = previous_state
        energy.append(hamiltonian)

    plt.plot(energy)
    plt.show()
    k_means.determine_centroids()
    k_means.assign_cluster()
    k_means.visualize_clusters_3d()
