import numpy as np


def initialize_centroids_forgy(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False), :]


def initialize_centroids_kmeans_pp(data, k):
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0]), :]
    for i in range(1, k):
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            distances[:, j] = np.sqrt(np.sum((data - centroids[j]) ** 2, axis=1))

        max_distances = np.sum(distances, axis=1)
        index = np.argmax(max_distances)
        centroids[i] = data[index]
    return centroids


def assign_to_cluster(data, centroids):
    distances = np.sqrt(np.sum((data[:, np.newaxis, :] - centroids) ** 2, axis=2))
    return np.argmin(distances, axis=1)


def update_centroids(data, assignments):
    return np.array(
        [
            np.mean(data[assignments == k], axis=0)
            for k in range(len(np.unique(assignments)))
        ]
    )


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(
            f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}"
        )
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return (
        new_assignments,
        centroids,
        mean_intra_distance(data, new_assignments, centroids),
    )
