import networkx as nx
import numpy as np
from statistics import mean


NODES = 9


np.set_printoptions(precision=3)


def score(g):
    g_mat = nx.to_numpy_array(g)
    defective = 2 - g_mat @ g_mat
    np.fill_diagonal(defective, 0)
    bad_count = (defective * defective).sum() / 2
    return bad_count, defective


def construct_graph_from_genome(genome):
    g_mat = (genome >= np.random.rand(NODES, NODES)).astype(int)
    return nx.from_numpy_array(g_mat)


def adjust(genome, delta=0.001):
    results = []
    TRIALS = 10
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        results.append((*score(g), g))
    for _, defects, g in results:
        genome = np.clip(genome + defects * delta, 0, 1)
    return genome, results


def create_genome():
    return np.tri(NODES, NODES, -1).T * np.random.rand(NODES, NODES)


def main():
    genome = create_genome()
    best_avg_score = float('inf')
    min_score = float('inf')
    while min_score != 0:
        genome, results = adjust(genome)
        scores, _, _ = zip(*results)
        if mean(scores) < best_avg_score:
            best_avg_score = mean(scores)
            min_score = min(scores)
            print(min_score, best_avg_score)


if __name__ == "__main__":
    main()
