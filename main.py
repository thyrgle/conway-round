import networkx as nx
import numpy as np


NODES = 9


np.set_printoptions(precision=3)


def score(g):
    g_mat = nx.to_numpy_array(g)
    defective = -(g_mat @ g_mat - (2 - g_mat))
    np.fill_diagonal(defective, 0)
    defective = np.tri(NODES, NODES, -1).T * defective
    bad_count = (defective ** 2).sum()
    return bad_count, defective


def construct_graph_from_genome(genome):
    g_mat = (genome >= np.random.rand(NODES, NODES)).astype(int)
    return nx.from_numpy_array(g_mat)


def adjust(genome, delta=0.00001):
    results = []
    TRIALS = 10
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        results.append((*score(g), g))
    for _, defects, g in results:
        genome = np.clip(genome + defects * delta, 0.0, 1.0)
    return genome, *zip(*results)


def create_genome():
    return np.tri(NODES, NODES, -1).T * np.random.rand(NODES, NODES)


def graph_to_genome(g):
    return np.tri(NODES, NODES, -1).T * nx.to_numpy_array(g)


def main():
    genome = create_genome()
    best_score = float('inf')
    while best_score != 0:
        genome, scores, _, graphs = adjust(genome)
        if min(scores) < best_score:
            best_score = min(scores)
            print(best_score)


if __name__ == "__main__":
    main()
