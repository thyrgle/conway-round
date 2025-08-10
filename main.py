import numpy as np


NODES = 9


np.set_printoptions(precision=3)


def score(g):
    defective = -(g @ g - (2 - g))
    np.fill_diagonal(defective, 0)
    defective = np.tri(NODES, NODES, -1).T * defective
    bad_count = (defective ** 2).sum()
    return bad_count, defective


def construct_graph_from_genome(genome):
    g = (genome >= np.random.rand(NODES, NODES)).astype(int)
    return g


def adjust(genome, delta=0.0005):
    trial_results = []
    TRIALS = 1000
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        s, defects = score(g)
        trial_results.append((s, g, defects))
    for s, g, defects in trial_results:
        up_g = np.tri(NODES, NODES, -1).T * (2 * g - 1)
        genome = np.clip(genome + up_g * (1 / s) * delta, 0.0, 1.0)
        genome = np.tri(NODES, NODES, -1).T * genome
    return genome, *min(trial_results, key=lambda x: x[0])


def create_genome():
    return np.tri(NODES, NODES, -1).T * 0.5


def main():
    genome = create_genome()
    best_score = float('inf')
    best_graph = None
    while best_score != 0:
        genome, score, graph, defects = adjust(genome)
        if score < best_score:
            best_score = score
            best_graph = graph
            print(best_score)
    print(best_graph)


if __name__ == "__main__":
    main()
