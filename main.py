import numpy as np
from itertools import combinations


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
    g = g + g.T
    return g

def search(genome, delta=0.01):
    trial_results = []
    TRIALS = 10
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        s, defects = score(g)
        trial_results.append((s, g, defects))
    for s, g, defects in trial_results:
        up_g = np.tri(NODES, NODES, -1).T * (2 * g - 1)
        genome = np.clip(genome + up_g * (1 / s) * delta, 0, 1)
    return genome, \
           *min(trial_results, key=lambda x: x[0])



def adjust(genome, delta=0.00001):
    trial_results = []
    TRIALS = 1000
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        s, defects = score(g)
        trial_results.append((s, g, defects))
    for s, g, defects in trial_results:
        apply = np.logical_or(
            np.logical_and(defects < 0, genome > 0.5),
            np.logical_and(defects > 0, genome <= 0.5)).astype(int)
        genome = np.clip(genome + apply * defects * (1 / s) * delta, 0, 1)
        for u, v, k in combinations(range(NODES), 3):
            genome[u, k] = np.clip(
                genome[u, k] + apply[u, k] * defects[u, v] * (1 / s) * delta,
                0, 1
                )
            genome[v, k] = np.clip(
                genome[v, k] + apply[v, k] * defects[u, v] * (1 / s) * delta,
                0, 1
            )
    return genome, \
           *min(trial_results, key=lambda x: x[0])


def create_genome():
    return np.tri(NODES, NODES, -1).T * 0.5


def main():
    genome = create_genome()
    best_score = float('inf')
    best_graph = None
    pairs = NODES * (NODES - 1) / 2
    CERTAINTY_UPPER_BOUND = 0.99
    CERTAINTY_LOWER_BOUND = 0.7
    # Have a certainty score (unlikelyness to change) normalized so that 0.5
    # in all entries is 1 and no certainty is 0.
    certainty = 1 - np.minimum(1 - genome, genome).sum() / (pairs / 2)
    while best_score != 0:
        print("PHASE 1")
        while certainty < CERTAINTY_UPPER_BOUND:
            genome, score, graph, defects = search(genome)
            certainty = 1 - np.minimum(1 - genome, genome).sum() / (pairs / 2)
            if score < best_score:
                best_score = score
                best_graph = graph
                print(best_score)
                if best_score == 0:
                    break 
        print("PHASE 2")
        while certainty > CERTAINTY_LOWER_BOUND:
            genome, score, graph, defects = adjust(genome)
            certainty = 1 - np.minimum(1 - genome, genome).sum() / (pairs / 2)
            if score < best_score:
                best_score = score
                best_graph = graph
                print(best_score)
            print(genome)
    print(best_graph)


if __name__ == "__main__":
    main()
