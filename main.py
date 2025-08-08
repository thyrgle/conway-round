import networkx as nx
import numpy as np
from random import random
from statistics import mean


NODES = 9
EDGES = 4


def score(g):
    bad_count = 0
    defective = {}
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            if j > len(g):
                continue
            c = len(nx.common_neighbors(g, i, j))
            e = int(i in g.neighbors(j))
            defective[(i, j)] = -(c - (2 - e))
            bad_count += defective[(i, j)] * defective[(i, j)]
    return bad_count, defective


def construct_graph_from_genome(genome):
    g = nx.Graph()
    g.add_nodes_from(range(NODES))
    for (u, v), value in genome.items():
        if random() <= value:
            g.add_edge(u, v)
    return g


def adjust(genome, delta=0.000001):
    results = []
    TRIALS = 10
    for _ in range(TRIALS):
        g = construct_graph_from_genome(genome)
        results.append((*score(g), g))
    for _, defects, g in results:
        for (u, v), value in defects.items():
            for k in range(NODES):
                if k == u:
                    continue
                if k == v:
                    continue
                try:
                    genome[(u, k)] = np.clip(
                            genome[(u, k)] + value * delta)
                except KeyError:
                    genome[(k, u)] = np.clip(
                            genome[(k, u)] + value * delta)
                try:
                    genome[(v, k)] = np.clip(
                            genome[(v, k)] + value * delta)
                except KeyError:
                    genome[(k, v)] = np.clip(
                            genome[(k, v)] + value * delta)
    return min(results, key=lambda x: x[0]), \
           mean([r[0] for r in results]), \
           max(results, key=lambda x: x[0]), \
           genome


def create_genome():
    prob_table = {}
    for u in range(NODES):
        for v in range(u+1, NODES):
            if v > NODES:
                continue
            prob_table[(u, v)] = random()
    return prob_table


def main():
    genome = create_genome()
    score = float('inf')
    best_score = float('inf')
    min_score = float('inf')
    while min_score != 0:
        min_score, score, _, genome = adjust(genome)
        min_score = min_score[0]
        if score < best_score:
            best_score = score
            print(score)

 
if __name__ == "__main__":
    main()
