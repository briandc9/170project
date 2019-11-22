import networkx as nx
import numpy as np
import utils as utils
import sys
import matplotlib.pyplot as plt

def inputmaker(n):
    if n == 50:
        f = open("50.in", "w")
    elif n == 100:
        f = open("100.in", "w")
    elif n == 200:
        f = open("200.in", "w")
    f.write(str(n))
    f.write("\n")
    f.write(str(int(n / 2)))
    f.write("\n")
    f.write(str(0))
    for i in np.arange(1, n):
        f.write(" ")
        f.write(str(i))
    f.write("\n")
    f.write(str(0))
    for i in np.arange(2, n, 2):
        f.write(" ")
        f.write(str(i))
    f.write("\n")
    f.write(str(0))
    f.write("\n")
    for i in np.arange(n):
        for j in np.arange(n):
            f.write(str(adjlist[i][j]))
            if not j == n - 1:
                f.write(" ")
        if not i == n - 1:
            f.write("\n")
    f.close()
    
def outputmaker(n):
    if n == 50:
        f = open("50.out", "w")
    elif n == 100:
        f = open("100.out", "w")
    elif n == 200:
        f = open("200.out", "w")
    f.write(str(0))
    f.write("\n")
    f.write(str(1))
    f.write("\n")
    f.write(str(0))
    for i in np.arange(2, n, 2):
        f.write(" ")
        f.write(str(i))
    f.write(" ")
    f.write(str(0))
    f.close()

def set_weights(G):
    for u, v in G.edges:
        G[u][v]['cost'] = np.random.randint(1, 5000)
    
def set_AM(G):
    init = [['x' for i in np.arange(n)] for j in np.arange(n)]
    for u, v in G.edges:
        if shortest_lengths[u][v] < G[u][v]['cost']:
            G[u][v]['cost'] = shortest_lengths[u][v]
        init[u][v] = G[u][v]['cost']
        init[v][u] = G[v][u]['cost']
    for i in np.arange(n):
        for j in np.arange(n):
            if i == j or init[i][j] == 0:
                init[i][j] = 'x'
    return init

# n = 50
# n = 100
n = int(input("n: "))
# interchange probabilities
if n == 50:
    G = nx.gnp_random_graph(n, 0.5)
elif n == 100:
    G = nx.gnp_random_graph(n, 0.5)
elif n == 200:
    G = nx.gnp_random_graph(n, 0.5)
    
set_weights(G)
shortest_lengths = dict(nx.floyd_warshall(G, weight='cost'))

adjlist = set_AM(G)
adjlist = np.reshape(adjlist, (n, n))

inputmaker(n)
outputmaker(n)
