import networkx as nx
import matplotlib.pyplot as plt
import random
import time

def generate_sparse_graph(n):
    # Create an empty graph with n nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add random edges until the graph is connected
    while not nx.is_connected(G):
        # Add random edge between two nodes from different connected components
        components = list(nx.connected_components(G))
        node1 = random.choice(list(components[0]))
        node2 = random.choice(list(components[1]))
        weight = random.randint(0, 100)  # Assign random weight to the edge
        G.add_edge(node1, node2, weight=weight)

    return G

def generate_dense_graph(n):
    # Create a complete graph
    G = nx.complete_graph(n)

    # Remove random edges
    edges = list(G.edges())
    random.shuffle(edges)
    num_edges_to_remove = int(0.2 * n * (n - 1) / 2)  # Adjust the density as desired
    edges_to_remove = edges[:num_edges_to_remove]
    G.remove_edges_from(edges_to_remove)

    # Assign random weights to the remaining edges
    for u, v in G.edges():
        weight = random.randint(0, 100)  # Assign random weight to the edge
        G[u][v]['weight'] = weight

    return G

def plot_graph(G):
    pos = nx.spring_layout(G)  # Layout for better visualization
    labels = nx.get_edge_attributes(G, 'weight')  # Get edge weights as labels

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', width=1.0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()

def prim_algorithm(G):
    # Initialize the MST and the set of nodes that have been added to the MST
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes())
    nodes_added = set()

    # Choose an arbitrary node to start from
    start_node = list(G.nodes())[0]
    nodes_added.add(start_node)

    # Keep adding edges until all nodes have been added to the MST
    while len(nodes_added) < G.number_of_nodes():
        # Find the edge with minimum weight that connects a node in the MST with a node outside the MST
        min_edge = None
        min_weight = float('inf')
        for u in nodes_added:
            for v in G.neighbors(u):
                if v not in nodes_added:
                    weight = G[u][v]['weight']
                    if weight < min_weight:
                        min_edge = (u, v)
                        min_weight = weight

        # Add the edge to the MST
        mst.add_edge(*min_edge, weight=min_weight)
        nodes_added.add(min_edge[1])

    return mst

def kruskal_algorithm(G):
    # Initialize the MST and the set of edges that have been added to the MST
    mst = nx.Graph()
    mst.add_nodes_from(G.nodes())
    edges_added = set()

    # Sort the edges by weight
    edges = list(G.edges(data=True))
    edges.sort(key=lambda x: x[2]['weight'])

    # Keep adding edges until all nodes are connected
    for u, v, data in edges:
        if not mst.has_edge(u, v):
            mst.add_edge(u, v, **data)
            edges_added.add((u, v))
            if nx.is_connected(mst):
                break

    return mst


def prim(G):
    start_time = time.time()
    # find all pairs of shortest paths
    mst = prim_algorithm(G)

    return time.time() - start_time + 0.1


def kruskal(G):
    start_time = time.time()
    # find all pairs of shortest paths
    mst = kruskal_algorithm(G)
    return time.time() - start_time + 0.1


def run(n):
    sparse_graph = generate_sparse_graph(n)
    dense_graph = generate_dense_graph(n)

    p1 = prim(sparse_graph)
    k1 = kruskal(sparse_graph)

    print("SPARSE GRAPH, n =", n)
    print("Prim time:", p1)
    print("Kruskal time:", k1)

    p2 = prim(dense_graph)
    k2 = kruskal(dense_graph)

    print("\n")
    print("DENSE GRAPH, n =", n)
    print("Prim time:", p2)
    print("Kruskal time:", k2)

    return p1, k1, p2, k2


n = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]

prim_time_sparse = []
kruskal_time_sparse = []

prim_time_dense = []
kruskal_time_dense = []


for i in n:
    result = run(i)

    prim_time_sparse.append(result[0])
    kruskal_time_sparse.append(result[1])

    prim_time_dense.append(result[2])
    kruskal_time_dense.append(result[3])

print("\n\n")
print(prim_time_sparse)
print(kruskal_time_sparse)
print("\n")
print(prim_time_dense)
print(kruskal_time_dense)


# Plotting sparse results
plt.figure(figsize=(8, 6))
plt.plot(n, prim_time_sparse, label='Prim')
plt.plot(n, kruskal_time_sparse, label='Kruskal')
plt.xlabel('Number of vertices')
plt.ylabel('Time')
plt.title('Sparse Graph Results')
plt.legend()
plt.grid(True)
plt.show()

# Plotting dense results
plt.figure(figsize=(8, 6))
plt.plot(n, prim_time_dense, label='Prim')
plt.plot(n, kruskal_time_dense, label='Kruskal')
plt.xlabel('Number of vertices')
plt.ylabel('Time')
plt.title('Dense Graph Results')
plt.legend()
plt.grid(True)
plt.show()