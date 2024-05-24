import networkx as nx
import math
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import random
import numpy as np

def similarity(G, v, u, cache):
    """Calculate the similarity between two nodes."""
    if (v, u) in cache or (u, v) in cache:
        return cache.get((v, u), cache.get((u, v)))
    
    v_set = set(G.neighbors(v)) | {v}
    u_set = set(G.neighbors(u)) | {u}
    inter = len(v_set & u_set)
    if not inter:
        return 0
    
    sim = inter / math.sqrt(len(v_set) * len(u_set))
    cache[(v, u)] = sim
    return sim

def neighborhood(G, v, eps, cache):
    """Find the neighborhood of a node based on similarity."""
    return [u for u in G.neighbors(v) if similarity(G, v, u, cache) > eps]
    
def process_nonmember(v, core_clusters, G):
    """Process each non-member node to find its closest core and update the cluster."""
    nearest_core = min(core_clusters, key=lambda core: nx.shortest_path_length(G, source=v, target=core))
    return (v, core_clusters[nearest_core])

def sameClusters(G, clusters, u):
    """Check if the neighbors of a node belong to the same cluster."""
    n = list(G.neighbors(u))
    b = []
    for neighbor in n:
        for k, v in clusters.items():
            if neighbor in v and k not in b:
                b.append(k)
    return len(b) <= 1

def closest_core(node, core_nodes, G):
    """Determine the closest core node's cluster for a given node."""
    min_distance = float('inf')
    nearest_core = None
    for core in core_nodes:
        distance = nx.shortest_path_length(G, source=node, target=core, weight='weight')
        if distance < min_distance:
            min_distance = distance
            nearest_core = core
    return nearest_core

def scan(G, eps=0.7, mu=2):
    """SCAN clustering algorithm."""
    clusters = {}
    node_labels = {}
    sim_cache = {}
    c = 0

    for n in G.nodes:
        if n in node_labels:  # Skip nodes that have already been processed
            continue
        N = neighborhood(G, n, eps, sim_cache)
        if len(N) >= mu:
            c += 1
            clusters[c] = [n]
            node_labels[n] = c
            Q = deque(N)

            while Q:
                w = Q.popleft()
                if w in node_labels:
                    continue
                R = neighborhood(G, w, eps, sim_cache) + [w]
                for s in R:
                    if s not in node_labels:
                        clusters[c].append(s)
                        node_labels[s] = c
                        Q.append(s)
        else:
            node_labels[n] = -1  # Mark non-core nodes distinctly

    # Label propagation for non-members
    flag = True
    while flag:
        flag = False
        for v, label in node_labels.items():
            if label == -1:
                flag = True
                cluster_membership = defaultdict(int)
                for neighbor in G.neighbors(v):
                    if neighbor in node_labels and node_labels[neighbor] != -1:
                        cluster_membership[node_labels[neighbor]] += 1
                if cluster_membership:
                    selected_cluster = max(cluster_membership, key=cluster_membership.get)
                    clusters[selected_cluster].append(v)
                    node_labels[v] = selected_cluster
    return clusters

def is_valid_partition(G, mods):
    """Check if the partition covers all nodes exactly once."""
    node_set = set(G.nodes())
    partition_nodes = set()
    for community in mods:
        community_set = set(community)
        if not partition_nodes.isdisjoint(community_set):
            return False, "Duplicate nodes found in multiple communities."
        partition_nodes.update(community_set)
    if partition_nodes != node_set:
        return False, f"Partition does not cover all nodes exactly once. Missing: {len(node_set - partition_nodes)}"
    return True, "Partition is valid."

memo = {}
def loss(G, eps, mu):
    """Calculate the loss function based on modularity."""
    if (eps, mu) not in memo:
        clusters = scan(G, eps, mu)
        mods = [list(map(int, cluster)) for cluster in clusters.values()]
        modularity = nx.algorithms.community.modularity(G, mods)
        memo[(eps, mu)] = -3 * modularity**3
    return memo[(eps, mu)]

def approximate_gradient(G, f, eps, mu, h=1e-5):
    """Approximate the gradient of a function using parallel computation."""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(f, G, eps + h, mu),
            executor.submit(f, G, eps - h, mu),
            executor.submit(f, G, eps, mu + h),
            executor.submit(f, G, eps, mu - h)
        ]
        results = [future.result() for future in futures]
    
    f_eps_plus_h, f_eps_minus_h, f_mu_plus_h, f_mu_minus_h = results
    dfdx = (f_eps_plus_h - f_eps_minus_h) / (2 * h)
    dfdy = (f_mu_plus_h - f_mu_minus_h) / (2 * h)
    return dfdx, dfdy

def gradient_descent_2d(G, f, initial_eps, initial_mu, learning_rate_eps, learning_rate_mu, n_iterations, h=0.05, compare=False, ground_truth=None):
    """Gradient descent algorithm for optimizing epsilon and mu."""
    eps, mu = initial_eps, initial_mu
    ari_list = []
    nmi_list = []
    if compare and ground_truth:
        ground_truth_labels = list(ground_truth.values())
    
    for i in range(n_iterations):
        grad_eps, grad_mu = approximate_gradient(G, f, eps, mu, h)
        eps = max(0, eps - learning_rate_eps * grad_eps)
        mu = max(0, mu - learning_rate_mu * grad_mu)
        print(f"Iteration {i+1}: eps = {eps}, mu = {mu}, f(eps, mu) = {f(G, eps, mu)}")
        
        if compare and ground_truth:
            clusters = scan(G, eps, mu)
            computed_labels = convert_to_labels(list(G.nodes()), clusters)
            ari, nmi = compare_clusters(ground_truth_labels, computed_labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            print(f"Adjusted Rand Index: {ari}")
            print(f"Normalized Mutual Information: {nmi}")
    return eps, mu, ari_list, nmi_list

def parse_ground_truth(filename):
    """Parse ground truth data from a file."""
    ground_truth = {}
    with open(filename, 'r') as file:
        for line in file:
            node, cluster_label = map(int, line.strip().split())
            ground_truth[node] = cluster_label
    return ground_truth

def convert_to_labels(node_list, cluster_data):
    """Convert cluster data to label format for evaluation."""
    labels = [-1] * len(node_list)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    for cluster_id, nodes in cluster_data.items():
        for node in nodes:
            if node in node_index:
                labels[node_index[node]] = cluster_id
    return labels

def compare_clusters(ground_truth_labels, computed_labels):
    """Compare ground truth clusters with computed clusters using ARI and NMI."""
    ari = adjusted_rand_score(ground_truth_labels, computed_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, computed_labels, average_method='arithmetic')
    return ari, nmi

def label_propagation_community_detection(G):
    """Label propagation community detection algorithm."""
    labels = {node: node for node in G.nodes()}
    nodes = list(G.nodes())
    
    while True:
        changes = False
        random.shuffle(nodes)
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
            if neighbor_labels:
                most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                if labels[node] != most_common_label:
                    labels[node] = most_common_label
                    changes = True
        if not changes:
            break
    
    communities = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    return communities

def determine_optimal_eps_mu(G, eps_range, mu_range):
    """Determine the optimal epsilon and mu values using loss function."""
    results = []
    for eps in eps_range:
        for mu in mu_range:
            mod = loss(G, eps, mu)
            results.append((eps, mu, mod))
    eps_values = [result[0] for result in results]
    mu_values = [result[1] for result in results]
    mods = [result[2] for result in results]
    return eps_values, mu_values, mods

def plot_metrics(ari_list, nmi_list):
    """Plot ARI and NMI metrics over iterations."""
    iterations = range(1, len(ari_list) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, ari_list, marker='o', label='ARI')
    plt.xlabel('Iteration')
    plt.ylabel('ARI')
    plt.title('ARI over Iterations')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(iterations, nmi_list, marker='o', label='NMI')
    plt.xlabel('Iteration')
    plt.ylabel('NMI')
    plt.title('NMI over Iterations')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("tc1-1_metrics.png")

def use_knee(G):
    """Use the knee method to determine optimal epsilon and mu."""
    eps_range = np.linspace(0.1, 1.0, 10)
    mu_range = range(2, 10)
    eps_values, mu_values, mods = determine_optimal_eps_mu(G, eps_range, mu_range)
    kneedle_eps = KneeLocator(eps_values, mods, curve='concave', direction='increasing')
    kneedle_mu = KneeLocator(mu_values, mods, curve='concave', direction='increasing')
    return kneedle_eps.knee, kneedle_mu.knee

def main():
    """Main function to run the SCAN algorithm and evaluate it."""
    G = nx.Graph()
    # filename = "data/TC1-1/1-1.dat"
    filename = "../data/tc11.dat"
    
    with open(filename) as file:
        edges = [tuple(map(int, line.split())) for line in file]

    G.add_edges_from(edges)
    initial_eps, initial_mu = use_knee(G)
    learning_rate_eps = 0.01
    learning_rate_mu = 0.1
    n_iterations = 20
    
    ground_truth = parse_ground_truth("data/TC1-1/1-1-c.dat")
    ground_truth = None
    
    clusters = scan(G, initial_eps, initial_mu)
    ground_truth_labels = list(ground_truth.values())
    computed_labels = convert_to_labels(list(G.nodes()), clusters)
    ari, nmi = compare_clusters(ground_truth_labels, computed_labels)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")

if __name__ == "__main__":
    main()
