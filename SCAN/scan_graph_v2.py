import networkx as nx
import math
from collections import deque, defaultdict
from networkx.algorithms import community
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import matplotlib.pyplot as plt

def similarity(G, v, u, cache):
    if (v, u) in cache or (u, v) in cache:
        return cache.get((v, u), cache.get((u, v)))
    
    v_set = set(G.neighbors(v)) | {v}
    u_set = set(G.neighbors(u)) | {u}
    inter = len(v_set & u_set)
    if not inter:
        return 0
    
    sim = (inter) / math.sqrt(len(v_set) * len(u_set))
    cache[(v, u)] = sim
    return sim

def neighborhood(G, v, eps, cache):
    return [u for u in G.neighbors(v) if similarity(G, v, u, cache) > eps]
    
def hasLabel(cliques,vertex):
    for k,v in cliques.items():
        if vertex in v:
            return True
    return False
def isNonMember(li,u):
    if u in li:
        return True
    return False

def process_nonmember(v, core_clusters, G):
    """Function to process each non-member node to find its closest core and update cluster."""
    nearest_core = min(core_clusters, key=lambda core: nx.shortest_path_length(G, source=v, target=core))
    return (v, core_clusters[nearest_core])

def sameClusters(G,clusters,u):
    n = list(G.neighbors(u))
    #belong 
    b = []
    i = 0
    
    while i < len(n):
        for k,v in clusters.items():
            if n[i] in v:
                if k in b:
                    continue
                else:
                    b.append(k)
        i = i + 1
    if len(b) > 1:
        return False
    return True
                

def closest_core(node, core_nodes, G):
    """Determine the closest core node's cluster for 'node'."""
    min_distance = float('inf')
    nearest_core = None
    for core in core_nodes:
        distance = nx.shortest_path_length(G, source=node, target=core, weight='weight')
        if distance < min_distance:
            min_distance = distance
            nearest_core = core
    return nearest_core
    
    
def scan(G, eps=0.7, mu=2):
    clusters = {}
    node_labels = {}  # Keeps track of which cluster each node belongs to
    sim_cache = {}
    c = 0
    cluster_nodes = 0
    nomem_nodes = 0
    

    for n in G.nodes:
        if n in node_labels:  # Skip nodes that have already been processed
            continue
        N = neighborhood(G, n, eps, sim_cache)
        if len(N) >= mu:
            c += 1
            clusters[c] = [n]
            node_labels[n] = c  # Mark this node as processed and belonging to cluster c
            cluster_nodes += 1
            Q = deque(N)

            while Q:
                w = Q.popleft()
                if w in node_labels:  # Skip if w is already processed
                    continue
                R = neighborhood(G, w, eps, sim_cache) + [w]
                for s in R:
                    if s not in node_labels:  # Only add s if it hasn't been processed yet
                        clusters[c].append(s)
                        node_labels[s] = c  # Mark s as processed
                        cluster_nodes += 1
                        Q.append(s)
            

        else:
            node_labels[n] = -1  # Mark non-core nodes distinctly if needed
            nomem_nodes += 1

    # Label propagation for non-members
    flag = True
    while flag:
        flag = False
        for v, label in node_labels.items():
            if label == -1:  # Process non-members specifically
                flag = True
                cluster_membership = defaultdict(int)
                for neighbor in G.neighbors(v):
                    if neighbor in node_labels and node_labels[neighbor] != -1:
                        cluster_membership[node_labels[neighbor]] += 1
                if cluster_membership:
                    selected_cluster = max(cluster_membership, key=cluster_membership.get)
                    clusters[selected_cluster].append(v)
                    node_labels[v] = selected_cluster  # Update the label to reflect the new cluster assignment
                    cluster_nodes += 1
                
    return clusters

def is_valid_partition(G, mods):
    node_set = set(G.nodes())
    partition_nodes = set()
    for community in mods:
        community_set = set(community)
        # Check for duplicate nodes in multiple communities
        if not partition_nodes.isdisjoint(community_set):
            return False, f"Duplicate nodes found in multiple communities."
        partition_nodes.update(community_set)
    
    # Check if all nodes are covered
    if partition_nodes != node_set:
        return False, f"Partition does not cover all nodes exactly once.{len(node_set - partition_nodes)} {len(partition_nodes)}"
    
    return True, "Partition is valid."

memo = {}  # Assuming memoization is required
def loss(G, eps, mu):
    # Check if the result is already computed and stored in memoization dictionary
    if (eps, mu) not in memo:
        clusters = scan(G, eps, mu)  # Get the clusters from the scan function
        
        
        # Prepare a list of communities for modularity calculation
        mods = [list(map(int, cluster)) for cluster in clusters.values()]
        # print([len(cluster) for cluster in clusters.values()])
        
        # print(is_valid_partition(G, mods))
        
        # Calculate modularity using networkx's community modularity function
        modularity = community.modularity(G, mods)
        
        # Store the loss in memo, modifying the formula if needed
        # Assuming you want to minimize a negative cube of the modularity
        memo[(eps, mu)] = -3 * modularity * modularity * modularity
    
    return memo[(eps, mu)]

# def approximate_gradient(G, f, eps, mu, h=1e-5):
#     """Approximate the gradient of 'f' at point (x, y)."""
#     dfdx = (f(G, eps + h, mu) - f(G, eps - h, mu)) / (2 * h)
#     dfdy = (f(G, eps, mu + h) - f(G, eps, mu - h)) / (2 * h)
#     return (dfdx, dfdy)

def approximate_gradient(G, f, eps, mu, h=1e-5):
    """Approximate the gradient of 'f' at point (x, y) using parallel computation."""
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
    
    return (dfdx, dfdy)

def gradient_descent_2d(G, f, initial_eps, initial_mu, learning_rate_eps, learning_rate_mu, n_iterations, h=0.05, compare=False, ground_truth=None):
    """Gradient descent algorithm for a function f(x, y) using numerical derivatives."""
    eps, mu = initial_eps, initial_mu
    ari_list = []
    nmi_list = []
    clusters = scan(G, eps, mu)
    ground_truth_labels = list(ground_truth.values())
    computed_labels = convert_to_labels(list(G.nodes()), clusters)
    ari, nmi = compare_clusters(ground_truth_labels, computed_labels)
    print(f"Adjusted Rand Index: {ari}")
    print(f"Normalized Mutual Information: {nmi}")
    for i in range(n_iterations):
        # start = time.time()
        grad_eps, grad_mu = approximate_gradient(G, f, eps, mu, h)
        eps = max(0, eps - learning_rate_eps * grad_eps)
        mu = max(0, mu - learning_rate_mu * grad_mu)
        print(f"Iteration {i+1}: eps = {eps}, mu = {mu}, f(eps, mu) = {f(G, eps, mu)}")
        if compare:
            clusters = scan(G, eps, mu)
            ground_truth_labels = list(ground_truth.values())
            computed_labels = convert_to_labels(list(G.nodes()), clusters)
            ari, nmi = compare_clusters(ground_truth_labels, computed_labels)
            ari_list.append(ari)
            nmi_list.append(nmi)
            print(f"Adjusted Rand Index: {ari}")
            print(f"Normalized Mutual Information: {nmi}")
        # print(f"Time taken: {time.time() - start} seconds")
    return eps, mu, ari_list, nmi_list
                        
def parse_ground_truth(filename):
    ground_truth = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            node = int(parts[0])
            cluster_label = int(parts[1])
            ground_truth[node] = cluster_label
    return ground_truth

def convert_to_labels(node_list, cluster_data):
    labels = [-1] * len(node_list)  # Default label for nodes not found in clusters
    node_index = {node: idx for idx, node in enumerate(node_list)}
    for cluster_id, nodes in cluster_data.items():
        for node in nodes:
            if node in node_index:  # Check if node is in the graph
                labels[node_index[node]] = cluster_id
    return labels
def compare_clusters(ground_truth_labels, computed_labels):
    ari = adjusted_rand_score(ground_truth_labels, computed_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, computed_labels, average_method='arithmetic')
    return ari, nmi

def plot_metrics(ari_list, nmi_list):
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
    plt.savefig("tc1-1_metrics_jongseop_babo.png")

def main():
    G=nx.Graph()
    filename = "../data/TC1-1/1-1.dat"
    
    edges = []

    with open(filename) as file:
        for line in file:
            line = line.split()
            line = (int(line[0]), int(line[1]))
            
            edges.append(line)

    vertice_ids  = range(len(edges))
    
    G.add_edges_from(edges)
    initial_eps = 0.7
    initial_mu = 2
    learning_rate_eps = 0.01  # Learning rate
    learning_rate_mu = 0.01  # Learning rate
    n_iterations = 100  # Number of iterations
    

    ground_truth = parse_ground_truth("../data/TC1-1/1-1-c.dat")
    eps, mu, ari_list, nmi_list = gradient_descent_2d(G, loss, initial_eps, initial_mu, learning_rate_eps, learning_rate_mu, n_iterations, compare=True, ground_truth=ground_truth)
    print("done")
    plot_metrics(ari_list, nmi_list)
    clusters = scan(G, eps, mu)
    # clusters = scan(G, initial_eps, initial_mu)
    # print('clusters: ')
    # for k,v in clusters.items():
    #     print(k,v)
    
main()