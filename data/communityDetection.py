import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse


from sklearn.metrics.cluster import normalized_mutual_info_score

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Reading the Network Data
def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G




# Applying the Louvain Algorithm
def detect_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    # Convert partition dictionary to list of lists for NMI calculation
    community_to_nodes = {}
    for node, community in partition.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())


# Step 5: Save the result
def save_communities_to_file(communities, file_path):
    # Convert the list of lists into a dictionary with community as key and nodes as values
    community_dict = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_dict[node] = community_id

    # Sort the dictionary by community key (which are the node numbers here)
    sorted_community_items = sorted(community_dict.items())

    # Write to file, now ensuring nodes are listed in the sorted order of their community keys
    with open(file_path, 'w') as f:
        for node, community_id in sorted_community_items:
            f.write(f"{node} {community_id}\n")


# Load the data


parser = argparse.ArgumentParser(description='Detect communities in a network.')
parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
args = parser.parse_args()

community_file_path = args.networkFile.replace('.dat', '.cmty')

G = load_network(args.networkFile)

# Detect communities using Louvain method
detected_communities = detect_communities_louvain(G)

save_communities_to_file(detected_communities, community_file_path)
