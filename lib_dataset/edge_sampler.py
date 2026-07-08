from tqdm import tqdm
import torch
from collections import defaultdict
import numpy as np
import math

def _edge_set(hyperedges):
    return {frozenset(edge) for edge in hyperedges}


def get_union(hyperedges):
    nodes = []
    for edge in hyperedges:
        nodes += list(edge)
    return set(nodes)


def _sample_filtered_negatives(sampler, positive_hyperedges, pred_num, forbidden_hyperedges):
    positive_hyperedges = _edge_set(positive_hyperedges)
    forbidden_hyperedges = _edge_set(forbidden_hyperedges)
    samples = []
    seen = set()

    attempts = 0
    max_attempts = 20
    while len(samples) < pred_num and attempts < max_attempts:
        attempts += 1
        for edge in sampler(set(positive_hyperedges)):
            edge = frozenset(edge)
            if edge in forbidden_hyperedges or edge in seen:
                continue
            samples.append(edge)
            seen.add(edge)
            if len(samples) >= pred_num:
                break

    if len(samples) < pred_num:
        nodes = sorted(get_union(positive_hyperedges))
        size_dist = list(generate_hyperedge_size_dist(positive_hyperedges).items())
        vals = [v for v, _ in size_dist]
        probs = [p for _, p in size_dist]
        while len(samples) < pred_num and nodes:
            sampled_size = int(np.random.choice(vals, p=probs))
            edge = _random_non_positive_edge(nodes, sampled_size, forbidden_hyperedges.union(seen))
            if edge in forbidden_hyperedges or edge in seen:
                break
            samples.append(edge)
            seen.add(edge)

    return [list(edge) for edge in samples]


def neg_generator(HE, pred_num, forbidden_HE=None):
    forbidden_HE = HE if forbidden_HE is None else forbidden_HE
    mns = MNSSampler(pred_num)
    sns = SNSSampler(pred_num)
    cns = CNSSampler(pred_num)

    t_mns = _sample_filtered_negatives(mns, HE, pred_num, forbidden_HE)
    t_sns = _sample_filtered_negatives(sns, HE, pred_num, forbidden_HE)
    t_cns = _sample_filtered_negatives(cns, HE, pred_num, forbidden_HE)

    return t_mns, t_sns, t_cns

def negative_sample(
        nodes_to_neighbors, size_dist, num_negative, hyperedges, method, corrupt_num = 1, half=False, rw_path = None):
    nodes = list(nodes_to_neighbors.keys())
    neg_samples = []
    if method == 'UNS':
        size_dist = get_pure_sample_size_dist(len(nodes))
    
    size_dist = list(size_dist.items())
    if not size_dist:
        size_dist = list(get_pure_sample_size_dist(len(nodes)).items())
    if method in ['SNS', 'UNS']:
        for i in tqdm(range(num_negative), leave=False):
            sampled_edge = sized_random_sampling(
                size_dist, nodes, nodes_to_neighbors, hyperedges)
            neg_samples.append(sampled_edge)
    elif method == 'MNS':
        for i in tqdm(range(num_negative), leave=False):
            sampled_node = sized_mf_sampling(
                size_dist, nodes, nodes_to_neighbors, hyperedges)
            neg_samples.append(sampled_node)     
    elif method == 'CNS':
        list_hyperedges = list(hyperedges)
        node_set = set(nodes_to_neighbors.keys())
        for i in tqdm(range(num_negative), leave=False):
            sampled_edge = clique_negative_sampling(
                hyperedges, nodes_to_neighbors, num_negative, list_hyperedges,
                node_set)
            neg_samples.append(sampled_edge)       

    return neg_samples

def generate_negative_samples_for_hyperedges(
        hyperedges, method, neg_samples_size, corrupt_num = 1, half=False, rw_path = None):
    #print(hyperedges)
    edges = {
        frozenset({u, v}) for hedge in hyperedges
        for u in hedge for v in hedge if u > v}
    nodes_to_neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        nodes_to_neighbors[u].add(v)
        nodes_to_neighbors[v].add(u)

    size_dist = generate_hyperedge_size_dist(hyperedges)
    
    #print('Generating Negative Samples')
    total = math.ceil(neg_samples_size)
    neg_samples = negative_sample(
        nodes_to_neighbors, size_dist, total, hyperedges, method, corrupt_num = 1, half=False, rw_path = rw_path)
    negative_hyperedges = [frozenset(x) for x, y in neg_samples]
    
    return negative_hyperedges

class SNSSampler(object):
    def __init__(self, pred_num):
        self.pred_num = pred_num
    def __call__(self, hedges):
        neg_samples_size = int(self.pred_num)
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'SNS', neg_samples_size)
        return neg_samples

class MNSSampler(object):
    def __init__(self, pred_num):
        self.pred_num = pred_num
    def __call__(self, hedges):
        neg_samples_size = int(self.pred_num)
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'MNS', neg_samples_size)
        return neg_samples
    
class CNSSampler(object):
    def __init__(self, pred_num):
        self.pred_num = pred_num
    def __call__(self, hedges):
        neg_samples_size = int(self.pred_num)
        neg_samples = generate_negative_samples_for_hyperedges(hedges, 'CNS', neg_samples_size)
        return neg_samples

'''-------------SNS Sampling Utils------------------'''

def get_pure_sample_size_dist(num_nodes):
    N = num_nodes
    nck = 1
    size_dist = dict()
    size_dist[0] = 1
    for idx in range(1, num_nodes):
        nck *= (N - (idx - 1)) / idx
        size_dist[idx] = nck
    size_dist[N] = 1
    total = sum(v for k, v in size_dist.items())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist

def sized_random_sampling(size_dist, nodes, nodes_to_neighbors, hyperedges):
    vals = [v for v, p in size_dist]
    p = [p for v, p in size_dist]
    sampled_nodes = frozenset({'a', 'b'})
    hyperedges.add(sampled_nodes)
    sampled_size = np.random.choice(vals, p=p)
    attempts = 0
    while frozenset(sampled_nodes) in hyperedges and attempts < 1000:
        sampled_nodes = [
            nodes[idx] for idx in np.random.choice(
                len(nodes), size=sampled_size, replace=False)]
        attempts += 1
    if frozenset(sampled_nodes) in hyperedges:
        sampled_nodes = list(_random_non_positive_edge(nodes, sampled_size, hyperedges))
    edges = {
        frozenset([node, node2])
        for node in sampled_nodes for node2 in sampled_nodes
        if node2 in nodes_to_neighbors[node] and node2 < node}
    hyperedges.remove(frozenset({'a', 'b'}))
    return sampled_nodes, edges

def generate_hyperedge_size_dist(hyperedges):
    size_dist = defaultdict(int)
    for edge in hyperedges:
        size_dist[len(edge)] += 1
    fallback_dist = {k: v for k, v in size_dist.items() if k > 1}
    if 1 in size_dist:
        del size_dist[1]
    if 2 in size_dist:
        del size_dist[2]
    total = sum(v for k, v in size_dist.items())
    if total == 0:
        size_dist = fallback_dist if fallback_dist else {2: 1}
        total = sum(size_dist.values())
    for i in size_dist:
        size_dist[i] = float(size_dist[i]) / total
    return size_dist

def _random_non_positive_edge(nodes, size, hyperedges, max_attempts=1000):
    if not nodes:
        return frozenset()
    size = max(1, min(int(size), len(nodes)))
    for _ in range(max_attempts):
        sampled_nodes = frozenset(np.random.choice(nodes, size=size, replace=False).tolist())
        if sampled_nodes not in hyperedges:
            return sampled_nodes
    return frozenset(np.random.choice(nodes, size=size, replace=False).tolist())

def _corrupt_hyperedge(edge, nodes, hyperedges, max_attempts=1000):
    edge_nodes = list(edge)
    if not edge_nodes:
        return _random_non_positive_edge(nodes, 1, hyperedges, max_attempts)
    for _ in range(max_attempts):
        corrupted = list(edge_nodes)
        replace_idx = np.random.choice(len(corrupted))
        candidates = [node for node in nodes if node not in corrupted or node == corrupted[replace_idx]]
        if not candidates:
            break
        corrupted[replace_idx] = np.random.choice(candidates)
        sampled_nodes = frozenset(corrupted)
        if sampled_nodes not in hyperedges:
            return sampled_nodes
    return _random_non_positive_edge(nodes, len(edge_nodes), hyperedges, max_attempts)

'''-------------MNS Sampling Utils------------------'''

def sample_initial_edge(nodes_to_neighbors):
    edgeidx = np.random.choice(
        sum(len(nodes_to_neighbors[n]) for n in nodes_to_neighbors))
    carry = 0
    for n in nodes_to_neighbors:
        if edgeidx < carry + len(nodes_to_neighbors[n]):
            edge = [n, list(nodes_to_neighbors[n])[edgeidx - carry]]
            break
        carry += len(nodes_to_neighbors[n])
    return edge

def mfinder_sampling(nodes_to_neighbors, k, nodes=None, max_restarts=100):
    nodes = nodes or list(nodes_to_neighbors.keys())
    neighbor_edges = []
    induced_edges = set()
    sampled_nodes = set()
    restarts = 0
    
    while len(sampled_nodes) < k:
        while len(neighbor_edges) == 0:
            restarts += 1
            if restarts > max_restarts:
                remaining = [node for node in nodes if node not in sampled_nodes]
                need = min(max(k - len(sampled_nodes), 0), len(remaining))
                if need > 0:
                    sampled_nodes.update(np.random.choice(remaining, size=need, replace=False).tolist())
                return sampled_nodes, induced_edges
            edge = sample_initial_edge(nodes_to_neighbors)
            sampled_nodes = set(edge)
            neighbor_edges = set([
                frozenset([node, nnode])
                for node in sampled_nodes for nnode in nodes_to_neighbors[node]
                if nnode not in sampled_nodes])
            neighbor_edges = list(neighbor_edges)

            induced_edges = set()
            induced_edges.add(frozenset(edge))
        
        selected_edge = neighbor_edges[np.random.choice(len(neighbor_edges))]
        induced_edges.add(selected_edge)
        new_node = [n for n in selected_edge.difference(sampled_nodes)][0]
        sampled_nodes.add(new_node)
        
        neighbor_edges = [
            edge for edge in neighbor_edges if new_node not in edge]
        
        new_edges = set()
        
        for node in nodes_to_neighbors[new_node]:
            if node not in sampled_nodes:
                new_edges.add(frozenset([new_node, node]))
            else:
                induced_edges.add(frozenset([new_node, node]))
        
        neighbor_edges.extend(list(new_edges))    
        #assert len(neighbor_edges) == len(set(neighbor_edges))

    return sampled_nodes, induced_edges

def sized_mf_sampling(size_dist, nodes, nodes_to_neighbors, hyperedges):
    vals = [v for v, p in size_dist]
    p = [p for v, p in size_dist]
    sampled_size = np.random.choice(vals, p=p)
    sentinel = frozenset({'a', 'b'})
    hyperedges.add(sentinel)
    for _ in range(1000):
        sampled_nodes, sampled_edge = mfinder_sampling(nodes_to_neighbors, sampled_size, nodes=nodes)
        sampled_nodes = frozenset(sampled_nodes)
        if sampled_nodes not in hyperedges:
            hyperedges.remove(sentinel)
            return sampled_nodes, sampled_edge
    sampled_nodes = _random_non_positive_edge(nodes, sampled_size, hyperedges)
    hyperedges.remove(sentinel)
    sampled_edge = {
        frozenset([node, node2])
        for node in sampled_nodes for node2 in sampled_nodes
        if node2 in nodes_to_neighbors[node] and node2 < node}
    return sampled_nodes, sampled_edge

'''-------------CNS Sampling Utils------------------'''

def clique_negative_sampling(
        hyperedges, nodes_to_neighbors, num_negative,
        list_hyperedges, node_set):
    nodes = list(node_set)
    neg = None
    for _ in range(1000):
        edgeidx = np.random.choice(len(hyperedges), size=1)[0]
        edge = list(list_hyperedges[edgeidx])
        node_to_remove = np.random.choice(len(edge), size=1)[0]
        nodes_to_keep = edge[:node_to_remove] + edge[node_to_remove+1:]
        probable_neighbors = node_set
        for node in nodes_to_keep:
            probable_neighbors = probable_neighbors.intersection(
                nodes_to_neighbors[node])
        
        probable_neighbors = probable_neighbors - set(nodes_to_keep)
        if len(probable_neighbors) == 0:
            continue
        probable_neighbors = list(probable_neighbors)
        neighbor_node = np.random.choice(probable_neighbors, size=1)[0]
        
        nodes_to_keep.append(neighbor_node)
        neg = frozenset(nodes_to_keep)
        if neg not in hyperedges:
            break
    else:
        edgeidx = np.random.choice(len(hyperedges), size=1)[0]
        neg = _corrupt_hyperedge(list_hyperedges[edgeidx], nodes, hyperedges)

    edges = {
        frozenset([node1, node2])
        for node1 in neg for node2 in neg if node1 < node2
    }
    return neg, edges
