
# MACROS
MIN_DEGREE = 0
ITERATIONS = 1

import pandas as pd
import numpy as np
from pathlib import Path
import os
from os.path import isfile, join
import networkx as nx
from statistics import mean, median
import inspect
import queue as Q
from time import process_time
from typing import List, Tuple, Set
from random import shuffle, random, choices, randrange
import time
from math import ceil
from tqdm.notebook import tqdm


class Graph(nx.Graph):
    def is_multilabel_graph(self):
        return False
    
    def without_secondary_edges(self):
        edited_graph = Graph()
        edited_graph.add_nodes_from(self.nodes())
        primary_colors = dict([(i, primary_color(self, i)) for i in self.nodes()])
        for u, v in self.edges():
            if primary_colors[u] == primary_colors[v] and primary_colors[u] in self.colors_of(u, v):
                edited_graph.add_edge(u, v, color=primary_colors[u])
        return edited_graph
        
    def primary_edge_graph(self):
        if not hasattr(self, 'primary_graph'):
            self.primary_graph = self.without_secondary_edges()
        return self.primary_graph

    def colors_of(self, a: int, b: int) -> List[int]:
        return [self.color_of(a, b)]
    
    def color_of(self, a: int, b: int) -> int:
        return self.edges[(a,b)]['color']

    def colors(self) -> Set[int]:
        if not hasattr(self, 'color_set'):
            colors = set()
            for edge in self.edges:
                colors.add(self.edges[edge]['color'])
            self.color_set = colors
        return self.color_set

    def node_pairs(self) -> List[Tuple[int, int]]:
        return [(a, b) for a in self.nodes for b in self.nodes if a < b]
    
    def is_valid_clustering(self, clustering: List[Tuple[List[int], int]]) -> bool:
        cluster_nodes = set()
        for cluster, col in clustering:
            if not isinstance(col, int):
                print(f"Cluster col {col} is not an integer.")
                print_cluster_head(cluster, 20)
                return False
            for node in cluster:
                if node in cluster_nodes:
                    print(f"{node} is already clustered in some other cluster.")
                    print_cluster_head(cluster, 20)
                    return False
                cluster_nodes.add(node)
        if cluster_nodes.symmetric_difference(set(self.nodes())):
            print(f"Set of clustered nodes is not equal to set of nodes.")
            print(f"{len(cluster_nodes)} were clustered, {len(self.nodes())} nodes in this graph.")
            print(f"{len(cluster_nodes.symmetric_difference(set(self.nodes())))} nodes in difference.")
            return False
        return True

    def error_of(self, clustering: List[Tuple[List[int], int]]) -> int:
        if not self.is_valid_clustering(clustering):
            raise Exception('Clustering is not valid')
        remaining_edges = self.number_of_edges()
        color_errors = 0
        non_edge_errors = 0
        for cluster, color in clustering:
            # print("color: ",color)
            for v in cluster:
                # print("---v: ",v)
                for u in cluster:
                    # print("u: ",u)
                    if u <= v: continue
                    if not self.has_edge(u, v):
                        non_edge_errors += 1
                        # print("inside violation on ",u,v)
                    else:
                        remaining_edges -= 1
                        if self.color_of(u,v) != color:
                            color_errors += 1
                            # print("color violation on ",u,v)
        return remaining_edges + non_edge_errors + color_errors

def most_frequent_color(graph: Graph, vertices: List[int]):
    count = {0: 0}
    for a in vertices:
        for b in vertices:
            if not a < b or not graph.has_edge(a, b): continue
            for color in graph.colors_of(a,b):
                count[color] = count.get(color, 0) + 1
    return max(count, key=count.get)

def cluster_ids(clustering: List[Tuple[List[int], int]]):
    cluster_id = {}
    for i, cluster in enumerate(clustering):
        for node in cluster[0]:
            cluster_id[node] = i
    return cluster_id

def intersect_clusterings(clustering_1: List[Tuple[List[int], int]], clustering_2: List[Tuple[List[int], int]]):
    cluster_ids_1 = cluster_ids(clustering_1)
    cluster_ids_2 = cluster_ids(clustering_2)
    clustering = {}
    for node, id1 in cluster_ids_1.items():
        id2 = cluster_ids_2[node]
        if not (id1, id2) in clustering:
            clustering[(id1, id2)] = ([], clustering_1[id1][1])
        clustering[(id1, id2)][0].append(node)
    return list(clustering.values())

def shuffled(collection):
    as_list = [i for i in collection]
    shuffle(as_list)
    return as_list

def vote(graph: Graph):
    graph = graph.primary_edge_graph()
    cluster_of = {}
    clustering = []
    cluster_colors = []
    for node in shuffled(graph.nodes()):
    # for node in graph.nodes():
        connectivity = {}
        # print("for1 node: ",node)
        # print("for1 adding naghbors")
        for u in graph.neighbors(node):
            # print("neighbors: ",u)
            if u not in cluster_of: continue
            # print("node: " + str(u) + " not clustered")
            cluster = cluster_of[u]
            connectivity[cluster] = connectivity.get(cluster, 0) + 1
        # print("end of adding naghbors")
        # print(" ".join(str(connectivity)))
        best_cluster = max(connectivity, key = lambda id: 2*connectivity[id] - len(clustering[id]) + random()*0.42) if connectivity else None
        # print("best_cluster for" + str(u) + " is " + str(best_cluster))
        if best_cluster is not None and 2*connectivity[best_cluster] - len(clustering[best_cluster]) > 0:
            clustering[best_cluster].append(node)
            cluster_of[node] = best_cluster
        else:
            clustering.append([node])
            cluster_colors.append(primary_color(graph, node))
            cluster_of[node] = len(clustering) - 1
    return list(zip(clustering, cluster_colors))

def vote_no_rand(graph: Graph):
    graph = graph.primary_edge_graph()
    cluster_of = {}
    clustering = []
    cluster_colors = []
    for node in shuffled(graph.nodes()):
    # for node in graph.nodes():
        connectivity = {}
        for u in graph.neighbors(node):
            if u not in cluster_of: continue
            cluster = cluster_of[u]
            connectivity[cluster] = connectivity.get(cluster, 0) + 1
        best_cluster = max(connectivity, key = lambda id: 2*connectivity[id] - len(clustering[id])) if connectivity else None
        if best_cluster is not None and 2*connectivity[best_cluster] - len(clustering[best_cluster]) > 0:
            clustering[best_cluster].append(node)
            cluster_of[node] = best_cluster
        else:
            clustering.append([node])
            cluster_colors.append(primary_color(graph, node))
            cluster_of[node] = len(clustering) - 1
    return list(zip(clustering, cluster_colors))

def vote_choose_best(graph: Graph):
    graph = graph.primary_edge_graph()
    H = Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True)))
    H.add_edges_from(graph.edges(data=True))
    cluster_of = {}
    clustering = []
    cluster_colors = []  
    # adding node 0 to cluster 0
    choosed_cluster = 0
    choosed_node = 0
    clustering.append([choosed_node])
    cluster_of[choosed_node] = choosed_cluster

    for _ in H.nodes(): 
        choosed_cluster = None
        choosed_node_error = -1
        for v2 in H.nodes():
            if v2 not in cluster_of: 
                connectivity = {}
                for nbr_v2 in H.neighbors(v2):
                    if nbr_v2 not in cluster_of: continue
                    cluster = cluster_of[nbr_v2]
                    connectivity[cluster] = connectivity.get(cluster, 0) + 1
                best_cluster = max(connectivity, key = lambda id: 2*connectivity[id] - len(clustering[id])) if connectivity else None
                
                if best_cluster is not None and 2*connectivity[best_cluster] - len(clustering[best_cluster]) > 0:
                    temp_error = 2*connectivity[best_cluster] - len(clustering[best_cluster])
                else:
                    temp_error = -1     

                if (temp_error > choosed_node_error):
                    choosed_node_error = temp_error
                    choosed_node = v2
                    choosed_cluster = best_cluster
        if choosed_cluster is not None:
            clustering[choosed_cluster].append(choosed_node)
            cluster_of[choosed_node] = choosed_cluster
            # print("node " +str(choosed_node) + " added to cluster " + str(choosed_cluster))
        else:
            for v3 in H.nodes():
                if v3 not in cluster_of: 
                    choosed_node = v3
                    break
            clustering.append([choosed_node])
            cluster_colors.append(primary_color(H, choosed_node))
            cluster_of[choosed_node] = len(clustering) - 1
            # print("node " +str(choosed_node) + " added to own cluster " + str(cluster_of[choosed_node]))
    return list(zip(clustering, cluster_colors))

def pivot(graph: Graph): # 3-Approximation
    clustering = []
    is_clustered = dict((i, False) for i in graph.nodes())
    for center in shuffled(graph.nodes()):
        if is_clustered[center]: continue
        is_clustered[center] = True
        cluster = [center]
        for a in graph.neighbors(center):
            if is_clustered[a]: continue
            is_clustered[a] = True
            cluster.append(a)
        cluster_color = most_frequent_color(graph, cluster)
        clustering.append((cluster, cluster_color))
    return clustering

def reduce_and_cluster(graph:Graph): # 5-Approximation
    graph = graph.primary_edge_graph()
    return pivot(graph)

def read_dataset(name: str):
    if name in ['facebook', 'twitter', 'microsoft_academic','facebook_multilabel', 'twitter_multilabel', 'microsoft_academic_multilabel']:
        return map(remove_self_loops, read_small_dataset(name))
    data_dict = {
        'dawn': (lambda : read_hyperedge('DAWN_majority.csv')),
        'cooking': (lambda : read_hyperedge('Cooking_majority.csv')),
        'legacy_dblp': (lambda : read_legacy('DBLP_ALL.csv')),
        'legacy_string': (lambda : read_legacy('STRING_ALL.csv')),
        '6_rows': (lambda : read_legacy('new_6_rows.csv')),
        '7_rows': (lambda : read_legacy('new_7_rows.csv')),
        '8_rows': (lambda : read_legacy('new_8_rows.csv')),
        '9_rows': (lambda : read_legacy('new_9_rows.csv')),
        '10_rows': (lambda : read_legacy('new_10_rows.csv')),
        '100_rows': (lambda : read_legacy('new_100_rows.csv')),
        '200_rows': (lambda : read_legacy('new_200_rows.csv')),
        '300_rows': (lambda : read_legacy('new_300_rows.csv')),
        '1000_rows': (lambda : read_legacy('new_1000_rows.csv')),
    }
    if name not in data_dict:
        raise Exception('Unknown Dataset')
    return map(remove_self_loops, data_dict[name]())

def remove_self_loops(graph):
    for node in graph.nodes():
        if graph.has_edge(node, node):
            graph.remove_edge(node, node)
    return graph

def read_social_circles(path, multilabel = False):
    graph = None
    if multilabel:
        graph = nx.read_edgelist(join(path, 'combined.txt'), nodetype = int, create_using = MultiGraph, comments = '#')
    else:
        graph = nx.read_edgelist(join(path, 'combined.txt'), nodetype=int, create_using = Graph,comments ='#')
    circle_path = join(path, 'circles')
    circle_files = [f for f in os.listdir(circle_path) if isfile(join(circle_path, f))]
    circles_of = {}
    circle_id = 0
    for circle_file in circle_files:
        with open(join(circle_path, circle_file), 'r') as file:
            ego = os.path.basename(circle_file).split('.')[0]
            for line in file.readlines():
                circle = [int(a) for a in line.split()[1:]]
                circle.append(ego)
                for node in circle:
                    if not node in circles_of:
                        circles_of[node] = set()
                    circles_of[node].add(circle_id)
                circle_id += 1
    random_edges = 0
    semi_random_edges = 0
    useless_nodes = [node for node in graph.nodes() if node not in circles_of]
    graph.remove_nodes_from(useless_nodes)
    for a, b in graph.edges():
        shared_circles = circles_of[a].intersection(circles_of[b])
        edge_color = None
        if multilabel:
            if len(shared_circles) == 0:
                shared_circles = [random.randrange(circle_id)]
                random_edges += 1  
            graph.edges[(a,b)]['colors'] = list(shared_circles)
        else:
            if shared_circles:
                edge_color = random.choice(list(shared_circles))
                if len(shared_circles) > 1: semi_random_edges += 1
            else:
                edge_color = random.randrange(circle_id)   
                random_edges += 1      
            graph.edges[(a, b)]['color'] = edge_color
    return [graph]
                
def read_microsoft_academic(multilabel = False):
    data_path = './data/microsoft_academic/'
    labels = []
    with open(join(data_path,'hyperedge-labels.txt')) as file:
        for line in file.readlines():
            labels.append(int(line)-1)
    edge_candidates = {}
    with open(join(data_path,'hyperedges.txt')) as file:
        for index, line in enumerate(file.readlines()):
            label = labels[index]
            authors = [int(i) for i in line.split()]
            for a in range(len(authors)):
                for b in range(a+1, len(authors)):
                    adrian = authors[a]
                    bdrian = authors[b]
                    edge = (min(adrian, bdrian), max(adrian, bdrian))
                    if edge not in edge_candidates:
                        edge_candidates[edge] = {}
                    edge_candidates[edge][label] = edge_candidates[edge].get(label, 0) + 1
    graph = MultiGraph() if multilabel else Graph()
    for edge, potential_topics in edge_candidates.items():
        if multilabel:
            graph.add_edge(edge[0], edge[1], colors = list(potential_topics.keys()))
        else:
            most_frequent_topic = max(potential_topics, key = potential_topics.get)
            graph.add_edge(edge[0], edge[1], color = most_frequent_topic)
    return [graph]

def spliturl(url):
    splitted = url.split('/')
    if len(splitted) >= 3:
        return splitted[2]
    else:
        return None

def generate_dblp():
    data_path = './data/dblp/'
    publication_types = ['output_article', 'output_inproceedings']
    for publication_type in publication_types:
        print(publication_type)
        headername = publication_type + '_header.csv'
        df_header = pd.read_csv(join(data_path, headername), sep =';')
        column_names = ([i.split(':')[0] for i in df_header.columns])
        if 'url' not in column_names:
            print('does not have a url: ', column_names)
            continue
        filename = publication_type + '.csv'
        column_types = {name:str for name in column_names}
        df = pd.read_csv(join(data_path, filename), sep = ';', header = None, names = column_names, dtype=column_types)
        df = df[['author', 'url']]
        print(df.shape)
        df = df.dropna()
        df['url'] = df['url'].apply(spliturl)
        df = df.dropna()
        print(df.shape)
        df.to_csv(join(data_path, publication_type + '_dataset.csv'))
        print(df.head())

def read_dblp_slow(multilabel = False):
    data_path = './data/dblp_original/'
    publication_types = ['output_article', 'output_inproceedings']
    authors_by_name = {}
    journals_by_name = {}
    author_id_counter = 0
    journal_id_counter = 0
    edge_candidates = {}
    for publication_type in publication_types:
        print(publication_type)
        filename = join(data_path, publication_type + '_dataset.csv')
        with open(filename, 'r', encoding='utf-8') as file: 
            line = file.readline() #header
            line = file.readline()
            while line:
                authors = line.split(',')[1].split('|')
                journal = line.split(',')[2]
                if journal not in journals_by_name:
                    journals_by_name[journal] = journal_id_counter
                    journal_id_counter+= 1
                for author in authors:
                    if author not in authors_by_name:
                        authors_by_name[author] = author_id_counter
                        author_id_counter+= 1
                journal_id = journals_by_name[journal]
                for i in range(len(authors)):
                    for j in range(i+1, len(authors)):
                        idA = authors_by_name[authors[i]]
                        idB = authors_by_name[authors[j]]
                        if idA == idB:
                            continue
                        edge = (min(idA, idB), max(idA, idB))
                        if edge not in edge_candidates:
                            edge_candidates[edge] = {}
                        if not journal_id in edge_candidates[edge]:
                            edge_candidates[edge][journal_id] = 0
                        edge_candidates[edge][journal_id] += 1
                line = file.readline()        
    graph = MultiGraph() if multilabel else Graph()
    for edge, potential_topics in edge_candidates.items():
        if multilabel:
            graph.add_edge(edge[0], edge[1], colors = potential_topics.keys())
        else:
            most_frequent_topic = max(potential_topics, key = potential_topics.get)
            graph.add_edge(edge[0], edge[1], color = most_frequent_topic)
    return [graph]

def read_dblp(multilabel = False):
    graph = None
    if multilabel:
        filename = "./data/dblp/dblp_multilabel.edgelist"
        graph = MultiGraph()
        with open(filename) as file:
            for index, line in enumerate(file.readlines()):
                parts = line.split(' ', maxsplit = 2)
                a = int(parts[0])
                b = int(parts[1])
                color_list = parts[2].split('[', maxsplit = 1)[1][:-3]
                colors = set(int(i) for i in color_list.split(','))
                graph.add_edge(a,b, colors = colors)
    else:
        graph = nx.read_edgelist("dblp.edgelist", create_using = Graph,nodetype =int, data=(("color", int),))
    return [graph]

def read_string(multilabel = False):
    def read_graph(path):
        graph = Graph() if not multilabel else MultiGraph()
        with open(path) as file:
            line = file.readline()
            line = file.readline()
            while line:
                elems = [int(x) for x in line.strip().split(',')]
                u = elems[0]
                v = elems[1]
                graph.add_edge(u, v)
                if not multilabel:
                    graph[u][v]['color'] = elems[2]
                else:
                    graph[u][v]['colors'] = set(elems[2:])
                line = file.readline()
        return graph

    directory = './data/string_protein'
    return (read_graph(os.path.join(directory, filename)) for filename in os.listdir(directory))

def read_hyperedge(filename, multilabel = False):
    def read_graph(path):
        graph = Graph() if not multilabel else MultiGraph()
        with open(path) as file:
            line = file.readline()
            while line:
                elems = [int(x) for x in line.strip().split(',')]
                u = elems[0]
                v = elems[1]
                graph.add_edge(u, v)
                if not multilabel:
                    graph[u][v]['color'] = elems[2]
                else:
                    graph[u][v]['colors'] = set(elems[2:])
                line = file.readline()
        return graph

    path = Path(__file__).parent / filename

    return [read_graph(path)]

def read_legacy(filename):
    def read_graph(path):
        return nx.read_edgelist(path, comments='#', delimiter=' ', create_using=Graph,nodetype =int, data=[('color', int)])

    path = Path(__file__).parent / filename
    return [read_graph(path)]

def read_small_dataset(filename):
    def read_graph(path, multilabel = False):
        print("answer: ")
        print(path)
        if multilabel:
            graph = MultiGraph()
            with open(path) as file:
                for index, line in enumerate(file.readlines()):
                    parts = line.split(' ', maxsplit = 2)
                    a = int(parts[0])
                    b = int(parts[1])
                    color_list = parts[2].split('[', maxsplit = 1)[1].split(']', maxsplit=1)[0]
                    colors = set(int(i) for i in color_list.split(','))
                    graph.add_edge(a,b, colors = colors)
            return graph
        return nx.read_edgelist(path, comments='#', delimiter=' ', create_using=Graph,nodetype =int, data=[('color', int)])

    directory = '.\data\small_datasets'
    multilabel = filename.endswith('_multilabel')
    print("hiii ")
    temp = os.path
    print (temp)
    return [read_graph(os.path.join(directory, filename+'.edgelist'), multilabel=multilabel)]

def print_cluster_head(cluster,n):
    if len(cluster) <= n:
        print("Cluster is ", cluster)
        return
    print("Cluster is ",cluster[:n], f"... (size {len(cluster)})")

def primary_color(graph, node: int):
    count = {}
    for _, v in graph.edges(node):
        for color in graph.colors_of(node, v):
            count[color] = count.get(color, 0) + 1
    if not count:
        return 0
    return max(count, key=count.get)

def clean_source_code_line(line):
    cleaned = line.split(':', maxsplit = 1)[1].split('#', maxsplit=1)[0].strip()
    if cleaned[-1] == ',':
        return cleaned[:-1].strip()
    return cleaned

def approx_errors(runs, graph, cluster_generator):
    result = {
        'errors': [],
        'cluster_counts': [],
        'wall_clock_times': []
    }
    for _ in range(runs):
        start_time = process_time()
        clustering = cluster_generator(graph)
        end_time = process_time()
        result['errors'].append(graph.error_of(clustering))
        result['cluster_counts'].append(len(clustering))
        result['wall_clock_times'].append(end_time - start_time)
        print('#', end = '',flush=True)
    print(end='\r')
    return result

def remove_nodes_with_low_degree(graph: Graph, min_degree):
    if min_degree == 0:
        return
    deleted = set()
    pq = Q.PriorityQueue()
    for node in graph.nodes():
        pq.put((graph.degree[node],node))
    while not pq.empty():
        degree, node = pq.get()
        if node in deleted: continue
        if degree >= min_degree:
            return
        for neig in graph.neighbors(node):
            pq.put((graph.degree[neig]-1, neig))
        graph.remove_node(node)
        deleted.add(node)

def run_Heuristics(dataset_name, algorithms, algorithm_names):
    print(f"Reading dataset: {dataset_name}")
    dataset = read_dataset(dataset_name)
    if MIN_DEGREE > 0:
        dataset_name = dataset_name + f"_min_degree_{MIN_DEGREE}"
    summaries = []
    for i in range(len(algorithms)+1):
        summary = {}
        summary['runs'] = ITERATIONS
        summary['number_of_nodes'] = 0
        summary['number_of_edges'] = 0
        summary['number_of_colors'] = 0
        summary['dataset'] = dataset_name
        summary['errors'] = [0]*summary['runs']
        summary['wall_clock_times'] = [0]*summary['runs']
        summary['number_of_clusters'] = [0]*summary['runs']
        # summary['algorithm'] = algorithm_names[algorithms[i]].split('.', maxsplit=1)[1] if i < len(algorithms) else 'primary_edge_graph'
        summaries.append(summary)
    summaries[-1]['runs'] = 1 #10
    summaries[-1]['wall_clock_times'] = [0]*summaries[-1]['runs']
    summaries[-1]['dataset'] = dataset_name

    for graph_number, graph in enumerate(dataset):
        print(f'part {graph_number}')
        if MIN_DEGREE > 0:
            remove_nodes_with_low_degree(graph, MIN_DEGREE)

        # measure time of to create primary_edge_graph(), we generally cache it to speed up the experiments
        for i in range(summaries[-1]['runs']):
            if hasattr(graph, 'primary_graph'):
                delattr(graph, 'primary_graph')
            start_time = process_time()
            graph.primary_edge_graph()
            end_time = process_time()
            summaries[-1]['wall_clock_times'][i] += end_time - start_time
        
        results_by_algorithm = {}
        for j in range(len(algorithms)):
            summary = summaries[j]
            alg = algorithms[j]
            summary['number_of_nodes'] += graph.number_of_nodes()
            summary['number_of_edges'] += graph.number_of_edges()
            summary['number_of_colors'] = max(summary['number_of_colors'], len(graph.colors()))
            measurements = approx_errors(summary['runs'], graph, alg)
            for i in range(summary['runs']):
                summary['errors'][i] += measurements['errors'][i]
                summary['number_of_clusters'][i] += measurements['cluster_counts'][i]
                summary['wall_clock_times'][i] += measurements['wall_clock_times'][i]
            results_by_algorithm[algorithm_names[alg]] = mean(measurements['errors'])
            if(j==0):
                print("pivot:")
            elif(j==1):
                print("vote:")
            elif(j==2):
                print("vote_no_rnd:")
            elif(j==3):
                print("vote_choose_best:")
            
            
            print("runs = {0:4} dataset = {1:15} mean: {2:8} median: {3:8} min: {4:8} time: {5:8} seconds".format(summary['runs'],summary['dataset'], round(mean(measurements['errors'])), round(median(measurements['errors'])), round(min(measurements['errors'])), round(sum(measurements['wall_clock_times']), 2)))
        print()

    for i in range(len(algorithms)+1):
        summary = summaries[i]
        # log_real_world(summary)

print("start running experiments...")      

algorithms = [  
    lambda graph: pivot(graph),
    lambda graph: vote(graph),
    lambda graph: vote_no_rand(graph),
    lambda graph: vote_choose_best(graph),
]

algorithm_names = {alg:clean_source_code_line(inspect.getsourcelines(alg)[0][0]) for alg in algorithms}

# LSH Part################################

def create_hash_func(size: int):
    # function for creating the hash vector/function
    hash_ex = list(range(size))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size: int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes

def create_hash(minhash_fn, vector: list):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for func in minhash_fn:
        for i in range(len(vector)):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature

def fast_score(data, cluster):
    score = 0
    nodes = len(data)
    for i in (range(nodes)):
        for j in range(i + 1, nodes):
            # if i and j are similar and not in a same cluster
            if data[i][j] == 1:
                # and not in the same cluster
                if cluster[i] != cluster[j]: score -= 1
            # else if they are not similar
            else:
                # and in the same cluster
                if cluster[i] == cluster[j]: score -= 1

    return score

def find_best_cluster(bands_clusters,data):
  max = -float('inf')
  max_cluster= None
  for cluster in bands_clusters:
    # scr = score(data, cluster)
    scr = fast_score(data, cluster)
    if scr > max:
      max = scr
      max_cluster = cluster
    if scr == 0:
      break
  return max_cluster, max

def LSH_simmilarity_matrix_permutation_arr(num_iter, data, row_num, band_num):
    #initialize
    best_cluster = None
    best_score = -float('inf')
    nodes = len(data)

    # num of iterations
    for _ in tqdm(range(num_iter)):
        
        # a perm. of size n
        permutation = np.arange(nodes).astype(int)
        shuffle(permutation)

        # new matrix with shuffled nodes
        shuff_data = []
        for i in range(nodes):
            shuff_data.append([data[i][j] for j in permutation])
        
        

        # simmilarity matrix is the new signiture
        sigs = np.array(shuff_data)

        # for each band
        bands_clusters = []
        for i in (range(band_num)):
            band_cluster = [0] * nodes

            # for each signiture calculate lsh and corresponding cluster
            for j, sig in enumerate(sigs):
                
                lsh = int(''.join(map(str,sig[i*row_num: (i+1)*row_num])), 2) % nodes
                
                # band_cluster[lsh] = (band_cluster.get(lsh,list()))+ [j]
                band_cluster[j] = lsh
            
            # add cluster to clusters
            bands_clusters.append(band_cluster)
        # print('claculating score')
        cluster, scr = find_best_cluster(bands_clusters,data)
        # print('score:', scr)
        if scr > best_score:
            best_score = scr
            best_cluster = cluster
        # print('best cluster:', best_cluster)
    return best_score , best_cluster
    # return best_score

def regular_LSH(num_iter, data, row_num, band_num, sigs_num):
  #initialize
  best_cluster = None
  best_score = -float('inf')
  nodes = len(data)

  for _ in tqdm(range(num_iter)):
    # we create 20 minhash vectors
    minhash_funcs = build_minhash_func(len(data), sigs_num)

    # now create signatures
    sigs = []
    for i in range(len(data)):
      sigs.append(create_hash(minhash_funcs, data[i]))

    bands_clusters = []
    for i in range(band_num):
      band_cluster = [0] * nodes
      for j, sig in enumerate(sigs):
        try:
          lsh = int(''.join(map(str,sig[i*row_num: (i+1)*row_num])), 10) % nodes
        except:
          print(''.join(map(str,sig[i*row_num: (i+1)*row_num])))
        band_cluster[j] = lsh
      bands_clusters.append(band_cluster)
    cluster, scr = find_best_cluster(bands_clusters,data)
    if scr > best_score:
      best_score = scr
      best_cluster = cluster
  return best_score, best_cluster

def LSH_singular_diff(num_iter, data, row_num, band_num, treshold, is_log = True):
    
    nodes = len(data) # number of nodes in dataset
    best_cluster = [0] * nodes
    single_nodes = [] # single nodes that have less similarity than treshold
    for i, row in enumerate(data):
        if sum(row) < treshold: # number of 1's in a row
            best_cluster[i] = nodes + i    # new cluster not involve in other clusters because of node++ index
            single_nodes.append(i)
    # print(data, single_nodes)

    # delete single elements from data
    for i in single_nodes:
        changed_data = np.concatenate((data[0:i, :], data[i + 1:, :]), axis=0) # delete correspondig row
        changed_data = np.concatenate((changed_data[:,0:i], changed_data[:, i + 1:]), axis=1) # delete correspondig column
    else:
      changed_data = data
    print(single_nodes)
    print('bst:', best_cluster)
    cluster = []
    if len(changed_data) != 0:
      if is_log:
        node_log = np.floor(np.log2(len(changed_data))).astype(int)
        row_num = node_log
        band_num = len(changed_data) // row_num
      else:
        row_num = ceil(len(changed_data) / 10)
        band_num = len(changed_data) // row_num
        
      # print('data',data, type(data))
      # score, cluster = LSH_simmilarity_matrix_permutation_arr(num_iter, changed_data, row_num, band_num)
      regular_LSH(num_iter, changed_data, row_num, band_num, 30)
    print(cluster, best_cluster)
    


    #merge two clusters
    j = 0 # index into cluster of changed_data
    for i in range(nodes): # loop over all nodes
       # if a node is not single meaning is not in the best cluster update it
       if i not in single_nodes: 
          best_cluster[i] = cluster[j]
          j += 1
    # print(best_cluster)
    score = fast_score(data, best_cluster)

    return score, best_cluster

def make_adjacecy_matrix(graph):
    # turn nx graph to adjacency matrix
    adjacecy_matrix = np.zeros((graph.number_of_nodes()+1,graph.number_of_nodes()+1)).astype(int)
    np.fill_diagonal(adjacecy_matrix, 1)
    for u,v in graph.edges():
        adjacecy_matrix[u][v] = 1
        adjacecy_matrix[v][u] = 1
    return adjacecy_matrix

def run_LSH(dataset_name):
    print(f"Reading dataset: {dataset_name}")
    dataset = read_dataset(dataset_name)
    if MIN_DEGREE > 0:
        dataset_name = dataset_name + f"_min_degree_{MIN_DEGREE}"

    for graph_number, graph in enumerate(dataset):
        print(f'part {graph_number}')
        if MIN_DEGREE > 0:
            remove_nodes_with_low_degree(graph, MIN_DEGREE)
        graph.primary_edge_graph()
        adjacecy_matrix = make_adjacecy_matrix(graph)

        # running sim_perm_arr LSH################################################################
        print("running sim_perm_arr LSH")
        total_num_sigs=len(adjacecy_matrix)
        node_log = np.floor(np.log2(len(adjacecy_matrix))).astype(int)
        row_num = node_log
        band_num = total_num_sigs // row_num
        j = total_num_sigs / 20
        start_time = process_time()
        score, cluster = LSH_simmilarity_matrix_permutation_arr(node_log * 5, adjacecy_matrix, row_num, band_num)
        # score, cluster = LSH_simmilarity_matrix_permutation_arr(1, adjacecy_matrix, row_num, band_num)
        end_time = process_time()
        lsh_time = end_time - start_time
        output_log = 'time: ' + str(lsh_time) +' \n row: '+ str(row_num) +' band: '+ str(band_num) +' \n score: '+ str(score) + ' \n number of signitures: ' +str(j)+'\n\n'
        print(output_log)

        # running regular_LSH ################################################################
        choosed_iterations = [1]
        print("running regular_LSH")
        for num_iter in choosed_iterations:
            print(num_iter)
            total_num_sigs=30
            start_time = process_time()
            score, cluster = regular_LSH(num_iter, adjacecy_matrix, row_num, band_num, total_num_sigs)
            end_time = process_time()
            lsh_time = end_time - start_time
            output_log = 'time: ' + str(lsh_time) +' \n row: '+ str(row_num) +' band: '+ str(band_num) +' \n score: '+ str(score) + ' \n number of signitures: ' +str(total_num_sigs)+'\n\n'
            print(output_log)
        
        '''
        # running singular_diff LSH################################################################
        print("running singular_diff LSH")
        total_num_sigs=len(adjacecy_matrix)
        j = total_num_sigs // 8
        num_iter = 10
        node_log = np.floor(np.log2(len(adjacecy_matrix))).astype(int)
        row_num = node_log
        band_num = total_num_sigs // row_num

        score, cluster = LSH_singular_diff(num_iter, adjacecy_matrix, row_num, band_num, j, is_log = True)
        output_log = 'dataset: '+ dataset_name +' \n row: '+ str(row_num) +' \n band: '+ str(band_num) +' \n score: '+ str(score) + ' \n number of signitures: ' +str(j)+'\n\n'
        print(output_log)
        '''
        
# run_LSH('cooking')
# run_Heuristics('6_rows', algorithms, algorithm_names)
# run_Heuristics('7_rows', algorithms, algorithm_names)
# run_Heuristics('8_rows', algorithms, algorithm_names)
# run_Heuristics('9_rows', algorithms, algorithm_names)
# run_Heuristics('10_rows', algorithms, algorithm_names)
# run_Heuristics('100_rows', algorithms, algorithm_names)
# run_Heuristics('200_rows', algorithms, algorithm_names)
# run_Heuristics('300_rows', algorithms, algorithm_names)
run_Heuristics('1000_rows', algorithms, algorithm_names)
# run_Heuristics('legacy_string', algorithms, algorithm_names)
# run_Heuristics('cooking', algorithms, algorithm_names)
# run_Heuristics('dawn', algorithms, algorithm_names)
# run_Heuristics('legacy_dblp', algorithms, algorithm_names)
