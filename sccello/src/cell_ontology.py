import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import ipdb
import networkx as nx

from sccello.src.utils import data_loading

ALPHA = 0.9
THRESHOLD = 1e-4

def get_cell_type_labelid2nodeid(data_branch, clid2nodeid):
    """
    Return: a dict mapping from cell type label ID in our pre-training datasets to its node ID on cell ontology graph 
    """
    # ensure mapping from CLID (cell lineage ID) to its node ID on cell ontology graph is correct
    _, clid2nodeid_tmp = data_loading.get_prestored_data("cell_taxonomy_graph_owl")
    assert clid2nodeid == clid2nodeid_tmp

    # mapping from CLID (cell lineage ID) to its cell type label ID in our pre-training datasets
    clid2labelid = data_loading.get_prestored_data(f"pretrain_newdata_{data_branch}_clid2labelid_pkl")
    
    # connect two mappings
    labelid2nodeid = {v: clid2nodeid[k] for k,v in clid2labelid.items()}

    return labelid2nodeid

def get_forbidden_cell_type_contrast(data_branch):
    """
    Return: a matrix sized by the number of cell types in our pre-training datasets,
        with a 1 indicating ancestry between nodes and a 0 indicating no ancestry.
    """

    # retrieve cell ontology graph (a directed acyclic graph)
    ct_graph_edge_list, clid2nodeid = data_loading.get_prestored_data("cell_taxonomy_graph_owl")
    ct_graph_edge_list = [[v, u] for u, v in ct_graph_edge_list] # [son, fa] -> [fa, son]
    ct_graph = nx.DiGraph(ct_graph_edge_list)

    # for each node, label one for its ancestor node or zero if otherwise
    num_node = ct_graph.number_of_nodes()
    ancestor_mask = np.zeros((num_node, num_node), dtype=np.int32)
    for idx in tqdm(range(num_node)):
        ancestor_list = list(nx.ancestors(ct_graph, idx))            
        ancestor_mask[idx][idx] = 1
        for v in ancestor_list:
            ancestor_mask[idx][v] = 1

    # get cell type label ID in our pre-training datasets to its node ID on cell ontology graph 
    cell_type_labelid2nodeid = get_cell_type_labelid2nodeid(data_branch, clid2nodeid)

    # for each cell type in our pre-training datasets, label one for its ancestor cell type or zero if otherwise
    num_labels = len(cell_type_labelid2nodeid)
    ret_mask = np.zeros((num_labels, num_labels), dtype=np.int32)
    for i in range(num_labels):
        for j in range(num_labels):
            newi = cell_type_labelid2nodeid[i]
            newj = cell_type_labelid2nodeid[j]
            if newi != -1 and newj != -1:
                ret_mask[i, j] = ancestor_mask[newi, newj]
    return ret_mask


def get_cell_taxonomy_similarity(nx_graph, get_raw=False, alpha=ALPHA, thresh=THRESHOLD):
    """
    Return: a matrix sized by the number of nodes on the cell ontology graph,
        with transformed PPR values (see paper App. A PPR Transformation for details)
    """
    
    # assume node indices ranging from 0
    assert np.max([node_id for node_id in nx_graph.__dict__["_node"].keys()]) + 1 == nx_graph.number_of_nodes()
    num_node = nx_graph.number_of_nodes()
    if get_raw:
        similarity = np.zeros((num_node, num_node), dtype=np.float32)
    else:
        similarity = np.zeros((num_node, num_node), dtype=np.int32)
    for node_id in tqdm(range(num_node), "getting cell taxonomy similarity"):
        personalization = {i: i == node_id for i in range(num_node)}
        ppr = nx.pagerank(nx_graph, alpha=alpha, personalization=personalization)
        for k, v in ppr.items():
            if not get_raw:
                similarity[node_id][k] = 1 if v < thresh else np.log2(v * (1 / thresh) + 1)
            else:
                similarity[node_id][k] = v
    return similarity

def get_hierarchy_cell_type_contrast(data_branch, get_raw=False):
    """
    Return: (1) a matrix sized by the number of cell types in our pre-training datasets,
        with transformed PPR values (see paper App. A PPR Transformation for details)
        (2) cell ontology graph related properties
    """

    ct_graph_edge_list, clid2nodeid = data_loading.get_prestored_data("cell_taxonomy_graph_owl")
    ct_graph = nx.Graph(ct_graph_edge_list)
    ct_sim = get_cell_taxonomy_similarity(ct_graph, get_raw=get_raw) # [num_node, num_node]
    
    cell_type_labelid2nodeid = get_cell_type_labelid2nodeid(data_branch, clid2nodeid)

    # num_label (~600 for human cell types appeared in our dataset) 
    # is smalled than num_node (~3k on the whole cell ontology graph)
    num_labels = len(cell_type_labelid2nodeid)
    ret_sim = np.ones((num_labels, num_labels), dtype=np.float32 if get_raw else np.int32) # [num_label, num_label]
    for i in range(num_labels):
        for j in range(num_labels):
            newi = cell_type_labelid2nodeid[i]
            newj = cell_type_labelid2nodeid[j]
            if newi != -1 and newj != -1:
                ret_sim[i, j] = ct_sim[newi, newj]
    return ret_sim, (ct_graph, clid2nodeid, cell_type_labelid2nodeid)