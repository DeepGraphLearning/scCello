import os
import json
import rdflib
import pickle
import numpy as np
import pandas as pd
import json
import ipdb
from datasets import load_dataset

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sccello.src.utils import helpers

def get_prestored_data(data_file_name):

    prestored_files = {
        "gene_median_dictionary_pkl": f"{EXC_DIR}/data/token_vocabulary/gene_median_dictionary.pkl",
        "token_dictionary_pkl": f"{EXC_DIR}/data/token_vocabulary/token_dictionary.pkl",  # ensembl id -> input id
        "vocab_id2name_csv": f"{EXC_DIR}/data/token_vocabulary/vocab_id2name.csv",  # ensembl id; name
        
        "cell_markers_tsv": f"{EXC_DIR}/data/marker_gene/cellmarkers.tsv",
        "cell_label2types_csv": f"{EXC_DIR}/data/marker_gene/celllabel2typesquality.csv",
        
        "cell_taxonomy_graph_owl": f"{EXC_DIR}/data/cell_taxonomy/cl.owl", # (edge_list, map_CLid2nodeid) 
        "cell_taxonomy_tree_json": f"{EXC_DIR}/data/cell_taxonomy/celltype_relationship.json",

        "pretrain_newdata_frac100_cell_type_idmap_pkl": f"{EXC_DIR}/data/new_pretrain/pretrain_frac100_cell_type_idmap.pkl",
        "pretrain_newdata_frac100_clid2labelid_pkl": f"{EXC_DIR}/data/new_pretrain/pretrain_frac100_clid2name.pkl",
        "general_clid2cellname_pkl": f"{EXC_DIR}/data/new_pretrain/general_CLid2cellname.pkl",
    }
    assert data_file_name in prestored_files
    assert os.path.exists(prestored_files[data_file_name]), (f"{data_file_name} "
            f"should be prestored under {prestored_files[data_file_name]}, but does not exist")

    if data_file_name.endswith("_tsv"):
        fetched_data = pd.read_csv(prestored_files[data_file_name], sep="\t")
        return fetched_data
    elif data_file_name.endswith("_json"):
        fetched_data = json.load(open(prestored_files[data_file_name]))

        assert data_file_name == "cell_taxonomy_tree_json"
        ct_tree_edge_list, ct_tree_vocab_clid2idx, ct_tree_vocab_clid2name = [], {}, {}
        for data in fetched_data:
            u_clid, fa_clid, u_name = data["id"], data["pId"], data["name"]
            if u_clid not in ct_tree_vocab_clid2idx:
                ct_tree_vocab_clid2idx[u_clid] = len(ct_tree_vocab_clid2idx)
                ct_tree_vocab_clid2name[u_clid] = u_name
            else:
                if u_clid in ct_tree_vocab_clid2name:
                    assert ct_tree_vocab_clid2name[u_clid] == u_name
                else:
                    ct_tree_vocab_clid2name[u_clid] = u_name

            if fa_clid not in ct_tree_vocab_clid2idx:
                ct_tree_vocab_clid2idx[fa_clid] = len(ct_tree_vocab_clid2idx)
            u_id, fa_id = ct_tree_vocab_clid2idx[u_clid], ct_tree_vocab_clid2idx[fa_clid]
            ct_tree_edge_list.append((u_id, fa_id))
        
        # ensure every CL index has its corresponding name
        for k in ct_tree_vocab_clid2idx.keys():
            assert k in ct_tree_vocab_clid2name or k == "#", "Something wrong when creating cell taxonomy tree"
        
        return fetched_data, ct_tree_edge_list, ct_tree_vocab_clid2idx, ct_tree_vocab_clid2name
    
    elif data_file_name.endswith("_owl"):
        assert data_file_name == "cell_taxonomy_graph_owl"
        
        g = rdflib.Graph()
        g.parse(prestored_files[data_file_name], format='xml')
        ct_graph_edge_list, ct_graph_vocab_clid2idx = [], {}
        clid_list = []
        for s, p, o in g:
            if "CL" in s and "CL" in o:
                u_clid, fa_clid = s.split("/")[-1], o.split("/")[-1]
                u_clid = u_clid.replace("_", ":")
                fa_clid = fa_clid.replace("_", ":")
                if u_clid == fa_clid:
                    continue
                if not (u_clid.startswith("CL:") and fa_clid.startswith("CL:")): # CL:xxxxxxx
                    continue

                ct_graph_edge_list.append((u_clid, fa_clid))
                clid_list.append(u_clid)
                clid_list.append(fa_clid)
        clid_list = sorted(list(set(clid_list)))
        ct_graph_vocab_clid2idx = {v:k for k,v in enumerate(clid_list)}
        ct_graph_edge_list = [(ct_graph_vocab_clid2idx[u], ct_graph_vocab_clid2idx[v]) for u,v in ct_graph_edge_list]

        return ct_graph_edge_list, ct_graph_vocab_clid2idx

    elif data_file_name.endswith("_csv"):
        fetched_data = pd.read_csv(prestored_files[data_file_name])
        return fetched_data
    
    elif data_file_name.endswith("_pkl"):
        with open(prestored_files[data_file_name], "rb") as f:
            fetched_data = pickle.load(f)

        if data_file_name == "pretrain_newdata_frac100_cell_type_idmap_pkl":
            
            general_clid2cellname = get_prestored_data("general_clid2cellname_pkl")
            name2clid = {x: y for x,y in general_clid2cellname}
            clid2labelid = get_prestored_data(f"pretrain_newdata_frac100_clid2labelid_pkl")
            mapped_data = {name: clid2labelid[name2clid[name]] for name in fetched_data.keys()}

            return mapped_data

        if data_file_name == "token_dictionary_pkl":
            fetched_data["<cls>"] = len(fetched_data)
        
        return fetched_data
    else:
        raise NotImplementedError

def get_fracdata_sample(name, num_proc=12, num_samples=10):

    from sccello.src.data.dataset import CellTypeClassificationDataset
    data1, data2 = CellTypeClassificationDataset.create_dataset(name)
    data1 = data1.rename_column("gene_token_ids", "input_ids")
    data2 = data2.rename_column("gene_token_ids", "input_ids")

    data1, data2 = data1.select(range(num_samples)), data2.select(range(num_samples))

    data1, eval_label_type_idmap = helpers.process_label_type(data1, num_proc, "label")
    data2, test_label_type_idmap = helpers.process_label_type(data2, num_proc, "label")

    return None, data1, data2, None

def get_fracdata(name, data_branch, indist, batch_effect, num_proc=12):
    
    from sccello.src.data.dataset import CellTypeClassificationDataset
    data1, data2 = CellTypeClassificationDataset.create_dataset(name)
    label_dict = get_prestored_data(f"pretrain_newdata_{data_branch}_cell_type_idmap_pkl")
    trainset = load_dataset("katarinayuan/scCello_pretrain_unsplitted")["train"]
    trainset = trainset.rename_column("cell_type", "label")

    trainset = trainset.train_test_split(test_size=0.001, seed=237)
    if indist: # replace the test data as ID data $D^{id}$
        train_split = trainset["train"].train_test_split(test_size=0.1, seed=42)
        trainset, data1, data2 = train_split["train"], train_split["test"], trainset["test"]
    else:
        trainset = trainset["train"]
    
    trainset = trainset.rename_column("gene_token_ids", "input_ids")
    data1 = data1.rename_column("gene_token_ids", "input_ids")
    data2 = data2.rename_column("gene_token_ids", "input_ids")

    # relabeling
    if batch_effect:
        trainset, _ = helpers.process_label_type(trainset, num_proc, "cell_dataset_id")
        data1, _ = helpers.process_label_type(data1, num_proc, "cell_dataset_id")
        data2, _ = helpers.process_label_type(data2, num_proc, "cell_dataset_id")

    if (not indist):
        data1, eval_label_type_idmap = helpers.process_label_type(data1, num_proc, "label")
        data2, test_label_type_idmap = helpers.process_label_type(data2, num_proc, "label")
        return trainset, data1, data2, {"train": label_dict, 
                                            "eval": eval_label_type_idmap, 
                                            "test": test_label_type_idmap}
    
    return trainset, data1, data2, label_dict

def map_gene_name2id(data_genes):

    assert (data_genes.astype("str") == data_genes).all()
    gene_names = data_genes.astype("str")
    if gene_names[0].startswith("ENSG"):
        return gene_names
    
    vocab_id2name_csv = get_prestored_data("vocab_id2name_csv")
    vocab_name2id_dict = vocab_id2name_csv.set_index("name").to_dict()["id"]
    gene_ids = np.vectorize(vocab_name2id_dict.get)(gene_names)
    return gene_ids

def get_cell_type_label2marker(data_name):

    cell_markers_tsv = get_prestored_data("cell_markers_tsv")
    cell_type2markers = cell_markers_tsv.set_index("cellType").to_dict()["geneMarker"]
    token_vocab_dict = get_prestored_data("token_dictionary_pkl")

    cell_label2types_csv = get_prestored_data("cell_label2types_csv")

    cell_label2marker = {}
    for x in cell_label2types_csv.iloc():
        data, original_label, aligned_label = x["dataset"], x["original label"], x["aligned label"]
        if data in data_name:
            cell_label2marker[original_label] = []
            for aligned_type in aligned_label.split(","):
                if aligned_type not in cell_type2markers:
                    continue
                marker_list = cell_type2markers[aligned_type].split(",")
                marker_ensembl_ids = map_gene_name2id(np.array(marker_list))
                marker_input_ids = [token_vocab_dict[x] for x in marker_ensembl_ids if x is not None and x != 'None']
                cell_label2marker[original_label] += marker_input_ids
            cell_label2marker[original_label] = list(set(cell_label2marker[original_label]))
    
    return cell_label2marker
