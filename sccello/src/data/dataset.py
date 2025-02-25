import os
import csv
import random
import hickle as hkl
from collections import Counter
import anndata

import numpy as np
import pandas as pd

import pandas as pd

from datasets import load_dataset

from sccello.src import utils
from sccello.src.data.tokenizer import TranscriptomeTokenizer
from sccello.src.data.data_proc import ProcessSingleCellData

EXC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

SCALING_RANGE = 10_000
TRUNCATE_LENGTH = 2048

class CellTypeClassificationDataset():
    seed = 42
    num_proc = 16

    subsets = {
        "frac": ["celltype", "tissue", "donor"]
    }

    @classmethod
    def create_dataset(cls, subset_name="celltype"):
        assert subset_name in cls.subsets["frac"]
        valid_data = load_dataset(f"katarinayuan/scCello_ood_{subset_name}_data1", split="train")
        test_data = load_dataset(f"katarinayuan/scCello_ood_{subset_name}_data2", split="train")

        valid_data = valid_data.rename_column("cell_type", "label")
        test_data = test_data.rename_column("cell_type", "label")
        return valid_data, test_data
    

class MarkerGenePredictionDataset():

    suffix = {
        "GSE96583_1": "-46B",
        "GSE96583_2": "-45A",
        "GSE96583_3": "-482.1",
        "GSE96583_4": "-492.2",
        "GSE96583_5": "-47C",
        "GSE130148": "",
    }
    split_ratio = [0.8, 0.1, 0.1]
    seed = 42
    num_proc = 16

    @classmethod
    def create_dataset(cls, data_name, add_marker_genes=True, without_split=False):
        
        assert data_name in cls.suffix

        raw_dataset = load_dataset(f"katarinayuan/scCello-{data_name.split('_')[0]}{cls.suffix[data_name]}")["train"]

        raw_dataset = raw_dataset.rename_column("gene_ensembl_ids", "gene_token_ids")
        raw_dataset = raw_dataset.rename_column("gene_expression_nums", "expnum")
        raw_dataset = raw_dataset.remove_columns("gene_class")
        raw_dataset = raw_dataset.remove_columns("gene_superclass")
        def measure_length(example):
            example["length"] = [len(x) for x in example["gene_token_ids"]]
            return example

        raw_dataset = raw_dataset.map(measure_length, num_proc=cls.num_proc, batched=True, batch_size=2048)
        tk = TranscriptomeTokenizer(num_proc=cls.num_proc, truncate_length=TRUNCATE_LENGTH)
        raw_dataset = tk.tokenize(raw_dataset)

        data, targetdict = cls._preprocess_single(raw_dataset, data_name=data_name, add_marker_genes=add_marker_genes)

        return data, targetdict

    @classmethod
    def _preprocess_single(cls, dataset, data_name=None, add_marker_genes=False):

        dataset_subset = cls._quality_control_filtering(dataset)
        # rename columns
        dataset_subset = dataset_subset.rename_column("cell_type", "label")
        if add_marker_genes:
            dataset_subset = cls._add_marker_genes(dataset_subset, data_name)
        target_name_id_dict = cls._create_cell_type_dict(dataset_subset)
        labeled_dataset = cls._map_labels2ids(dataset_subset, target_name_id_dict)
        
        return labeled_dataset, target_name_id_dict
    
    @classmethod
    def _quality_control_filtering(cls, dataset):
        
        # per scDeepsort published method, drop cell types representing <0.5% of cells
        # ensure every train/valid/test split could have at least one instance
        celltype_counter = Counter(dataset["cell_type"])
        total_cells = sum(celltype_counter.values())
        cells_to_keep = [k for k, v in celltype_counter.items() 
                         if v > (0.005 * total_cells) and v - int(v * cls.split_ratio[0] + 0.5) > 1]
        
        def if_not_rare_celltype(example):
            return example["cell_type"] in cells_to_keep
        dataset_subset = dataset.filter(if_not_rare_celltype, num_proc=cls.num_proc)
        return dataset_subset
    
    @classmethod
    def _add_marker_genes(cls, dataset, data_name):
        cell_label2marker = utils.data_loading.get_cell_type_label2marker(data_name)
        def proc_marker_genes(example):
            example["marker_genes"] = cell_label2marker[example["label"]] if example["label"] in cell_label2marker else []
            return example
        dataset = dataset.map(proc_marker_genes, num_proc=cls.num_proc, batched=False)
        return dataset
    
    @classmethod
    def _create_cell_type_dict(cls, dataset):
        # create dictionary of cell types : label ids
        target_names = list(Counter(dataset["label"]).keys())
        target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

        return target_name_id_dict

    @classmethod
    def _map_labels2ids(cls, dataset, target_name_id_dict):
        # change labels to numerical ids
        def classes_to_ids(example):
            example["label"] = target_name_id_dict[example["label"]]
            return example
        labeled_dataset = dataset.map(classes_to_ids, num_proc=cls.num_proc)

        return labeled_dataset
    
class CancerDrugResponseDataset():
    data_dir = os.path.join(EXC_DIR, "data", "cancer_drug_response")
    seed = 0 # follow DeepCDR
    num_proc = 16

    paths = {
        "DeepCDR_data": {
            "drug_info_file": "./GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv",
            "cell_line_info_file": "./CCLE/Cell_lines_annotations_20181226.txt",
            "drug_feature_file": "./GDSC/drug_graph_feat",
            "genomic_mutation_file": "./CCLE/genomic_mutation_34673_demap_features.csv",
            "cancer_response_exp_file": "./CCLE/GDSC_IC50.csv",
            "gene_expression_file": "./CCLE/genomic_expression_561celllines_697genes_demap_features.csv",
            "methylation_file": "./CCLE/genomic_methylation_561celllines_808genes_demap_features.csv"
        }
    }

    metadatas = {
        "DeepCDR_data": {
            "TCGA_label_set": [
                "ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                "STAD", "THCA", "COAD/READ"
            ]
        }
    }
    

    @classmethod
    def create_dataset(cls, data_name, **kwargs):
        
        if data_name == "DeepCDR_data":
            return cls.preprocess_DeepCDR_data(data_name, **kwargs)
        else:
            raise NotImplementedError

    @classmethod
    def preprocess_DeepCDR_data(cls, data_name, ratio=0.95):
        """
        adapted from https://github.com/biomap-research/scFoundation/blob/main/DeepCDR/prog/run_DeepCDR.py
        """

        #drug_id --> pubchem_id
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["drug_info_file"])
        reader = csv.reader(open(file_name, 'r'))
        rows = [item for item in reader]
        drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

        #map cellline --> cancer type
        cellline2cancertype ={}
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["cell_line_info_file"])
        for line in open(file_name).readlines()[1:]:
            cellline_id = line.split('\t')[1]
            TCGA_label = line.strip().split('\t')[-1]
            #if TCGA_label in TCGA_label_set:
            cellline2cancertype[cellline_id] = TCGA_label

        #load demap cell lines genomic mutation features
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["genomic_mutation_file"])
        mutation_feature = pd.read_csv(file_name, sep=',', header=0, index_col=[0])
        cell_line_id_set = list(mutation_feature.index)

        # load drug features
        drug_pubchem_id_set = []
        drug_feature = {}
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["drug_feature_file"])
        for each in os.listdir(file_name):
            drug_pubchem_id_set.append(each.split('.')[0])
            # print('%s/%s'%(Drug_feature_file,each))
            feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (file_name, each))
            drug_feature[each.split('.')[0]] = [feat_mat,adj_list, degree_list]
        assert len(drug_pubchem_id_set)==len(drug_feature.values())
        
        #load gene expression faetures
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["gene_expression_file"])
        gexpr_feature = pd.read_csv(file_name, sep=',', header=0, index_col=[0])
        
        #only keep overlapped cell lines
        mutation_feature = mutation_feature.loc[gexpr_feature.index.tolist()]
        
        #load methylation 
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["methylation_file"])
        methylation_feature = pd.read_csv(file_name, sep=',', header=0, index_col=[0])
        assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
        
        file_name = os.path.join(cls.data_dir, data_name, cls.paths[data_name]["cancer_response_exp_file"])
        experiment_data = pd.read_csv(file_name, sep=',', header=0, index_col=[0])
        #filter experiment data
        drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
        experiment_data_filtered = experiment_data.loc[drug_match_list]
        
        data_idx = []
        for each_drug in experiment_data_filtered.index:
            for each_cellline in experiment_data_filtered.columns:
                pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                        ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                        data_idx.append((each_cellline,pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
        nb_celllines = len(set([item[0] for item in data_idx]))
        nb_drugs = len(set([item[1] for item in data_idx]))
        
        print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))

        #split into training and test set
        data_train_idx, data_test_idx = [], []
        for each_type in cls.metadatas[data_name]["TCGA_label_set"]:
            data_subtype_idx = [item for item in data_idx if item[-1] == each_type]

            train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
            test_list = [item for item in data_subtype_idx if item not in train_list]
            
            data_train_idx += train_list
            data_test_idx += test_list
        
        return (
            (mutation_feature, drug_feature, gexpr_feature, methylation_feature), 
            (data_idx, data_train_idx, data_test_idx)
        )


    @classmethod
    def from_csv(cls, data):
        # csv to AnnData
        assert isinstance(data, pd.DataFrame)
        value = data.values
        value = value / value.sum(1, keepdims=True) * SCALING_RANGE
        adata = anndata.AnnData(value)
        adata.var_names = data.columns
        
        # AnnData to HuggingFace 
        raw_dataset = ProcessSingleCellData.from_h5ad_adata(adata, "cancer_drug_response", is_cellxgene=False, require_raw=False)
        raw_dataset = raw_dataset.rename_column("gene_expression_nums", "expnum")
        def measure_length(example):
            example["length"] = [len(x) for x in example["gene_token_ids"]]
            return example

        raw_dataset = raw_dataset.map(measure_length, num_proc=cls.num_proc, batched=True, batch_size=2048)
        tk = TranscriptomeTokenizer(num_proc=cls.num_proc, truncate_length=TRUNCATE_LENGTH)
        raw_dataset = tk.tokenize(raw_dataset)

        return raw_dataset