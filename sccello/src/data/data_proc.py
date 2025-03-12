import os
from tqdm import tqdm
from glob import glob
import time
import anndata
from scipy.sparse import csr_matrix, csr_array

import numpy as np
import pandas as pd
import datasets

from sccello.src.utils import data_loading

SCALING_RANGE = 10000
ENSEMBL_IDS_TAG = "ensembl_ids"
PROTOCOL_DICT = {
    "10x 5' v1": 0,
    "10x 5' v2": 1,
    "10x 3' v1": 2,
    "10x 3' v2": 3,
    "10x 3' v3": 4,
    "10x 3' transcription profiling": 5,
    "10x 5' transcription profiling": 6
}

class ProcessSingleCellData(datasets.Dataset):

    """
    Cache single cell data from h5ad formats to huggingface dataset
    """

    @staticmethod
    def to_anndata(dataset, cell_type_tag="label", input_ids_tag="input_ids", 
        expnum_tag="expnum", save_obs={}
    ):
        """
        Transform back to h5ad format
        """
        token_dct = data_loading.get_prestored_data("token_dictionary_pkl") # ensembl_id to token_id

        all_value, all_rows, all_cols = [], [], []
        for i in tqdm(range(len(dataset))):
            indices = np.array(dataset[i][input_ids_tag]).astype(np.int32)
            expnum = np.array(dataset[i][expnum_tag])
            all_rows += [i] * len(indices)
            all_cols += indices.tolist()
            all_value += expnum.tolist()

        all_values = csr_array((np.array(all_value), (np.array(all_rows), np.array(all_cols))), 
                               shape=(len(dataset), len(token_dct)))
        obs = {tag: np.array(dataset[tag]) for tag in save_obs}
        obs = pd.DataFrame({
            **obs, 
            "cell_type": np.array(dataset[cell_type_tag])
        })
        adata = anndata.AnnData(csr_matrix(all_values), 
                                var=token_dct.keys(), obs=obs)
        return adata
    
    @staticmethod
    def from_h5ad_dir_caching(
        h5ad_data_dir="./data/cell-census_2023-07-25",
        save_dir=None,
    ):
        """
        Transform multiple single cell data under the same directory to huggingface datasets,
        and cache intermediate results to local file systems
        """
        assert save_dir

        # search all file names
        h5ad_files = list(glob(os.path.join(h5ad_data_dir, "*.h5ad")))
        print(f"Total {len(h5ad_files)} h5ad files to process")

        all_cell_data = []
        for h5ad_file in tqdm(h5ad_files):
            print(f"Loading {h5ad_file} ...")
            beg_tm = time.time()
            cell_data = ProcessSingleCellData.from_h5ad_file_caching(h5ad_file, save_dir=save_dir)
            print("Processing time: ", round((time.time() - beg_tm) / 60 / 60, 4))
            
            all_cell_data.append(cell_data)
        
        all_cell_dataset = datasets.concatenate_datasets(all_cell_data)
        return all_cell_dataset

    
    @staticmethod
    def from_h5ad_file_caching(
        h5ad_file,
        save_dir=None,
        hf_suffix="_hf"
    ):
        """
        Transform a single h5ad data file to huggingface datasets, 
        and cache intermediate results to local file systems
        """
        assert save_dir
        # already saved or just need to gather from the saved
        h5ad_file_name, _ = os.path.splitext(os.path.basename(h5ad_file))
        
        cached_hf_dir = os.path.join(save_dir, h5ad_file_name + hf_suffix)
        if os.path.exists(cached_hf_dir):
            print("Already cached to local file system, loading the data from disk.")
            return datasets.load_from_disk(cached_hf_dir)
    
        hf_data = ProcessSingleCellData.from_h5ad_file(h5ad_file, is_cellxgene=True)
        hf_data.save_to_disk(cached_hf_dir)

        return hf_data

    @staticmethod
    def from_h5ad_file(h5ad_file, is_cellxgene=True, require_raw=True):
        adata = anndata.read_h5ad(h5ad_file)
        h5ad_file_name, _ = os.path.splitext(os.path.basename(h5ad_file))
        return ProcessSingleCellData.from_h5ad_adata(adata, h5ad_file_name, is_cellxgene=is_cellxgene, require_raw=require_raw)


    @staticmethod
    def from_h5ad_adata(
        adata, h5ad_file_name, is_cellxgene=True, require_raw=True, batch_size=512
    ):
        # prepare gene ensembl ids
        gene_median_dict = data_loading.get_prestored_data("gene_median_dictionary_pkl")
        gene_ensembl_id_keys = list(gene_median_dict.keys())
        gene_ensembl_ids_dict = dict(zip(gene_ensembl_id_keys, [True] * len(gene_ensembl_id_keys)))
        
        token_vocab_dict = data_loading.get_prestored_data("token_dictionary_pkl") # ensembl_id to token_id

        ### prepare adata ###
        adata = ProcessSingleCellData.preprocess_anndata(adata, require_raw=require_raw)

        # filter genes (columns)
        filter_genes = adata.var[ENSEMBL_IDS_TAG].isin(gene_ensembl_ids_dict)
        if is_cellxgene:
            # filter cells (rows)
            filter_organism = adata.obs.organism == "Homo sapiens"
            filter_assay = adata.obs.assay.isin(PROTOCOL_DICT.keys())
            filter_primary = adata.obs.is_primary_data
            filter_cells = np.bitwise_and.reduce([filter_organism, filter_assay, filter_primary])
        
        print(f"{h5ad_file_name} has adata of {adata.n_obs} cells x {adata.n_vars} genes. After filtering,\n"
              f"\t#ignored genes: {adata.n_vars - filter_genes.sum()}\n")
        if is_cellxgene:
            print(f"\t#ignored cells: {adata.n_obs - filter_cells.sum()}\n")
        
        adata = adata[:, filter_genes]
        if is_cellxgene:
            adata = adata[filter_cells, :]
        print(f"... resulted in adata of {adata.n_obs} cells x {adata.n_vars} genes")

        assert adata.n_obs != 0, "[warning]: this dataset is not in our interests, potentially not human data"

        # get norm factor similar to Geneformer
        norm_factor_vector = adata.var[ENSEMBL_IDS_TAG].map(gene_median_dict).to_numpy()

        gene_expression_nums, gene_token_ids = [], []
        cell_counts = []
        for idx in tqdm(range(0, adata.n_obs, batch_size)):
            batch_data = adata[idx:idx+batch_size, :]

            # size factor normalization
            size_counts = batch_data.X.sum(axis=1)
            # avoid zero division error
            size_counts += size_counts == 0.
            size_factor = SCALING_RANGE / size_counts
            cell_counts += size_counts.tolist()
            
            if is_cellxgene:
                expression_num = batch_data.X.multiply(size_factor.reshape((-1, 1))) # coo sparse view: [bsz, #shared_genes]
                expression_rank = expression_num.multiply(1.0 / norm_factor_vector.reshape((1, -1)))
                expression_num = expression_num.log1p()
            else:
                expression_num = np.array(batch_data.X) * size_factor.reshape((-1, 1))
                expression_rank = expression_num * (1.0 / norm_factor_vector.reshape((1, -1)))
                expression_num = np.log1p(expression_num)
            
            # sort similar to Geneformer, after w/ both column-wise and row-wise normalization
            for j in range(batch_data.n_obs):
                if is_cellxgene:
                    rx, ry = expression_rank.getrow(j), expression_num.getrow(j)
                else:
                    rx, ry = expression_rank[j], expression_num[j]
                expnum, token_ids = ProcessSingleCellData._extract_info_from_cell(
                    rx, ry, batch_data.var[ENSEMBL_IDS_TAG], token_vocab_dict, is_cellxgene=is_cellxgene)
                gene_expression_nums.append(expnum)
                gene_token_ids.append(token_ids)
        
        # get annotations
        cell_dataset_id = [h5ad_file_name] * adata.n_obs
        if is_cellxgene:
            cell_disease = adata.obs["disease"].tolist()
            cell_assay_ids = adata.obs["assay"].map(PROTOCOL_DICT).tolist()
            donor_local_id_dict = dict([(k, i) for i, k in enumerate(set(adata.obs["donor_id"].tolist()))])
            cell_donor_local_ids = adata.obs["donor_id"].map(donor_local_id_dict).tolist()
            cell_ct_ontology = adata.obs["cell_type_ontology_term_id"].tolist()
            cell_ct = adata.obs["cell_type"].tolist()
            cell_tissue = adata.obs["tissue"].tolist()
            cell_tissue_ontology = adata.obs["tissue_ontology_term_id"].tolist()
            cell_dev = adata.obs["development_stage"].tolist()

        dataset_dict = {
            "gene_expression_nums": gene_expression_nums, 
            "gene_token_ids": gene_token_ids, 
            "cell_dataset_id": cell_dataset_id,
            "cell_counts": cell_counts,
        }

        if is_cellxgene:
            dataset_dict = {
                **dataset_dict,
                "cell_disease": cell_disease,
                "cell_assay_ids": cell_assay_ids,
                "cell_donor_local_ids": cell_donor_local_ids,
                "cell_ct_ontology": cell_ct_ontology,
                "cell_type": cell_ct,
                "cell_tissue": cell_tissue,
                "cell_tissue_ontology": cell_tissue_ontology,
                "cell_dev": cell_dev,
            }

        return datasets.Dataset.from_pandas(pd.DataFrame(dataset_dict))

    @staticmethod
    def _extract_info_from_cell(exprank, expnum, gene_ensembl_ids, token_vocab_dict, is_cellxgene=True):
        """
        For one cell, sort gene token ids based on normalized expression
        """
        if is_cellxgene:
            assert (np.nonzero(exprank)[1] == np.nonzero(expnum)[1]).all()
            # mask undetected genes
            nonzero_mask = np.nonzero(exprank)[1]
            # sort by normalized gene expression values
            # prepend <cls>
            sorted_indices = np.argsort(-exprank.data)

            sorted_ensembl_ids = (gene_ensembl_ids.iloc[nonzero_mask]).iloc[sorted_indices] # pd.Series
            sorted_expnums = [0] + expnum.data[sorted_indices].tolist()
            sorted_token_ids = [token_vocab_dict["<cls>"]] + [token_vocab_dict[x] for x in sorted_ensembl_ids]
        else:
            assert (np.nonzero(exprank)[0] == np.nonzero(expnum)[0]).all()
            nonzero_mask = np.nonzero(exprank)[0]
            sorted_indices = np.argsort(-exprank[nonzero_mask])
            sorted_ensembl_ids = (gene_ensembl_ids.iloc[nonzero_mask]).iloc[sorted_indices] # pd.Series
            sorted_expnums = [0] + expnum[nonzero_mask][sorted_indices].tolist()
            sorted_token_ids = [token_vocab_dict["<cls>"]] + [token_vocab_dict[x] for x in sorted_ensembl_ids]

        return sorted_expnums, sorted_token_ids


    @staticmethod
    def preprocess_anndata(adata, require_raw=True):
        if require_raw:
            try:
                adata.X = adata.raw.X # X is already normalized
            except:
                assert (adata.X.astype("int") != adata.X).nnz == 0
        if ENSEMBL_IDS_TAG not in adata.var:
            adata.var[ENSEMBL_IDS_TAG] = data_loading.map_gene_name2id(adata.var_names.array)
        
        return adata
