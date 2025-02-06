# scCello Processed Data

## Pretraining Dataset
- [scCello pretraining dataset](https://huggingface.co/datasets/katarinayuan/scCello_pretrain_unsplitted) is processed from [CellxGene census LTS release 2023-07-25](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html). We select all primary data with 10x protocols sequencing on non-cancer human cells. See paper **App. B Data Preprocessing Details** for details.

## Gene Token Vocabulary
- [token_vocabulary/token_dictionary.pkl](token_vocabulary/token_dictionary.pkl): We use Geneformer's gene vocabulary. The vocabulary has 25424 gene ensembl ids, with 3 special tokens "pad", "mask" and "cls" (total vocab size 25427). 
- Matching ensembl ids with names using `biomart`:
    - Matched names: 25137 genes ([token_vocabulary/vocab_id2name.cs](token_vocabulary/vocab_id2name.csv)`)
    - Unmatched names: 291 genes ([token_vocabulary/vocab_ids_notFoundName.csv](token_vocabulary/vocab_ids_notFoundName.csv))
- [token_vocabulary/gene_median_dictionary.pkl](token_vocabulary/gene_median_dictionary.pkl): Non-zero median value of expression of each detected gene across all cells for Geneformer-like gene-wise normalization.

## Cell Type Label
- [new_pretrain/general_CLid2cellname.pkl](new_pretrain/general_CLid2cellname.pkl): Associates textual cell types used in pre-training with their cell type lineage ID (CLID).
- [new_pretrain/pretrain_frac100_clid2name.pkl](new_pretrain/pretrain_frac100_clid2name.pkl): Maps CLID to cell type label indices used in pre-training.
- [new_pretrain/pretrain_frac100_cell_type_idmap.pkl](new_pretrain/pretrain_frac100_cell_type_idmap.pkl): Associates textual cell types used in pre-training with their cell type label indices. Note that this file is not consistent with [new_pretrain/general_CLid2cellname.pkl](new_pretrain/general_CLid2cellname.pkl) and [new_pretrain/pretrain_frac100_clid2name.pkl](new_pretrain/pretrain_frac100_clid2name.pkl). Its dict keys is used for the correct dict mapping, which can be obtained from `get_prestored_data` in `sccello/src/utils/data_loading.py`.

## Cell Ontology Graph
- [cell_taxonomy/cl.owl](cell_taxonomy/cl.owl): Cell ontology graph obtained from [Cell Ontology](https://bioportal.bioontology.org/ontologies/CL).
- [cell_taxonomy/celltype_relationship.json](cell_taxonomy/celltype_relationship.json): A simpler version of cell ontology that adopts a tree structure for subclass relationships obtained from the authors of [Cell Taxonomy](https://ngdc.cncb.ac.cn/celltaxonomy/). Note that we only use this data to associate textual cell types with their cell type lineage ID (CLID), since we are using the graph version of cell ontology.


## Marker Gene Label
- [marker_gene/cellmarkers.tsv](marker_gene/cellmarkers.tsv): All cell types with their marker genes obtained from [Cell Marker](http://xteam.xbio.top/CellMarker/download/Human_cell_markers.txt) and [PanglaoDB](https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz). 
- [marker_gene/celllabel2typesquality.csv](marker_gene/celllabel2typesquality.csv): Aligns cell labels provided in downstream datasets to cell types.

## Cancer Drug Response
- Adapted from [DeepCDR repo](https://github.com/kimmo1019/DeepCDR/tree/master/data).
