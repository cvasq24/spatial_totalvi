import sys
sys.dont_write_bytecode = True
dependency_path = ""
sys.path.insert(0, f"{dependency_path}model/src/")
from scvi.model._totalvi import TOTALVI
import os
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_DISABLE_FAST_MM"] = "1"
import warnings
warnings.filterwarnings("ignore", message="Intel MKL")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import mudata as md
import scanpy as sc
from graph.graph import prep_graph_splits
import muon


choices = [
    "6_conn", "6_sim_gau", "6_sim_inv",
    "10_conn", "10_sim_gau", "10_sim_inv",
    "30_conn", "30_sim_gau", "30_sim_inv",
    "50_conn", "50_sim_gau", "50_sim_inv"
]
for choice in choices:

    path_to_mdata = f"{dependency_path}data/tonsil/tonsil_pp_svg2.h5mu"
    path_to_full_graph = f"{dependency_path}data/graph/graph_k{choice}.pt"
    path_to_split_graph = f"{dependency_path}data/graph/k{choice}_train_val_subgraphs.pt"
    path_to_model = f"{dependency_path}data/trained_models/conn/totalvi_k{choice}_1layer_SGC_k1.pt"

    #############################
    ### Load data & subgraphs ###
    #############################
    mdata = muon.read_h5mu(path_to_mdata)
   # print(mdata)
    print(path_to_full_graph)
   # print(path_to_split_graph)

    subgraphs = prep_graph_splits(
        mdata,
        path_to_graph=path_to_full_graph,
        path_to_save_graph_splits=path_to_split_graph,
        train_split=0.75
    )

    #spatial_totalvi/data/tonsil/tonsil_pp_svg2.h5mu

    ######################
    ### Set up totalVI ###
    ######################
    TOTALVI.setup_mudata(
        mdata,
        rna_layer=None,
        protein_layer=None,
        modalities ={
            "rna_layer": "RNA",
            "protein_layer": "Protein"
 