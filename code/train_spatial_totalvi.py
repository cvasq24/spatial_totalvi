import sys
sys.dont_write_bytecode = True

sys.path.insert(0, "/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/model/src/")
from scvi.model._totalvi import TOTALVI

import torch
import numpy as np
import mudata as md
import scanpy as sc
from graph.graph import prep_graph_splits

path_to_mdata = "/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/data/tonsil/tonsil_pp_svg.h5mu"
path_to_full_graph = "/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/data/graph/graph_k5_conn.pt"
path_to_split_graph = "/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/data/graph/k5_conn_train_val_subgraphs.pt"
path_to_model = "/ubc/cs/research/beaver/projects/carlos/spatial_totalvi/data/trained_models/conn/totalvi_k5_1layer_SGC_k1.pt"

#############################
### Load data & subgraphs ###
#############################
mdata = md.read_h5mu(path_to_mdata)
print(mdata)

subgraphs = prep_graph_splits(
    mdata,
    path_to_graph=path_to_full_graph,
    path_to_save_graph_splits=path_to_split_graph,
    train_split=0.75
)

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
    }
)

model = TOTALVI(
    mdata,
    path_to_graphs=path_to_split_graph,
    graph_n_layers=1,
    graph_conv_type="SGC",
    graph_norm_type="layer",
    graph_act_type="elu",
    graph_sgc_Kparam=1
)
print(model.module.gnn)

####################
### Train & save ###
####################
model.train(
    max_epochs=400,
    batch_size=len(subgraphs["train_indices"]),
    external_indexing=[subgraphs["train_indices"], subgraphs["val_indices"]]
)
model.save(path_to_model)