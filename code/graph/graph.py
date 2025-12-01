import torch
import numpy as np
import mudata as md
from mudata import MuData
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph


def graph_construction(
    spatial_coords: np.ndarray, 
    k: int, 
    mode: str = 'connectivity', 
    similarity_method: str = 'inverse',
    device: torch.device = torch.device('cpu')
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        spatial_coords: NumPy array of shape [N, 2] containing (x, y) coordinates.
        k: The number of K-Nearest Neighbors to connect to.
        mode: Graph construction mode. 'connectivity' (weights are 1) or 'similarity' (distance-based).
        similarity_method: If mode='similarity', the function used to convert distance to affinity.
                         Options: 'inverse' (1/Distance) or 'gaussian' (exp(-d^2)).
        device: The target device (cpu or cuda:0) where the final Tensors should reside.

    Returns:
        edge_index, edge_weight (both as PyTorch Tensors) for SGConv layer input.
    """
    if mode == 'connectivity':
        adj_matrix = kneighbors_graph(
            spatial_coords, 
            n_neighbors=k, 
            mode='connectivity', 
            include_self=False
        )
        adj_matrix = adj_matrix.tocoo()
        weights = torch.ones(adj_matrix.nnz, dtype=torch.float32)

    elif mode == 'similarity':
        dist_matrix = kneighbors_graph(
            spatial_coords, 
            n_neighbors=k, 
            mode='distance', 
            include_self=False
        )
        dist_matrix = dist_matrix.tocoo()
        distances = torch.from_numpy(dist_matrix.data).to(torch.float32)

        if similarity_method == 'inverse': # more distance -> less weight
            weights = 1.0 / (distances + 1e-6)

        else: # smth from spagcn
            sigma = distances.mean() # parameter? 
            print(f"Similarity Method: Gaussian. Using sigma={sigma:.2f}")
            weights = torch.exp(- (distances**2) / (2 * sigma**2))

        adj_matrix = dist_matrix
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'connectivity' or 'similarity'.")
        

    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    edge_weight = weights
    edge_index = edge_index.to(device)
    edge_weight = weights.to(device)
    
    return edge_index, edge_weight

def prep_graph_splits(
    mdata: MuData, 
    path_to_graph: str,
    path_to_save_graph_splits: str,
    train_split: float = 0.75
):
    # First generate the splits, get train/val indices
    all_idx = np.arange(mdata.n_obs)
    train_idx, val_idx = train_test_split(
        all_idx,
        train_size=train_split,
        random_state=0,
        shuffle=True
    )  

    # Load the entire graph
    graph = torch.load(path_to_graph)
    edge_idx = graph["edge_index"]
    edge_weight = graph["edge_weight"]

    # Get the train/val subgraphs
    train_edge_idx, train_edge_weight = subgraph(
        torch.from_numpy(train_idx),
        edge_idx,
        edge_weight,
        relabel_nodes=True
    )

    val_edge_idx, val_edge_weight = subgraph(
        torch.from_numpy(val_idx),
        edge_idx,
        edge_weight,
        relabel_nodes=True
    )    

    result = {
        "full_edge_index": edge_idx,
        "full_edge_weight": edge_weight,
        "train_edge_index": train_edge_idx,
        "train_edge_weight": train_edge_weight,
        "val_edge_index": val_edge_idx,
        "val_edge_weight": val_edge_weight,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    
    if path_to_save_graph_splits:
        torch.save(result, path_to_save_graph_splits)
    
    return result