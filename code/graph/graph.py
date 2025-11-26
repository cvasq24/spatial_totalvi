import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix



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

