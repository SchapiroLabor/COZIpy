import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from cozipy.neighbors import knn_graph, radius_graph, delaunay_graph
import warnings

def _encode_labels(labels):
    labels = pd.Series(labels, dtype="category")
    codes = labels.cat.codes.to_numpy()
    categories = labels.cat.categories.to_list()
    return codes, categories


def _compute_pair_counts_and_denominators(adj, labels_int, n_types):
    """Compute edge counts and denominators for NEP analysis.
    
    Returns
    -------
    counts : ndarray
        [A, B] = total edges from type A to type B.
    denom : ndarray
        [A, B] = number of cells of type A with >=1 neighbor of type B.
    """
    i, j = adj.row, adj.col
    labels_int = np.asarray(labels_int)

    if n_types == 0 or labels_int.size == 0:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int)

    labels_int_64 = labels_int.astype(np.int64)
    n_types_64 = np.int64(n_types)

    counts = np.bincount(
        labels_int_64[i] * n_types_64 + labels_int_64[j],
        minlength=n_types_64 * n_types_64
    ).reshape(n_types, n_types)

    neigh_labels = labels_int[j]
    cell_has_neighbor_type = np.zeros((len(labels_int), n_types), dtype=bool)
    cell_has_neighbor_type[i, neigh_labels] = True

    denom = np.zeros((n_types, n_types), dtype=int)
    for A in range(n_types):
        denom[A] = cell_has_neighbor_type[labels_int == A].sum(axis=0)

    return counts, denom


def nep_analysis(adj,
                 labels,
                 n_permutations=1000,
                 random_state=None,
                 return_df=True,
                 fixed_type=None,
                 min_cell_count=0,
                 normalize_zscore=False):
    """Perform NEP (neighbor preference) analysis.
    
    Parameters
    ----------
    adj : scipy.sparse COO matrix
        Adjacency matrix of neighborhood graph.
    labels : array-like
        Cell type labels.
    n_permutations : int, default 1000
        Number of permutations for statistical testing.
    random_state : int or None, default None
        Random seed for reproducibility.
    return_df : bool, default True
        If True, return DataFrame; if False, return dict.
    fixed_type : str or int or None, default None
        Cell type to keep fixed during permutations.
    min_cell_count : int, default 0
        Filter out types with fewer cells.
    normalize_zscore : bool, default False
        Normalize z-scores by sqrt(total cell count).
    
    Returns
    -------
    DataFrame or dict
        Results with columns/keys: index_ct, neighbor_ct, index_ct_counts,
        neighbor_ct_counts, zscore, cond_ratio.
    """
    rng = np.random.default_rng(random_state)
    labels_int, label_names = _encode_labels(labels)
    n_types = len(label_names)
    total_cell_count = len(labels_int)

    if n_types == 0:
        if return_df:
            return pd.DataFrame(columns=['source_ct', 'target_ct', 'cond_ratio', 'zscore'])
        else:
            return {
                "cond_ratio": np.array([]).reshape(0, 0),
                "zscore": np.array([]).reshape(0, 0),
                "index_ct_counts": np.array([]).reshape(0, 0),
                "neighbor_ct_counts": np.array([]).reshape(0, 0),
                "interaction_count": 0,
            }

    obs_counts, obs_denom = _compute_pair_counts_and_denominators(adj, labels_int, n_types)
    obs_norm = obs_counts / np.maximum(obs_denom, 1)

    cond_ratio = np.zeros((n_types, n_types), float)
    for A in range(n_types):
        total_A = (labels_int == A).sum()
        cond_ratio[A] = obs_denom[A] / max(total_A, 1)

    index_ct_counts_matrix = obs_denom
    neighbor_ct_counts_matrix = obs_counts

    perm_norm = np.zeros((n_permutations, n_types, n_types), float)

    if fixed_type is not None and isinstance(fixed_type, str):
        fixed_type = label_names.index(fixed_type)

    for k in range(n_permutations):
        if fixed_type is None:
            perm = rng.permutation(labels_int)
        else:
            fixed_mask = labels_int == fixed_type
            perm = labels_int.copy()
            perm[~fixed_mask] = rng.permutation(labels_int[~fixed_mask])

        c, d = _compute_pair_counts_and_denominators(adj, perm, n_types)
        perm_norm[k] = c / np.maximum(d, 1)

    expected = perm_norm.mean(axis=0)
    std = perm_norm.std(axis=0) + 1e-6
    z = (obs_norm - expected) / std

    if normalize_zscore:
        z = z / np.sqrt(total_cell_count)

    if min_cell_count > 0:
        total_counts = np.array([(labels_int == A).sum() for A in range(n_types)])
        keep_mask = total_counts >= min_cell_count
        if not keep_mask.all():
            removed_types = [name for name, keep in zip(label_names, keep_mask) if not keep]
            print(f"Cell type(s) {', '.join(map(str, removed_types))} were removed because they have fewer than {min_cell_count} cells.")
        if not keep_mask.any():
            if return_df:
                return pd.DataFrame(columns=['source_ct', 'target_ct', 'cond_ratio', 'zscore'])
            else:
                return {
                    "cond_ratio": np.zeros((0, 0), float),
                    "zscore": np.zeros((0, 0), float),
                    "index_ct_counts": np.zeros((0, 0), int),
                    "neighbor_ct_counts": np.zeros((0, 0), int),
                    "interaction_count": 0,
                }
        else:
            label_names = [name for name, keep in zip(label_names, keep_mask) if keep]
            cond_ratio = cond_ratio[keep_mask][:, keep_mask]
            z = z[keep_mask][:, keep_mask]
            index_ct_counts_matrix = index_ct_counts_matrix[keep_mask][:, keep_mask]
            neighbor_ct_counts_matrix = neighbor_ct_counts_matrix[keep_mask][:, keep_mask]
            n_types = len(label_names)

    if return_df:
        idx = label_names
        df = pd.DataFrame({
            "index_ct": np.repeat(idx, len(idx)),
            "neighbor_ct": np.tile(idx, len(idx)),
            "index_ct_counts": index_ct_counts_matrix.ravel(),
            "neighbor_ct_counts": neighbor_ct_counts_matrix.ravel(),
            "zscore": z.ravel(),
            "cond_ratio": cond_ratio.ravel(),
        })
        return df
    else:
        return {
            "cond_ratio": cond_ratio,
            "zscore": z,
            "index_ct_counts": index_ct_counts_matrix,
            "neighbor_ct_counts": neighbor_ct_counts_matrix,
        }


def run_cozi(
    coords,
    labels,
    nbh_def="knn",
    n_neighbors=6,
    radius=0.2,
    n_permutations=100,
    random_state=None,
    fixed_type=None,
    return_df=True,
    min_cell_count=0,
    normalize_zscore=False
):
    """Run NEP analysis with specified neighborhood definition.
    
    Parameters
    ----------
    coords : ndarray
        (n_cells, 2) array of cell coordinates.
    labels : array-like
        Cell type labels.
    nbh_def : str, default "knn"
        Neighborhood: 'knn', 'radius', or 'delaunay'.
    n_neighbors : int, default 6
        Number of neighbors for knn.
    radius : float, default 0.2
        Radius for radius neighborhood.
    n_permutations : int, default 100
        Number of permutations.
    random_state : int or None, default None
        Random seed.
    fixed_type : str or int or None, default None
        Cell type to keep fixed.
    min_cell_count : int, default 0
        Filter out cell types with fewer cells.
    normalize_zscore : bool, default False
        Normalize z-scores by sqrt(total count).
    """
    if nbh_def == "knn":
        adj = knn_graph(coords, n_neighbors=n_neighbors)
    elif nbh_def == "radius":
        adj = radius_graph(coords, radius=radius)
    elif nbh_def == "delaunay":
        adj = delaunay_graph(coords)
    else:
        raise ValueError(f"Unknown neighborhood definition: {nbh_def}")

    return nep_analysis(adj, labels, n_permutations=n_permutations, random_state=random_state,
                        fixed_type=fixed_type, min_cell_count=min_cell_count, return_df=return_df,
                        normalize_zscore=normalize_zscore)
