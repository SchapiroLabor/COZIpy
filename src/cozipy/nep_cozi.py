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
    """
    Returns:
        counts[A, B] = total edges from A to B
        denom[A, B]  = number of cells of type A having >=1 neighbor of type B
    
    The indices fed to np.bincount are calculated using 64-bit integers 
    to prevent overflow on large numbers of cell types (n_types).
    """
    i, j = adj.row, adj.col
    labels_int = np.asarray(labels_int)

    if n_types == 0 or labels_int.size == 0:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int)

    # Use 64-bit integers for the calculation that previously overflowed a 32-bit integer.
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
        maskA = (labels_int == A)
        denom[A] = cell_has_neighbor_type[maskA].sum(axis=0)

    return counts, denom


def nep_analysis(adj,
                 labels,
                 n_permutations=1000,
                 random_state=None,
                 return_df=True,
                 fixed_type=None,
                 min_cell_count=0):
    """
    NEP analysis with conditional ratios and z-scores.

    Parameters
    ----------
    STILL UNDER DEVELOPMENT IF MAKES SENSE
    fixed_type : str or int or None
        If provided, this cell type will remain fixed in permutations while
        all other cell types are shuffled.
    min_cell_count : int, default 0
        Minimum number of cells required for a cell type to be included in the output.
        Cell types with fewer cells are filtered out from the results after analysis.
    """
    rng = np.random.default_rng(random_state)
    labels_int, label_names = _encode_labels(labels)
    n_types = len(label_names)  # SAFER calculation of n_types

    # Ensure n_types > 0 before proceeding
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

    # Pass the safely calculated n_types
    obs_counts, obs_denom = _compute_pair_counts_and_denominators(
        adj, labels_int, n_types)
    obs_norm = obs_counts / np.maximum(obs_denom, 1)

    cond_ratio = np.zeros((n_types, n_types), float)
    for A in range(n_types):
        total_A = (labels_int == A).sum()
        cond_ratio[A] = obs_denom[A] / max(total_A, 1)

    # Create matrices for index and neighbor counts
    index_ct_counts_matrix = obs_denom  # [A, B] = number of cells of type A with >=1 neighbor of type B
    neighbor_ct_counts_matrix = obs_counts  # [A, B] = total edges from A to B

    perm_norm = np.zeros((n_permutations, n_types, n_types), float)

    # convert fixed_type name -> integer index if needed
    if fixed_type is not None and isinstance(fixed_type, str):
        fixed_type = label_names.index(fixed_type)

    for k in range(n_permutations):
        if fixed_type is None:
            perm = rng.permutation(labels_int)
        else:
            # Mask cells to remain unchanged vs permuted
            fixed_mask = labels_int == fixed_type
            other_mask = ~fixed_mask

            # Copy original labels then permute only others
            perm = labels_int.copy()
            perm[other_mask] = rng.permutation(labels_int[other_mask])

        # Pass the safely calculated n_types
        c, d = _compute_pair_counts_and_denominators(adj, perm, n_types)
        perm_norm[k] = c / np.maximum(d, 1)

    expected = perm_norm.mean(axis=0)
    std = perm_norm.std(axis=0) + 1e-6
    z = (obs_norm - expected) / std

    # Filter cell types with low counts
    if min_cell_count > 0:
        total_counts = np.array([(labels_int == A).sum() for A in range(n_types)])
        keep_mask = total_counts >= min_cell_count
        if not keep_mask.all():
            removed_types = [name for name, keep in zip(label_names, keep_mask) if not keep]
            print(f"Cell type(s) {', '.join(map(str, removed_types))} were removed because they have fewer than {min_cell_count} cells.")
        if not keep_mask.any():
            # All filtered out, return empty
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
        z_df = pd.DataFrame(z, index=idx, columns=idx)
        z = z_df.loc[idx, idx].to_numpy()

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
    min_cell_count=0
):
    """
    Runs NEP analysis with specified neighborhood definition.
    Accepts string labels.
    """
    # build adjacency
    if nbh_def == "knn":
        adj = knn_graph(coords, n_neighbors=n_neighbors)
    elif nbh_def == "radius":
        adj = radius_graph(coords, radius=radius)
    elif nbh_def == "delaunay":
        adj = delaunay_graph(coords)
    else:
        raise ValueError(f"Unknown neighborhood definition: {nbh_def}")

    # run NEP
    return nep_analysis(adj, labels, n_permutations=n_permutations, random_state=random_state, fixed_type=fixed_type, min_cell_count=min_cell_count)
