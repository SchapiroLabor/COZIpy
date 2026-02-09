import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from cozipy.neighbors import knn_graph, radius_graph, delaunay_graph

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
    src_labels = labels_int[i]

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
                 fixed_type=None):
    """
    NEP analysis with conditional ratios and z-scores.

    Parameters
    ----------
    STILL UNDER DEVELOPMENT IF MAKES SENSE
    fixed_type : str or int or None
        If provided, this cell type will remain fixed in permutations while
        all other cell types are shuffled.
    """
    rng = np.random.default_rng(random_state)
    print("hello")
    labels_int, label_names = _encode_labels(labels)
    n_types = len(label_names)  # SAFER calculation of n_types

    # Ensure n_types > 0 before proceeding
    if n_types == 0:
        if return_df:
            # Return empty dataframes if there are no cell types
            idx = []
            return {
                "cond_ratio": pd.DataFrame([], index=idx, columns=idx),
                "zscore": pd.DataFrame([], index=idx, columns=idx),
            }
        else:
            return {
                "cond_ratio": np.array([]).reshape(0, 0),
                "zscore": np.array([]).reshape(0, 0),
            }

    # Pass the safely calculated n_types
    obs_counts, obs_denom = _compute_pair_counts_and_denominators(
        adj, labels_int, n_types)
    obs_norm = obs_counts / np.maximum(obs_denom, 1)

    cond_ratio = np.zeros((n_types, n_types), float)
    for A in range(n_types):
        total_A = (labels_int == A).sum()
        cond_ratio[A] = obs_denom[A] / max(total_A, 1)

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

    if return_df:
        idx = label_names
        z_df = pd.DataFrame(z, index=idx, columns=idx)
        cond_df = pd.DataFrame(cond_ratio, index=idx, columns=idx)
        return {
            "cond_ratio": cond_df,
            "zscore": z_df,
        }
    else:
        return {
            "cond_ratio": cond_ratio,
            "zscore": z,
        }


def run_cozi(
    coords,
    labels,
    nbh_def="knn",
    n_neighbors=6,
    radius=0.2,
    n_permutations=100,
    random_state=None,
    fixed_type=None
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
    return nep_analysis(adj, labels, n_permutations=n_permutations, random_state=random_state, fixed_type=fixed_type)
