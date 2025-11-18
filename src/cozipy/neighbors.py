import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix


def knn_graph(coords, n_neighbors=6):
    """
    Lightweight KNN graph using full distance matrix (NumPy).
    No sklearn dependency. Directional.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff ** 2, axis=2)

    idx = np.argsort(dist, axis=1)
    neighbors = idx[:, 1:n_neighbors + 1]

    rows = np.repeat(np.arange(coords.shape[0]), n_neighbors)
    cols = neighbors.flatten()
    data = np.ones(len(rows), dtype=int)

    return coo_matrix((data, (rows, cols)),
                      shape=(coords.shape[0], coords.shape[0]))


def radius_graph(coords, radius):
    """
    Directional radius graph:
    For every pair within radius, include BOTH directions (i->j and j->i).
    This guarantees asymmetry for NEP analysis.
    """
    tree = cKDTree(coords)
    neighborhoods = tree.query_ball_tree(tree, radius)

    rows, cols = [], []
    for i, neighs in enumerate(neighborhoods):
        for j in neighs:
            if i != j:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=int)
    n = coords.shape[0]

    return coo_matrix((data, (rows, cols)), shape=(n, n))


def delaunay_graph(coords):
    """
    Delaunay adjacency using SciPy.
    Produces directional edges (both i->j and j->i) in COO format.
    """
    tri = Delaunay(coords)
    simplices = tri.simplices

    edges = set()
    for s in simplices:
        a, b, c = s
        edges.add((a, b))
        edges.add((b, a))
        edges.add((b, c))
        edges.add((c, b))
        edges.add((a, c))
        edges.add((c, a))

    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    data = np.ones(len(rows), dtype=int)
    n = coords.shape[0]

    return coo_matrix((data, (rows, cols)), shape=(n, n))
