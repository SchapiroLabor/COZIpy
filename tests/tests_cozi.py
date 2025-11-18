import os
import pandas as pd
import numpy as np
import pytest

from cozipy.nep_cozi import nep_analysis, run_cozi
from cozipy.neighbors import knn_graph, radius_graph, delaunay_graph

def create_dummy_csv(folder, filename, n_cells=50, n_types=3):
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame({
        "x": np.random.rand(n_cells),
        "y": np.random.rand(n_cells),
        "cell_type": np.random.randint(0, n_types, size=n_cells)
    })
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    return filepath

def test_run_cozi_pipeline(tmp_path):
    # Create three dummy CSVs
    files = [
        create_dummy_csv(tmp_path, "img1.csv"),
        create_dummy_csv(tmp_path, "img2.csv"),
        create_dummy_csv(tmp_path, "img3.csv")
    ]

    df_list = []
    for f in files:
        img_id = os.path.splitext(os.path.basename(f))[0]
        df_tmp = pd.read_csv(f)
        df_tmp["img_id"] = img_id
        df_list.append(df_tmp)
    combined_df = pd.concat(df_list, ignore_index=True)
    
    assert "img_id" in combined_df.columns
    assert combined_df.shape[0] > 0

    for img_id in combined_df["img_id"].unique():
        df_img = combined_df[combined_df["img_id"] == img_id]
        coords = df_img[["x","y"]].values
        labels = df_img["cell_type"].values.astype(int)

        # Test all three neighborhood definitions with run_cozi
        for nbh_def in ["knn", "radius", "delaunay"]:
            res = run_cozi(coords, labels, nbh_def=nbh_def, n_neighbors=5, radius=0.2, n_permutations=10)
            
            # Check that outputs exist
            assert "cond_ratio" in res
            assert "zscore" in res
            assert isinstance(res["cond_ratio"], pd.DataFrame)
            assert isinstance(res["zscore"], pd.DataFrame)

            # Shapes should match number of cell types
            n_types = len(np.unique(labels))
            assert res["cond_ratio"].shape == (n_types, n_types)
            assert res["zscore"].shape == (n_types, n_types)

            # Values should be finite
            assert np.all(np.isfinite(res["cond_ratio"].values))
            assert np.all(np.isfinite(res["zscore"].values))
