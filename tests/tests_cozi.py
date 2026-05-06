import os
import pandas as pd
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from cozipy.nep_cozi import nep_analysis, run_cozi
from cozipy.neighbors import knn_graph, radius_graph, delaunay_graph


def create_dummy_csv(folder, filename, n_cells=50, n_types=3):
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(
        {
            "x": np.random.rand(n_cells),
            "y": np.random.rand(n_cells),
            "cell_type": np.random.randint(0, n_types, size=n_cells),
        }
    )
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    return filepath


def test_run_cozi_pipeline(tmp_path):
    files = [
        create_dummy_csv(tmp_path, "img1.csv"),
        create_dummy_csv(tmp_path, "img2.csv"),
        create_dummy_csv(tmp_path, "img3.csv"),
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
        coords = df_img[["x", "y"]].values
        labels = df_img["cell_type"].values.astype(int)

        for nbh_def in ["knn", "radius", "delaunay"]:
            res = run_cozi(
                coords,
                labels,
                nbh_def=nbh_def,
                n_neighbors=5,
                radius=0.2,
                n_permutations=10,
            )

            assert isinstance(res, pd.DataFrame)
            expected_columns = {
                "index_ct",
                "neighbor_ct",
                "index_ct_counts",
                "neighbor_ct_counts",
                "zscore",
                "cond_ratio",
            }
            assert set(res.columns) == expected_columns

            n_types = len(np.unique(labels))
            assert res.shape == (n_types * n_types, len(expected_columns))

            assert np.all(np.isfinite(res["zscore"]))
            assert np.all(np.isfinite(res["cond_ratio"]))
            assert np.issubdtype(res["index_ct_counts"].dtype, np.integer)
            assert np.issubdtype(res["neighbor_ct_counts"].dtype, np.integer)


def _simple_directed_adj():
    # 0->1, 0->2, 1->2, 2->0
    rows = np.array([0, 0, 1, 2])
    cols = np.array([1, 2, 2, 0])
    data = np.ones(len(rows), dtype=int)
    return coo_matrix((data, (rows, cols)), shape=(3, 3))


def test_nep_analysis_return_dict_shapes_and_keys():
    adj = _simple_directed_adj()
    labels = np.array(["A", "A", "B"])

    out = nep_analysis(adj, labels, n_permutations=5, random_state=7, return_df=False)

    assert set(out.keys()) == {
        "cond_ratio",
        "zscore",
        "index_ct_counts",
        "neighbor_ct_counts",
    }
    for key in out:
        assert out[key].shape == (2, 2)


def test_run_cozi_invalid_neighborhood_definition_raises():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="Unknown neighborhood definition"):
        run_cozi(coords, labels, nbh_def="invalid", n_permutations=1)


def test_min_cell_count_filters_rare_types_and_emits_message(capsys):
    adj = _simple_directed_adj()
    labels = np.array(["A", "A", "B"])

    df = nep_analysis(adj, labels, n_permutations=5, random_state=11, min_cell_count=2)
    captured = capsys.readouterr()

    assert "were removed" in captured.out
    assert set(df["index_ct"]) == {"A"}
    assert set(df["neighbor_ct"]) == {"A"}
    assert df.shape[0] == 1


def test_min_cell_count_all_removed_returns_empty_output():
    adj = _simple_directed_adj()
    labels = np.array(["A", "A", "B"])

    out_df = nep_analysis(adj, labels, n_permutations=3, min_cell_count=10, return_df=True)
    assert out_df.empty

    out_dict = nep_analysis(adj, labels, n_permutations=3, min_cell_count=10, return_df=False)
    assert out_dict["cond_ratio"].shape == (0, 0)
    assert out_dict["zscore"].shape == (0, 0)
    assert out_dict["index_ct_counts"].shape == (0, 0)
    assert out_dict["neighbor_ct_counts"].shape == (0, 0)


def test_normalize_zscore_scales_by_sqrt_cell_count():
    adj = _simple_directed_adj()
    labels = np.array(["A", "A", "B"])

    unnorm = nep_analysis(
        adj,
        labels,
        n_permutations=50,
        random_state=3,
        return_df=False,
        normalize_zscore=False,
    )
    norm = nep_analysis(
        adj,
        labels,
        n_permutations=50,
        random_state=3,
        return_df=False,
        normalize_zscore=True,
    )

    factor = np.sqrt(len(labels))
    assert np.allclose(norm["zscore"] * factor, unnorm["zscore"], atol=1e-6)


def test_fixed_type_accepts_string_and_int_equivalently():
    adj = _simple_directed_adj()
    labels = np.array(["A", "A", "B"])

    out_str = nep_analysis(adj, labels, n_permutations=20, random_state=23, fixed_type="A")
    out_int = nep_analysis(adj, labels, n_permutations=20, random_state=23, fixed_type=0)

    pd.testing.assert_frame_equal(out_str, out_int)
