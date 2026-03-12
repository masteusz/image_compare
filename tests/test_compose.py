import numpy as np
import pytest
from PIL import Image

from image_compare.compose import cluster_images, create_grid


class TestClusterImages:
    def test_single_image_returns_single_cluster(self):
        matrix = np.array([[1.0]])
        clusters = cluster_images(matrix)
        assert clusters == [[0]]

    def test_identical_images_in_same_cluster(self):
        matrix = np.array([
            [1.0, 1.0, 0.1],
            [1.0, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ])
        clusters = cluster_images(matrix)
        # Images 0 and 1 should be in same cluster, 2 separate
        all_indices = {idx for cluster in clusters for idx in cluster}
        assert all_indices == {0, 1, 2}

        for cluster in clusters:
            if 0 in cluster:
                assert 1 in cluster

    def test_all_similar_images_in_one_cluster(self):
        matrix = np.array([
            [1.0, 0.9, 0.85],
            [0.9, 1.0, 0.88],
            [0.85, 0.88, 1.0],
        ])
        clusters = cluster_images(matrix, threshold=0.5)
        assert len(clusters) == 1

    def test_dissimilar_images_in_separate_clusters(self):
        matrix = np.array([
            [1.0, 0.1],
            [0.1, 1.0],
        ])
        clusters = cluster_images(matrix, threshold=0.5)
        assert len(clusters) == 2


class TestCreateGrid:
    def test_creates_grid_image(self):
        images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "green", "blue"]]
        labels = ["model_a", "model_b", "model_c"]
        grid = create_grid(images, labels)
        assert isinstance(grid, Image.Image)
        assert grid.size[0] > 0
        assert grid.size[1] > 0

    def test_respects_cols_parameter(self):
        images = [Image.new("RGB", (100, 100), color="red") for _ in range(4)]
        labels = ["a", "b", "c", "d"]
        grid = create_grid(images, labels, cols=2)
        # 2 cols of 100px = 200 width
        assert grid.size[0] == 200

    def test_single_image(self):
        images = [Image.new("RGB", (100, 100), color="red")]
        labels = ["model_a"]
        grid = create_grid(images, labels)
        assert isinstance(grid, Image.Image)

    def test_empty_images_raises(self):
        with pytest.raises(ValueError, match="No images"):
            create_grid([], [])
