import numpy as np
import pytest

from image_compare.metrics import (
    compute_histogram_correlation,
    compute_pairwise_similarities,
    compute_phash_similarity,
    compute_ssim,
)


class TestComputeSSIM:
    def test_identical_images_return_one(self, solid_red_array):
        assert compute_ssim(solid_red_array, solid_red_array) == pytest.approx(1.0, abs=0.01)

    def test_different_images_return_less_than_one(self, solid_red_array, solid_blue_array):
        result = compute_ssim(solid_red_array, solid_blue_array)
        assert result < 1.0

    def test_noise_images_have_low_similarity(self, noise_array_a, noise_array_b):
        result = compute_ssim(noise_array_a, noise_array_b)
        assert result < 0.5


class TestComputeHistogramCorrelation:
    def test_identical_images_return_one(self, solid_red_array):
        result = compute_histogram_correlation(solid_red_array, solid_red_array)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_different_solid_colors_have_low_correlation(self, solid_red_array, solid_blue_array):
        result = compute_histogram_correlation(solid_red_array, solid_blue_array)
        assert result < 0.5

    def test_grayscale_images(self):
        gray_a = np.full((64, 64), 128, dtype=np.uint8)
        gray_b = np.full((64, 64), 128, dtype=np.uint8)
        result = compute_histogram_correlation(gray_a, gray_b)
        assert result == pytest.approx(1.0, abs=0.01)


class TestComputePhashSimilarity:
    def test_identical_images_return_one(self, solid_red_image):
        assert compute_phash_similarity(solid_red_image, solid_red_image) == pytest.approx(
            1.0, abs=0.01
        )

    def test_different_images_return_less_than_one(self, noise_image_a, noise_image_b):
        result = compute_phash_similarity(noise_image_a, noise_image_b)
        assert result < 1.0

    def test_returns_value_in_range(self, noise_image_a, noise_image_b):
        result = compute_phash_similarity(noise_image_a, noise_image_b)
        assert 0.0 <= result <= 1.0


class TestComputePairwiseSimilarities:
    def test_diagonal_is_ones(self, noise_array_a, noise_array_b, noise_image_a, noise_image_b):
        matrix = compute_pairwise_similarities(
            [noise_array_a, noise_array_b],
            [noise_image_a, noise_image_b],
        )
        np.testing.assert_array_almost_equal(np.diag(matrix), [1.0, 1.0])

    def test_matrix_is_symmetric(self, noise_array_a, noise_array_b, noise_image_a, noise_image_b):
        matrix = compute_pairwise_similarities(
            [noise_array_a, noise_array_b],
            [noise_image_a, noise_image_b],
        )
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_single_image_returns_1x1(self, noise_array_a, noise_image_a):
        matrix = compute_pairwise_similarities([noise_array_a], [noise_image_a])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0)
