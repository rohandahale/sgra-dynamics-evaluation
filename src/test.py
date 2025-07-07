import unittest
import numpy as np
from skimage.metrics import structural_similarity as ssim

def normalized_cross_correlation(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    matrix1_mean_subtracted = matrix1 - np.mean(matrix1)
    matrix2_mean_subtracted = matrix2 - np.mean(matrix2)

    numerator = np.sum(matrix1_mean_subtracted * matrix2_mean_subtracted)
    denominator = np.sqrt(np.sum(matrix1_mean_subtracted ** 2) * np.sum(matrix2_mean_subtracted ** 2))

    ncc = numerator / denominator

    return ncc


class TestNormalizedCrossCorrelation(unittest.TestCase):
    def test_positive_values(self):
        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        result = normalized_cross_correlation(matrix1, matrix2)
        self.assertAlmostEqual(result, -1.0, places=5)

    def test_negative_values(self):
        matrix1 = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
        matrix2 = np.array([[-9, 8, -7], [6, -5, 4], [-3, 2, -1]])
        result = normalized_cross_correlation(matrix1, matrix2)
        # Check if the result is within a reasonable range
        self.assertTrue(-1.0 <= result <= 1.0)

    def test_different_shapes(self):
        matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
        matrix2 = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            normalized_cross_correlation(matrix1, matrix2)

    def test_identical_matrices(self):
        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = normalized_cross_correlation(matrix1, matrix2)
        self.assertAlmostEqual(result, 1.0, places=5)
        
    def test_opposite_matrices(self):
        matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix2 = -matrix1
        result = normalized_cross_correlation(matrix1, matrix2)
        self.assertAlmostEqual(result, -1.0, places=5)


class TestSSIM(unittest.TestCase):
    def test_identical_images(self):
        image1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        image2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        data_range = image1.max() - image1.min()
        ssim_value, _ = ssim(image1, image2, data_range=data_range, full=True, win_size=3)
        self.assertAlmostEqual(ssim_value, 1.0, places=5)

    def test_different_images(self):
        image1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        image2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float64)
        data_range = image1.max() - image1.min()
        ssim_value, _ = ssim(image1, image2, data_range=data_range, full=True, win_size=3)
        print(f"SSIM value for different images: {ssim_value}")
        self.assertNotEqual(ssim_value, 1.0)

    def test_negative_values(self):
        image1 = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]], dtype=np.float64)
        image2 = np.array([[-9, 8, -7], [6, -5, 4], [-3, 2, -1]], dtype=np.float64)
        data_range = image1.max() - image1.min()
        ssim_value, _ = ssim(image1, image2, data_range=data_range, full=True, win_size=3)
        self.assertTrue(0 <= ssim_value <= 1)

    def test_different_shapes(self):
        image1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        image2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        with self.assertRaises(ValueError):
            ssim(image1, image2, data_range=1, win_size=3)

if __name__ == '__main__':
    unittest.main()