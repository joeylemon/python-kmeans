import unittest
from skimage import io
import numpy as np
import kmeans


class TestReduceImage(unittest.TestCase):
    def test_reduce_image(self):
        """ Ensure that images are reduced to the given amount of colors. """
        img = io.imread("images/smokey.jpeg")

        for K in [1, 2, 3, 4]:
            reduced_img, _, _ = kmeans.reduce_image(img, n_colors=K)

            # get unique rows using axis=0
            pixels = np.unique(reduced_img.reshape((-1, 3)), axis=0)

            self.assertEqual(len(pixels), K)


if __name__ == "__main__":
    unittest.main()
