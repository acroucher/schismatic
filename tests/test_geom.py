import numpy as np
import unittest
from schismatic import geom

class geometryTestCase(unittest.TestCase):

    def test_in_polygon(self):
        """in_polygon()"""
        poly = [
            np.array([0., 0.]),
            np.array([1., 0.]),
            np.array([0., 1.]),
        ]
        p = np.array([0.8, 0.9])
        self.assertFalse(geom.in_polygon(p, poly))
        p = np.array([0.4, 0.3])
        self.assertTrue(geom.in_polygon(p, poly))

        poly = [[0, 0], [0, 1], [1, 1]]
        p = [0.5, 0.8]
        self.assertTrue(geom.in_polygon(p, poly))

    def test_in_rectangle(self):
        """in_rectangle()"""
        rect = [
            np.array([5., 4.]),
            np.array([8., 11.]),
        ]
        p = np.array([0.8, 0.9])
        self.assertFalse(geom.in_rectangle(p, rect))
        p = np.array([7., 10.5])
        self.assertTrue(geom.in_rectangle(p, rect))

    def test_rect_intersect(self):
        """rectangles_intersect()"""
        r1 = [
            np.array([5., 4.]),
            np.array([8., 11.]),
        ]
        r2 = [
            np.array([7., 5.]),
            np.array([13., 10.]),
        ]
        self.assertTrue(geom.rectangles_intersect(r1, r2))
        self.assertTrue(geom.rectangles_intersect(r2, r1))
        r2 = [
            np.array([1., 2.]),
            np.array([4.6, 10.]),
        ]
        self.assertFalse(geom.rectangles_intersect(r1, r2))
        self.assertFalse(geom.rectangles_intersect(r2, r1))

    def test_bounds_of_points(self):
        """bounds_of_points()"""
        pts = [
            np.array([-1.5, 2.1]),
            np.array([1., -1.1]),
            np.array([4., 3.])
        ]
        bds = geom.bounds_of_points(pts)
        self.assertTrue(np.allclose(bds[0], np.array([-1.5, -1.1])))
        self.assertTrue(np.allclose(bds[1], np.array([4., 3.])))

    def test_polygon_centroid(self):
        """polygon_centroid()"""
        poly = [
            np.array([1., 2.]),
            np.array([3., 2.]),
            np.array([3., 4.]),
            np.array([1., 4.]),
        ]
        c = geom.polygon_centroid(poly)
        self.assertTrue(np.allclose(c, np.array([2, 3])))
        poly = [
            np.array([0., 0.]),
            np.array([15., 0.]),
            np.array([15., 5.]),
            np.array([10., 5.]),
            np.array([10., 10.]),
            np.array([0., 10.])
        ]
        c = geom.polygon_centroid(poly)
        self.assertTrue(np.allclose(c, np.array([6.5, 4.5])))

if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(geometryTestCase)
    unittest.TextTestRunner(verbosity = 1).run(suite)
