"""Copyright 2023 University of Auckland.

This file is part of schismatic.

schismatic is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

schismatic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with schismatic.  If not, see <http://www.gnu.org/licenses/>."""

import numpy as np

def polygon_centroid(polygon):
    """Calculates the centroid of an arbitrary polygon (a tuple, list or
    array of points, each one a tuple, list or array of length 2).
    """
    c, area = np.zeros(2), 0.
    poly = [np.array(v, dtype = float) for v in polygon]
    n = len(poly)
    shift = poly[0]
    poly = [p - shift for p in poly] # shift to reduce roundoff for large coordinates
    if n < 3: return sum(poly) / n + shift
    else:
        for j, p1 in enumerate(poly):
            p2 = poly[(j+1) % n]
            t = p1[0] * p2[1] - p2[0] * p1[1]
            area += t
            c += (p1 + p2) * t
        area *= 0.5
        return c / (6. * area) + shift

def in_polygon(pos, polygon):
    """Tests if the point *pos* (a tuple, list or array of length 2) a
    lies within a given polygon (a tuple or list of points, each
    itself a tuple, list or array of length 2).

    """
    if len(polygon) == 2:
        return in_rectangle(pos, polygon)
    else:
        tolerance = 1.e-6
        numcrossings = 0
        pos = np.array(pos)
        poly = [np.array(v, dtype = float) for v in polygon]
        ref = poly[0]
        v = pos - ref
        for i in range(len(poly)):
            p1 = poly[i] - ref
            i2 = (i+1) % len(poly)
            p2 = poly[i2] - ref
            if p1[1] <= v[1] < p2[1] or p2[1] <= v[1] < p1[1]:
                d = p2 - p1
                if abs(d[1]) > tolerance:
                    x = p1[0] + (v[1] - p1[1]) * d[0] / d[1]
                    if v[0] < x: numcrossings += 1
        return (numcrossings % 2) == 1

def in_rectangle(pos, rect):
    """Tests if the point *pos* lies in an axis-aligned rectangle, defined
    as a two-element tuple or list of points [bottom left, top right],
    each itself a tuple, list or array of length 2.

    """
    return all([rect[0][i] <= pos[i] <= rect[1][i] for i in range(2)])

def sub_rectangles(rect):
    """Returns the sub-rectangles formed by subdividing the given rectangle evenly in four."""
    centre = 0.5 * (rect[0] + rect[1])
    r0 = [rect[0],centre]
    r1 = [np.array([centre[0], rect[0][1]]), np.array([rect[1][0], centre[1]])]
    r2 = [np.array([rect[0][0], centre[1]]), np.array([centre[0], rect[1][1]])]
    r3 = [centre,rect[1]]
    return [r0,r1,r2,r3]

def rectangles_intersect(rect1, rect2):
    """Returns True if two rectangles intersect."""
    return all([(rect1[1][i] >= rect2[0][i]) and (rect2[1][i] >= rect1[0][i])
                for i in range(2)])

def bounds_of_points(points):
    """Returns bounding box around the specified 2D points."""
    bottomleft = np.array([min([pos[i] for pos in points]) for i in range(2)])
    topright = np.array([max([pos[i] for pos in points]) for i in range(2)])
    return [bottomleft, topright]
