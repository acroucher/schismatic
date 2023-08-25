"""Copyright 2023 University of Auckland.

This file is part of schismatic.

schismatic is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

schismatic is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with schismatic.  If not, see <http://www.gnu.org/licenses/>."""

from schismatic import geom
from copy import copy

class quadtree(object):
    """Quadtree for spatial searching in 2D grids."""

    def __init__(self, bounds, elements, parent = None):
        self.parent = parent
        self.bounds = bounds
        self.elements = elements
        self.child = []
        if self.parent:
            self.generation = self.parent.generation + 1
            self.all_elements = self.parent.all_elements
        else:
            self.generation = 0
            self.all_elements = set(elements)
        if self.num_elements > 1:
            rects = geom.sub_rectangles(self.bounds)
            rect_elements = [[], [], [], []]
            for elt in self.elements:
                for irect,rect in enumerate(rects):
                    if geom.in_rectangle(elt.centroid, rect):
                        rect_elements[irect].append(elt)
                        break
            for rect, elts in zip(rects, rect_elements):
                if len(elts) > 0:
                    self.child.append(quadtree(rect,elts, self))

    def __repr__(self): return self.bounds.__repr__()

    def get_num_elements(self): return len(self.elements)
    num_elements = property(get_num_elements)

    def get_num_children(self): return len(self.child)
    num_children=property(get_num_children)

    def search_wave(self, pos):
        todo = copy(self.elements)
        done = []
        while len(todo) > 0:
            elt = todo.pop(0)
            if elt.contains(pos): return elt
            done.append(elt)
            for nbr in elt.neighbour & self.all_elements:
                if geom.rectangles_intersect(nbr.bounds, self.bounds) \
                   and not ((nbr in done) or (nbr in todo)):
                    todo.append(nbr)
        return None

    def search(self, pos):
        leaf = self.leaf(pos)
        if leaf: return leaf.search_wave(pos)
        else: return None

    def leaf(self, pos):
        if geom.in_rectangle(pos, self.bounds):
            for child in self.child:
                childleaf = child.leaf(pos)
                if childleaf: return childleaf
            return self
        else: return None
