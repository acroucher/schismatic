"""SCHISM gr3 grids."""

import os
from os.path import splitext
import xml.dom.minidom
from schismatic import geom, quadtree
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from pyproj import Transformer
import netCDF4
from vtk import vtkUnstructuredGrid, vtkPoints, vtkIdList, vtkFloatArray
from vtk import vtkXMLUnstructuredGridWriter

class node(object):
    """Grid node with 2D position and value"""

    def __init__(self, index, pos, value):
        self.index = index
        self.pos = np.array(pos)
        self.value = value
        self.element = set()

    def __repr__(self):
        return str(self.index) + ':' + str(list(self.pos))

    def get_num_elements(self): return len(self.element)
    num_elements = property(get_num_elements)

    def inside(self, polygon):
        """Returns True if the node is inside the polygon."""
        return geom.in_polygon(self.pos, polygon)

    def _get_element_size(self):
        dx = np.array([e.size for e in self.element])
        return np.average(dx)
    element_size = property(_get_element_size)

class element(object):
    """Grid element"""

    def __init__(self, index, node):
        self.index = index
        self.node = node
        self._centroid = None
        self._size = None

    def __repr__(self):
        return str(self.index)

    def _get_num_nodes(self): return len(self.node)
    num_nodes = property(_get_num_nodes)

    def _get_node_indices(self):
        # Returns (zero-based) indices of nodes in the element. 
        return np.array([n.index - 1 for n in self.node], dtype = int)
    node_indices = property(_get_node_indices)

    def _get_centroid(self):
        if self._centroid is None:
            poly = [n.pos for n in self.node]
            self._centroid = geom.polygon_centroid(poly)
        return self._centroid
    centroid = property(_get_centroid)

    def get_neighbour(self):
        """Returns set of other elements sharing a node with the element."""
        nbrs = set([])
        for node in self.node: nbrs = nbrs | node.element
        nbrs.remove(self)
        return nbrs
    neighbour = property(get_neighbour)

    def get_bounds(self):
        return geom.bounds_of_points([n.pos for n in self.node])
    bounds = property(get_bounds)

    def basis(self, xi):
        """Returns linear finite element basis functions at local coordinate
        xi."""
        if self.num_nodes == 3:
            return np.array([xi[0], xi[1], 1. - xi[0] - xi[1]])
        elif self.num_nodes == 4:
            a0, a1, b0, b1 = 1. - xi[0], 1. + xi[0], 1. - xi[1], 1. + xi[1]
            return 0.25 * np.array([a0 * b0, a1 * b0, a1 * b1, a0 * b1])

    def basis_derivatives(self, xi):
        if self.num_nodes == 3:
            return np.array([[1., 0.], [0., 1.], [-1., -1.]])
        elif self.num_nodes == 4:
            a0, a1, b0, b1 = 1. - xi[0], 1. + xi[0], 1. - xi[1], 1. + xi[1]
            return 0.25 * np.array([[-b0, -a0], [b0, -a1], [b1, a1], [-b1, a0]])

    def interpolate(self, vals, xi, psi = None):
        """Interpolates array of nodal values (scalars or arrays) at local
        coordinate xi.  Basis functions at xi can be optionally
        supplied, if available, otherwise they will be calculated.

        """
        if psi is None: psi = self.basis(xi)
        return np.sum(psi * vals, axis = -1)

    def global_pos(self, xi, psi = None):
        """Returns global position of local element coordinates xi.  Basis
        functions at xi can be optionally supplied, if available,
        otherwise they will be calculated.
        """
        if psi is None: psi = self.basis(xi)
        xnode, ynode = [], []
        for n in self.node:
            xnode.append(n.pos[0])
            ynode.append(n.pos[1])
        x = self.interpolate(xnode, xi, psi)
        y = self.interpolate(ynode, xi, psi)
        return np.array([x, y])

    def Jacobian(self, xi, dpsi = None):
        """Returns Jacobian matrix at specified local element coordinates xi.
        Derivatives of basis functions at xi can be optionally
        supplied, if available, otherwise they will be calculated.
        """
        if (dpsi is None): dpsi = self.basis_derivatives(xi)
        J = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                for k, node_k in enumerate(self.node):
                    J[i, j] += dpsi[k, j] * node_k.pos[i]
        return J
    
    def contains_local(self, xi):
        """Returns true if the element contains the point with local
        coordinates xi."""
        if self.num_nodes == 3:
            return all([xii >= 0.0 for xii in xi]) and np.sum(xi) <= 1.0
        elif self.num_nodes == 4:
            return all([abs(xii) <= 1.0 for xii in xi])
        else: return False

    def local_pos(self, x):
        """Returns local position of global position x, if it is in the
        element (None otherwise)."""
        tolerance, max_iterations = 1.e-8, 15
        x = np.array(x)
        if self.num_nodes == 3: xi = np.array([1/3., 1/3.])
        else: xi = np.zeros(2)
        found = False
        for n in range(max_iterations):
            dx = self.global_pos(xi) - x
            if np.linalg.norm(dx) <= tolerance:
                found = True
                break
            else:
                J = self.Jacobian(xi)
                try:
                    xi -= np.linalg.solve(J, dx)
                except np.linalg.LinAlgError: break
        if not found: return None
        else:
            if self.contains_local(xi): return xi
            else: return None
    
    def contains(self, pos):
        return self.local_pos(pos) is not None

    def inside(self, polygon):
        """Returns True if the element centroid is inside the polygon."""
        return geom.in_polygon(self.centroid, polygon)

    def _get_size(self):
        """Returns approximate horizontal size of element, from length of
        longest side."""
        if self._size is None:
            side_lengths = np.array([np.linalg.norm(n.pos - self.node[i - 1].pos)
                                     for i,n in enumerate(self.node)])
            self._size = np.max(side_lengths)
        return self._size
    size = property(_get_size)

class boundary(object):
    """Grid boundary"""

    def __init__(self, index, node, open = False, island = False):
        self.index = index
        self.node = node
        self.open = open
        self.island = island

    def __repr__(self):
        return str(self.index)

    def _get_num_nodes(self): return len(self.node)
    num_nodes = property(_get_num_nodes)
    
class grid(object):
    """Grid object"""

    def __init__(self, filename = None):
        self.header = ''
        self.node = []
        self.element = []
        self.boundary = []
        self._qtree = None
        self._kdtree = None
        if filename is not None: self.read(filename)
        
    def __repr__(self):
        return '%s: %d nodes, %d elements' % \
            (self.header, self.num_nodes, self.num_elements)

    def _get_num_nodes(self):
        return len(self.node)
    num_nodes = property(_get_num_nodes)

    def _get_num_elements(self):
        return len(self.element)
    num_elements = property(_get_num_elements)
    
    def _get_num_boundaries(self):
        return len(self.boundary)
    num_boundaries = property(_get_num_boundaries)

    def _get_num_open_boundaries(self):
        return len([bdy for bdy in self.boundary if bdy.open])
    num_open_boundaries = property(_get_num_open_boundaries)

    def _get_num_land_boundaries(self):
        return len([bdy for bdy in self.boundary if not bdy.open])
    num_land_boundaries = property(_get_num_land_boundaries)

    def _get_values(self):
        return np.array([n.value for n in self.node])
    def _set_values(self, vals):
        for i,n in enumerate(self.node): n.value = vals[i]
    values = property(_get_values, _set_values)

    def _get_value_range(self):
        v = self.values
        return np.min(v), np.max(v)
    value_range = property(_get_value_range)

    def _get_pos(self):
        return np.array([n.pos for n in self.node])
    def _set_pos(self, pos):
        for i,n in enumerate(self.node): n.pos = pos[i,:]
    pos = property(_get_pos, _set_pos)

    def add_node(self, n):
        self.node.append(n)

    def add_element(self, elt):
        self.element.append(elt)
        for n in elt.node: n.element.add(elt)

    def add_boundary(self, bdy):
        self.boundary.append(bdy)

    def open_land_boundary(self, nodes):
        # converts a section of land boundary, defined by the
        # specified nodes, into an open boundary (e.g. for a river
        # input).
        land_bdy = None
        for bdy in self.boundary:
            if not bdy.open:
                if all([n in bdy.node for n in nodes]):
                    land_bdy = bdy
                    break                        
        if land_bdy is None:
            raise Exception('Nodes not found in any land boundary.')
        else:
            indices = np.array([land_bdy.node.index(n) for n in nodes], dtype = int)
            indices.sort()
            nodes = [land_bdy.node[i] for i in indices]
            i0 = indices[0]
            i1 = i0 + len(nodes) - 1
            if indices.tolist() == list(range(i0, i1 + 1, 1)):
                if i0 == 0:
                    land_bdy.node = land_bdy.node[i1:]
                elif i1 == land_bdy.num_nodes:
                    land_bdy.node = land_bdy.node[:i0 + 1]
                else:
                    after_nodes = land_bdy.node[i1:]
                    land_bdy.node = land_bdy.node[:i0 + 1]
                    after_bdy = boundary(self.num_boundaries, after_nodes, open = False)
                    self.add_boundary(after_bdy)
                open_bdy = boundary(self.num_boundaries, nodes, open = True)
                self.add_boundary(open_bdy)
            else:
                raise Exception('Non-adjacent nodes passed to open_land_boundary().')

    def get_bounds(self):
        return geom.bounds_of_points(self.pos)
    bounds = property(get_bounds)

    def compute_quadtree(self):
        self._qtree = quadtree.quadtree(self.bounds, self.element)

    def _get_qtree(self):
        if self._qtree is None:
            self.compute_quadtree()
        return self._qtree
    qtree = property(_get_qtree)

    def compute_kdtree(self):
        self._kdtree = cKDTree(self.pos)

    def _get_kdtree(self):
        if self._kdtree is None:
            self.compute_kdtree()
        return self._kdtree
    kdtree = property(_get_kdtree)

    def read(self, filename):
        """Reads grid from *.gr3 file"""

        with open(filename, 'r') as f:
            self.header = f.readline().strip()

            items = f.readline().split()
            num_elements, num_nodes = [int(item) for item in items[:2]]

            for i in range(num_nodes):
                items = f.readline().split()
                pos = [float(item) for item in items[1:3]]
                value = float(items[3])
                n = node(i + 1, pos, value)
                self.add_node(n)

            for i in range(num_elements):
                items = f.readline().split()
                node_nums = [int(item) for item in items[2:]]
                nodes = [self.node[node_num - 1] for node_num in node_nums]
                elt = element(i + 1, nodes)
                self.add_element(elt)

            line = f.readline().strip()
            if line:

                items = line.split()
                num_open_boundaries = int(items[0])
                f.readline()
                ibdy = 1
                for i in range(num_open_boundaries):
                    items = f.readline().split()
                    num_nodes = int(items[0])
                    nodes = []
                    for j in range(num_nodes):
                        node_num = int(f.readline())
                        n = self.node[node_num - 1]
                        nodes.append(n)
                    bdy = boundary(ibdy, nodes, open = True)
                    self.add_boundary(bdy)
                    ibdy += 1

                items = f.readline().split()
                num_land_boundaries = int(items[0])
                f.readline()
                for i in range(num_land_boundaries):
                    items = f.readline().split()
                    num_nodes, island_flag = int(items[0]), int(items[1])
                    island = island_flag == 1
                    nodes = []
                    for j in range(num_nodes):
                        node_num = int(f.readline())
                        n = self.node[node_num - 1]
                        nodes.append(n)
                    bdy = boundary(ibdy, nodes, open = False, island = island)
                    self.add_boundary(bdy)
                    ibdy += 1

    def write(self, filename, boundary = True):
        """Writes grid to gr3 file"""

        with open(filename, 'w') as f:
            f.write('%s\n' % self.header.strip())

            f.write('%d %d\n' % (self.num_elements, self.num_nodes))

            for i, n in enumerate(self.node):
                f.write('%d ' % n.index)
                for x in n.pos: f.write('%22.16e ' % x)
                f.write('%22.16e\n' % n.value)                
        
            for i, e in enumerate(self.element):
                f.write('%d %d ' % (e.index, e.num_nodes))
                for n in e.node: f.write('%d ' % n.index)
                f.write('\n')

            if boundary and self.boundary:
                open_boundaries, land_boundaries = [], []
                num_open_nodes, num_land_nodes = 0, 0
                for bdy in self.boundary:
                    if bdy.open:
                        open_boundaries.append(bdy)
                        num_open_nodes += bdy.num_nodes
                    else:
                        land_boundaries.append(bdy)
                        num_land_nodes += bdy.num_nodes
                f.write('%d ! total number of open boundaries\n' % len(open_boundaries))
                f.write('%d ! total number of open boundary nodes\n' % num_open_nodes)
                for i, bdy in enumerate(open_boundaries):
                    f.write('%d ! number of nodes for open_boundary_%d\n' % (bdy.num_nodes, i))
                    for n in bdy.node: f.write('%d\n' % n.index)
                f.write('%d ! total number of land boundaries\n' % len(land_boundaries))
                f.write('%d ! total number of land boundary nodes\n' % num_land_nodes)
                for i, bdy in enumerate(land_boundaries):
                    island_flag = 1 if bdy.island else 0
                    f.write('%d %d ! boundary %d\n' % (bdy.num_nodes, island_flag, i))
                    for n in bdy.node: f.write('%d\n' % n.index)

    def transform_crs(self, old, new):
        """Transforms CRS from old to new."""
        prefix = 'EPSG:'
        if not old.startswith(prefix): old = prefix + old
        if not new.startswith(prefix): new = prefix + new
        transformer = Transformer.from_crs(old, new, always_xy = True)
        for n in self.node:
            n.pos = np.array(transformer.transform(n.pos[0], n.pos[1]))
        for e in self.element: e._centroid = None
        self._qtree = None

    def plot(self, **kwargs):
        """Creates Matplotlib plot of grid"""

        import matplotlib.pyplot as plt
        import matplotlib.collections as collections

        if 'axes' in kwargs:
            ax = kwargs['axes']
            fig = ax.get_figure()
        else: fig, ax = plt.subplots()

        labels = kwargs.get('label', None)
        label_fmt = kwargs.get('label_format', '%g')
        label_colour = kwargs.get('label_colour', 'black')
        verts = []
        for e in self.element:
            poslist = [tuple([p for p in n.pos]) for n in e.node]
            verts.append(tuple(poslist))
            if labels == 'element':
                elt_label = label_fmt % e.index
            else: elt_label = None
            if elt_label:
                ax.text(e.centroid[0], e.centroid[1], elt_label,
                        clip_on = True, horizontalalignment = 'center',
                        color = label_colour)

        linewidth = kwargs.get('linewidth', 0.1)
        linecolour = kwargs.get('linecolour', 'black')
        colourmap = kwargs.get('colourmap', None)

        contours = kwargs.get('contours', True)
        if contours:
            pos = self.pos
            v = kwargs.get('values', self.values)
            mask = np.isfinite(v)
            maski = dict([(x, i) for i, x in enumerate(np.where(mask)[0])])
            tri = []
            for e in self.element:
                if e.num_nodes == 3:
                    en = e.node_indices
                    if np.all(mask[en]):
                        tri.append([maski[i] for i in en])
            if 'levels' in kwargs:
                levels = kwargs.get('levels')
                tc = ax.tricontourf(pos[mask, 0], pos[mask, 1], tri, v[mask], levels)
            else: # default contour levels
                tc = ax.tricontourf(pos[mask, 0], pos[mask, 1], tri, v[mask])
            values_label = kwargs.get('values_label', None)
            fig.colorbar(tc, label = values_label)

        elements = kwargs.get('elements', False)
        if elements:
            polys = collections.PolyCollection(verts,
                                               linewidth = linewidth,
                                               facecolors = [],
                                               edgecolors = linecolour,
                                               cmap = colourmap)
            ax.add_collection(polys)

        bdy_colour = kwargs.get('boundary_colour', {'open': 'red', 'land': 'black'})
        for bdy in self.boundary:
            bdy_type = 'open' if bdy.open else 'land'
            colour = bdy_colour[bdy_type]
            x = [p.pos[0] for p in bdy.node]
            y = [p.pos[1] for p in bdy.node]
            ax.plot(x, y, '-', color = colour)

        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))

        ax.set_aspect(kwargs.get('aspect', 'equal'))
        ax.autoscale_view()

        if 'axes' not in kwargs: plt.show()

    def _get_meshio_points_cells(self):

        points = []
        cells = {'triangle': [], 'quad': []}
        cell_type = {3: 'triangle', 4: 'quad'}

        for n in self.node:
            pos = np.array([n.pos[0], n.pos[1], -n.value])
            points.append(pos)

        for e in self.element:
            elt = [n.index - 1 for n in e.node]
            cells[cell_type[len(elt)]].append(elt)

        points = np.array(points)
        cells = dict([(k, np.array(v)) for k, v in cells.items() if v])
        return points, cells

    meshio_points_cells = property(_get_meshio_points_cells)
        
    def export(self, filename, fmt = None):
        """Exports grid using meshio, to file with the specified name. If
        the format is not specified via the fmt parameter, it is determined
        from the filename extension."""
        import meshio
        points, cells = self.meshio_points_cells
        meshio.write_points_cells(filename, points, cells, file_format = fmt)

    def export_2dm(self, filename):
        """Exports grid to 2dm format."""

        with open(filename, 'w') as f:
            f.write('MESH2D\n')
            for e in self.element:
                etype = 'E3T' if e.num_nodes == 3 else 'E4Q'
                f.write('%s %d ' % (etype, e.index))
                for n in e.node:
                    f.write('%d ' % n.index)
                f.write('\n')
            for n in self.node:
                f.write('ND %d %22.16e %22.16e %22.16e\n' % (n.index,
                                                           n.pos[0], n.pos[1],
                                                           -n.value))
        
    def get_vtk_grid(self, flow, time_index):
        """Returns a vtkUnstructuredGrid object representing the 2D
        depth-averaged flow at the specified time index (for visualisation
        with VTK).
        """

        vtkgrid = vtkUnstructuredGrid()
        pts = vtkPoints()
        pts.SetNumberOfPoints(self.num_nodes)
        elev = flow['elevation'][time_index, :]
        for n in self.node:
            i = n.index - 1
            pos = np.array([n.pos[0], n.pos[1], elev[i]])
            pts.SetPoint(i, pos)
        vtkgrid.SetPoints(pts)

        VTK_TRIANGLE, VTK_QUAD = 5, 9
        celltype = {3: VTK_TRIANGLE, 4: VTK_QUAD}
        for elt in self.element:
            ids = vtkIdList()
            for n in elt.node:
                ids.InsertNextId(n.index - 1)
            vtkgrid.InsertNextCell(celltype[elt.num_nodes], ids)

        array_names = ['velocity','elevation']
        num_components = [3, 1]
        arrays = {}
        for name, nc in zip(array_names, num_components):
            arrays[name] = vtkFloatArray()
            arrays[name].SetName(name)
            arrays[name].SetNumberOfComponents(nc)
            if nc == 1: arrays[name].SetNumberOfValues(self.num_nodes)
            else: arrays[name].SetNumberOfTuples(self.num_nodes)

        vx = flow['depthAverageVelX'][time_index, :]
        vy = flow['depthAverageVelY'][time_index, :]
        for n in self.node:
            i = n.index - 1
            arrays['velocity'].SetTuple3(i, vx[i], vy[i], 0.0)
            arrays['elevation'].SetValue(i, elev[i])

        sortedkeys = list(arrays.keys())
        sortedkeys.sort()
        for key in sortedkeys:
            vtkgrid.GetPointData().AddArray(arrays[key])

        return vtkgrid

    def write_2D_flow_vtk(self, stacks = [1]):
        """Writes *.pvd and *.vtu files representing depth-averaged flows from
        out2d_1.nc file in the outputs directory."""

        os.chdir('outputs')

        writer = vtkXMLUnstructuredGridWriter()

        pvd = xml.dom.minidom.Document()
        vtkfile = pvd.createElement('VTKFile')
        vtkfile.setAttribute('type', 'Collection')
        pvd.appendChild(vtkfile)
        collection = pvd.createElement('Collection')

        i = 0
        for stack in stacks:
            filename = 'out2d_%d.nc' % stack
            base, ext = splitext(filename)
            flow = netCDF4.Dataset(filename)
            for istack, t in enumerate(flow['time']):
                vtugrid = self.get_vtk_grid(flow, istack)
                filename_time = base + '_' + str(i) + '.vtu'
                writer.SetFileName(filename_time)
                if hasattr(writer, 'SetInput'): writer.SetInput(vtugrid)
                elif hasattr(writer, 'SetInputData'): writer.SetInputData(vtugrid)
                writer.Write()
                dataset = pvd.createElement('DataSet')
                dataset.setAttribute('timestep', str(t))
                dataset.setAttribute('file', filename_time)
                collection.appendChild(dataset)
                i += 1
            flow.close()

        vtkfile.appendChild(collection)
        pvdfile = open('out2d.pvd', 'w')
        pvdfile.write(pvd.toprettyxml())
        pvdfile.close()

    def find_nodes(self, polygon):
        """Returns nodes in specified polygon."""
        return [n for n in self.node if n.inside(polygon)]

    def find_boundary_nodes(self, polygon):
        """Returns boundary nodes in specified polygon."""
        nodes = []
        for bdy in self.boundary:
            nodes += [n for n in bdy.node if n.inside(polygon)]
        return nodes

    def find_element(self, pos):
        """Returns element containing position pos, or None."""
        return self.qtree.search(pos)

    def find_elements(self, polygon):
        """Returns elements in specified polygon."""
        return [e for e in self.element if e.inside(polygon)]

    def nodes_in_elements(self, elements):
        """Returns list of nodes in specified elements."""
        nodes = set()
        for e in elements: nodes = nodes | set(e.node)
        return list(nodes)

    def find_node(self, pos):
        """Finds node nearest to specified point."""
        r, i = self.kdtree.query(pos)
        return self.node[i]

    def fit(self, data, smooth = 0., nodes = None):
        """Fits nodel values to scattered data using least-squares
        finite-element fitting with Sobolev smoothing."""

        if nodes is None:
            nodes = self.node
            elements = self.element
        else:
            elements = set()
            for n in nodes: elements = elements | set(n.element)
            elements = list(elements)

        if len(nodes) > 0:

            all_nodes = set(nodes)
            for e in elements: all_nodes = all_nodes | set(e.node)
            bdy_nodes = all_nodes - set(nodes)
            all_nodes = list(all_nodes)

            node_index = dict([(node.index, i) for i, node in enumerate(all_nodes)])
            num_nodes = len(all_nodes)
            bounds = geom.bounds_of_points([n.pos for n in all_nodes])
            qt = quadtree.quadtree(bounds, elements)

            A = sparse.lil_matrix((num_nodes, num_nodes))
            b = np.zeros(num_nodes)
            for idata, d in enumerate(data):
                pos, val = d[0:2], d[2]
                if geom.in_rectangle(pos, bounds):
                    e = qt.search(pos)
                    if e:
                        xi = e.local_pos(pos)
                        psi = e.basis(xi)
                        for i, nodei in enumerate(e.node):
                            I = node_index[nodei.index]
                            for j, nodej in enumerate(e.node):
                                J = node_index[nodej.index]
                                A[I, J] += psi[i] * psi[j]
                            b[I] += psi[i] * val

            smooth = {3: 0.5 * smooth * np.array([[1., 0., -1.],
                                                  [0., 1., -1.],
                                                  [-1., -1., 2.]]),
                      4: smooth / 6. * np.array([[4., -1., -2., -1.],
                                                 [-1., 4., -1., -2.],
                                                 [-2., -1., 4., -1.],
                                                 [-1., -2., -1., 4.]])}
            for e in elements:
                for i, nodei in enumerate(e.node):
                    I = node_index[nodei.index]
                    for j, nodej in enumerate(e.node):
                        J = node_index[nodej.index]
                        A[I, J] += smooth[e.num_nodes][i, j]

            for n in bdy_nodes:
                I = node_index[n.index]
                A[I,:] = 0.
                A[I,I] = 1.
                b[I] = n.value

            A = A.tocsr()
            z = sparse.linalg.spsolve(A, b)
            for i, n in enumerate(all_nodes): n.value = z[i]

    def pos_value(self, pos, val = None):
        """Returns interpolated value at given position (or None if pos is
        outside the grid)."""
        v = self.values if val is None else val
        elt = self.find_element(pos)
        if elt:
            xi = elt.local_pos(pos)
            vals = v[elt.node_indices]
            return elt.interpolate(vals, xi)
        else:
            return None

    def cfl(self, timestep = 200., min_depth = 0.1):
        """Return array of nodal estimates of CFL number."""
        g = 9.8
        cfl = []
        for n in self.node:
            if n.value < min_depth:
                u = 1.
                root_gh = 0.
            else:
                u = 0.
                root_gh = np.sqrt(g * n.value)
            dx = n.element_size
            cr = (u + root_gh) * timestep / dx
            cfl.append(cr)
        return np.array(cfl)
