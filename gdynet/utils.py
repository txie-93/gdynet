from __future__ import print_function, division

import os

import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree, cKDTree
import itertools
import heapq

__all__ = ['distance_pbc', 'PeriodicKDTree', 'PeriodicCKDTree']

_TRAJ_KEYS = ['traj_coords', 'lattices', 'nbr_lists', 'nbr_dists']


def split_traj_files(traj_fname, split_dir, n_splits, zip=False):
    """split the trajectories data into specified number of parts"""
    if traj_fname[-4:] != '.npz':
        traj_fname += '.npz'
    traj_basename = os.path.basename(traj_fname).split('.')[0]
    split_dicts = [{} for _ in range(n_splits)]
    with np.load(traj_fname) as data:
        for key, array in data.items():
            if key in _TRAJ_KEYS:
                array_splits = np.array_split(array, n_splits)
            else:
                array_splits = [array for _ in range(n_splits)]
            for i in range(n_splits):
                split_dicts[i][key] = array_splits[i]
                split_dicts[i]['index'] = i
    for d in split_dicts:
        split_fname = os.path.join(split_dir, traj_basename + str(d['index']))
        if zip:
            np.savez_compressed(split_fname, **d)
        else:
            np.savez(split_fname, **d)
    return traj_basename


def combine_split_files(split_dir, out_file, traj_basename, n_splits,
                        zip=False):
    """combine the splited trajectory data into a single file"""
    combined_dict_list = defaultdict(list)
    for i in range(n_splits):
        split_file = os.path.join(split_dir, traj_basename + str(i) + '.npz')
        with np.load(split_file) as data:
            for key, array in data.items():
                if key == 'index':
                    assert i == array
                else:
                    combined_dict_list[key].append(array)
    combined_dict = {}
    for key, array_list in combined_dict_list.items():
        if key in _TRAJ_KEYS:
            combined_dict[key] = np.concatenate(array_list, axis=0)
        else:
            combined_dict[key] = array_list[0]
    if zip:
        np.savez_compressed(out_file, **combined_dict)
    else:
        np.savez(out_file, **combined_dict)


def _gen_relevant_images(x, bounds, distance_upper_bound):
    # Map x onto the canonical unit cell, then produce the relevant
    # mirror images
    real_x = x - np.where(bounds > 0.0,
                          np.floor(x / bounds) * bounds, 0.0)
    m = len(x)

    xs_to_try = [real_x]
    for i in range(m):
        if bounds[i] > 0.0:
            disp = np.zeros(m)
            disp[i] = bounds[i]

            if distance_upper_bound == np.inf:
                xs_to_try = list(
                    itertools.chain.from_iterable(
                        (_ + disp, _, _ - disp) for _ in xs_to_try))
            else:
                extra_xs = []

                # Point near lower boundary, include image on upper side
                if abs(real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ + disp for _ in xs_to_try)

                # Point near upper boundary, include image on lower side
                if abs(bounds[i] - real_x[i]) < distance_upper_bound:
                    extra_xs.extend(_ - disp for _ in xs_to_try)

                xs_to_try.extend(extra_xs)

    return xs_to_try


class PeriodicKDTree(KDTree):
    """
    kd-tree for quick nearest-neighbor lookup with periodic boundaries

    See scipy.spatial.kdtree for details on kd-trees.

    Searches with periodic boundaries are implemented by mapping all
    initial data points to one canonical periodic image, building an
    ordinary kd-tree with these points, then querying this kd-tree multiple
    times, if necessary, with all the relevant periodic images of the
    query point.

    Note that to ensure that no two distinct images of the same point
    appear in the results, it is essential to restrict the maximum
    distance between a query point and a data point to half the smallest
    box dimension.
    """

    def __init__(self, bounds, data, leafsize=10):
        """Construct a kd-tree.

        Parameters
        ----------
        bounds : array_like, shape (k,)
            Size of the periodic box along each spatial dimension.  A
            negative or zero size for dimension k means that space is not
            periodic along k.
        data : array_like, shape (n,k)
            The data points to be indexed. This array is not copied, and
            so modifying this data will result in bogus results.
        leafsize : positive int
            The number of points at which the algorithm switches over to
            brute-force.
        """

        # Map all points to canonical periodic image
        self.bounds = np.array(bounds)
        self.real_data = np.asarray(data)
        wrapped_data = (
            self.real_data - np.where(bounds > 0.0,
                                      (np.floor(self.real_data / bounds)
                                       * bounds), 0.0))

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.bounds > 0, 0.5 * self.bounds, np.inf))

        # Set up underlying kd-tree
        super(PeriodicKDTree, self).__init__(wrapped_data, leafsize)

    # The following name is a kludge to override KDTree's private method
    def _KDTree__query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points
        #
        # A slight complication: k could be "None", which means "return
        # all neighbors within the given distance_upper_bound".

        # Cap distance_upper_bound
        distance_upper_bound = np.min([distance_upper_bound,
                                      self.max_distance_upper_bound])

        # Run queries over all relevant images of x
        hits_list = []
        for real_x in _gen_relevant_images(x, self.bounds,
                                           distance_upper_bound):
            hits_list.append(
                super(PeriodicKDTree, self)._KDTree__query(
                    real_x, k, eps, p, distance_upper_bound))

        # Now merge results
        if k is None:
            return list(heapq.merge(*hits_list))
        elif k > 1:
            return heapq.nsmallest(k, itertools.chain(*hits_list))
        elif k == 1:
            return [min(itertools.chain(*hits_list))]
        else:
            raise ValueError("Invalid k in periodic_kdtree._KDTree__query")

    # The following name is a kludge to override KDTree's private method
    def _KDTree__query_ball_point(self, x, r, p=2., eps=0):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points

        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        for real_x in _gen_relevant_images(x, self.bounds, r):
            results.extend(
                super(PeriodicKDTree, self)._KDTree__query_ball_point(
                    real_x, r, p, eps))
        return results

    def query_ball_tree(self, other, r, p=2., eps=0):
        raise NotImplementedError()

    def query_pairs(self, r, p=2., eps=0):
        raise NotImplementedError()

    def count_neighbors(self, other, r, p=2.):
        raise NotImplementedError()

    def sparse_distance_matrix(self, other, max_distance, p=2.):
        raise NotImplementedError()


class PeriodicCKDTree(cKDTree):
    """
    Cython kd-tree for quick nearest-neighbor lookup with periodic boundaries

    See scipy.spatial.ckdtree for details on kd-trees.

    Searches with periodic boundaries are implemented by mapping all
    initial data points to one canonical periodic image, building an
    ordinary kd-tree with these points, then querying this kd-tree multiple
    times, if necessary, with all the relevant periodic images of the
    query point.

    Note that to ensure that no two distinct images of the same point
    appear in the results, it is essential to restrict the maximum
    distance between a query point and a data point to half the smallest
    box dimension.
    """

    def __init__(self, bounds, data, leafsize=10):
        """Construct a kd-tree.

        Parameters
        ----------
        bounds : array_like, shape (k,)
            Size of the periodic box along each spatial dimension.  A
            negative or zero size for dimension k means that space is not
            periodic along k.
        data : array-like, shape (n,m)
            The n data points of dimension mto be indexed. This array is
            not copied unless this is necessary to produce a contiguous
            array of doubles, and so modifying this data will result in
            bogus results.
        leafsize : positive integer
            The number of points at which the algorithm switches over to
            brute-force.
        """

        # Map all points to canonical periodic image
        self.bounds = np.array(bounds)
        self.real_data = np.asarray(data)
        wrapped_data = (
            self.real_data - np.where(bounds > 0.0,
                                      (np.floor(self.real_data / bounds)
                                       * bounds), 0.0))

        # Calculate maximum distance_upper_bound
        self.max_distance_upper_bound = np.min(
            np.where(self.bounds > 0, 0.5 * self.bounds, np.inf))

        # Set up underlying kd-tree
        super(PeriodicCKDTree, self).__init__(wrapped_data, leafsize)

    # Ideally, KDTree and cKDTree would expose identical query and __query
    # interfaces.  But they don't, and cKDTree.__query is also inaccessible
    # from Python.  We do our best here to cope.
    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points
        #
        # A slight complication: k could be "None", which means "return
        # all neighbors within the given distance_upper_bound".

        # Cap distance_upper_bound
        distance_upper_bound = np.min([distance_upper_bound,
                                      self.max_distance_upper_bound])

        # Run queries over all relevant images of x
        hits_list = []
        for real_x in _gen_relevant_images(x, self.bounds,
                                           distance_upper_bound):
            d, i = super(PeriodicCKDTree, self).query(
                    real_x, k, eps, p, distance_upper_bound)
            if k > 1:
                hits_list.append(list(zip(d, i)))
            else:
                hits_list.append([(d, i)])

        # Now merge results
        if k > 1:
            return heapq.nsmallest(k, itertools.chain(*hits_list))
        elif k == 1:
            return [min(itertools.chain(*hits_list))]
        else:
            raise ValueError("Invalid k in periodic_kdtree._KDTree__query")

    def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """
        Query the kd-tree for nearest neighbors

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.

        """
        x = np.asarray(x)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but "
                             "has shape %s" % (self.m, np.shape(x)))
        if p < 1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape != ():
            if k > 1:
                dd = np.empty(retshape+(k,), dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,), dtype=np.int)
                ii.fill(self.n)
            elif k == 1:
                dd = np.empty(retshape, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape, dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable "
                                 "numbers are integers greater than or equal "
                                 "to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, eps=eps, p=p,
                                    distance_upper_bound=distance_upper_bound)
                if k > 1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k == 1:
                    if len(hits) > 0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, eps=eps, p=p,
                                distance_upper_bound=distance_upper_bound)
            if k == 1:
                if len(hits) > 0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k > 1:
                dd = np.empty(k, dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k, dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable "
                                 "numbers are integers greater than or equal "
                                 "to one, or None")

    # Ideally, KDTree and cKDTree would expose identical __query_ball_point
    # interfaces.  But they don't, and cKDTree.__query_ball_point is also
    # inaccessible from Python.  We do our best here to cope.
    def __query_ball_point(self, x, r, p=2., eps=0):
        # This is the internal query method, which guarantees that x
        # is a single point, not an array of points

        # Cap r
        r = min(r, self.max_distance_upper_bound)

        # Run queries over all relevant images of x
        results = []
        for real_x in _gen_relevant_images(x, self.bounds, r):
            results.extend(super(PeriodicCKDTree, self).query_ball_point(
                real_x, r, p, eps))
        return results

    def query_ball_point(self, x, r, p=2., eps=0):
        """
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may
        save substantial amounts of time by putting them in a
        PeriodicCKDTree and using query_ball_tree.
        """
        x = np.asarray(x).astype(np.float)
        if x.shape[-1] != self.m:
            raise ValueError("Searching for a %d-dimensional point in a "
                             "%d-dimensional KDTree" % (x.shape[-1], self.m))
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p, eps)
            return result

    def query_ball_tree(self, other, r, p=2., eps=0):
        raise NotImplementedError()

    def query_pairs(self, r, p=2., eps=0):
        raise NotImplementedError()

    def count_neighbors(self, other, r, p=2.):
        raise NotImplementedError()

    def sparse_distance_matrix(self, other, max_distance, p=2.):
        raise NotImplementedError()


def distance_pbc(x0, x1, dimensions):
    """
    Distance between atoms in periodic boundary conditions.

    x0: (3, ) numpy array
    x1: (m, 3) numpy array
    dimensions: (3, ) numpy array
    """
    assert x0.shape[0] == 3 and x1.shape[1] == 3
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))


def batch_pdist_pbc(x0, x1, dimensions):
    """
    Calculate pairwise distance between atoms in periodic boundary conditions
    in batch.

    x0: shape (B, N0, 3)
    x1: shape (B, N1, 3)
    dimensions: shape (B, 3)

    returns
    pdist: shape (B, N1, N0)
    """
    assert x0.shape[0] == x1.shape[0]
    assert x0.shape[-1] == x1.shape[-1] == 3
    distances = np.abs(x0[:, np.newaxis] - x1[:, :, np.newaxis])
    dimensions = dimensions[:, np.newaxis, np.newaxis]
    distances = np.where(distances > 0.5 * dimensions, distances - dimensions,
                         distances)
    return np.sqrt((distances ** 2).sum(axis=-1))


def relative_coord_pbc(x0, x1, dimensions):
    """
    The coordinate of atoms in x1 relative to x0 in periodic boundary
    conditions.

    x0: (3, ) numpy array
    x1: (m, 3) numpy array
    dimensions: (3, ) numpy array
    """
    assert x0.shape[0] == 3 and x1.shape[1] == 3
    delta = x1 - x0
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    delta = np.where(delta < -0.5 * dimensions, delta + dimensions, delta)
    return delta
