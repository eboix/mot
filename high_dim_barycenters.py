from MOT_algs import MOTProblem
from functools import partial
import itertools
import numpy as np
from collections import Counter
import time
import networkx as nx
import itertools
from disjoint_set import DisjointSet
import random
import os
import pickle

## Example code for MOT cost associated to high-dimensional barycenters
## We implement the naive solver since there is no polynomial-time algorithm in high dimension
## See https://github.com/eboix/high_precision_barycenters for polynomial-time implementation in 2D

class BarycenterProblem(MOTProblem):
    def __init__(self, pts, problemname='Barycenter_Problem'):
        # pts is a length-k list, where each element is a n_i x d list of points
        self.pts = pts
        self.k = len(pts)

        mus = []
        for i in range(self.k):
            ni = pts[i].shape[0]
            mus.append(np.ones(ni) / ni)
        super(BarycenterProblem, self).__init__(mus,problemname)

    def get_tuple_cost(self, tup):
        """
        Cost function. Takes in a k-tuple and returns the corresponding cost.
        Used in _get_explicit_cost_tensor.
        """
        cst = 0
        for i in range(len(tup)):
            xi = self.pts[i][tup[i],:]
            for j in range(i+1,len(tup)):
                xj = self.pts[j][tup[j],:]
                cst += np.sum((xi - xj)**2)
        return cst

    def get_cutting_plane(self, dualwts):
        """
        Cutting plane function for column generation. Takes in the dual
        variables, and returns a list of violating tuples, if there are any
        """
        raise NotImplementedError()

    def get_cost_range(self):
        """
        Range of costs. Needed to run multiplicative weights.
        """
        raise NotImplementedError()


    def marginalize(self, eta, p, i):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_i(P).

        Needed to run fast (non-naive) implementation of Sinkhorn.
        """
        raise NotImplementedError()

import matplotlib.pyplot as plt

def main():

    data_file = 'high_dim_barycenters_data.pkl'
    if not os.path.exists(data_file):
        pickle.dump({}, open(data_file, 'wb'))
    run_data = pickle.load(open(data_file, 'rb'))

    ##########################################################################
    # A) Generate problem with points
    np.random.seed(7)

    ns = [7, 6, 8, 12]
    d = 4
    pts = []
    for n in ns:
        pts.append(np.random.randn(n,d))

    prob=BarycenterProblem(pts)

    ##########################################################################
    # B) Solve

    naive_sinkhorn_eta_list = [1,5,10] # different regularization parameters (taking them too high can give floating point precision errors)
    do_naive_exact = True


    for eta in naive_sinkhorn_eta_list:
        starttime = time.time()
        p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic', naive=True)
        endtime = time.time()
        time_naive_sinkhorn = endtime-starttime

        # Compute exact cost of the solution found by Sinkhorn
        obj_naive_sinkhorn = prob.compute_rounded_sinkhorn_solution_cost_naive(eta, p, rankone)

        # Compute tensor containing exact solution found by Sinkhorn
        sol_naive_sinkhorn = prob.get_rounded_sinkhorn_solution_probability_tensor(eta, p, rankone)
        print('Sinkhorn solution is probability tensor of shape', sol_naive_sinkhorn.shape)

        prob._delete_explicit_cost_tensor() # Do some garbage collection, since naive sinkhorn is very expensive.
        print('Total time sinkhorn: naive',time_naive_sinkhorn)
        print('Objective sinkhorn naive:',obj_naive_sinkhorn)
        run_data[('naive_sinkhorn' + str(eta))] = (obj_naive_sinkhorn, time_naive_sinkhorn)

    if do_naive_exact:
        """ Naive exact solver """
        starttime = time.time()
        obj_naive, sol_naive = prob.solve_naive()

        print('Exact solution (list of probability + tuple):', sol_naive)

        endtime = time.time()
        time_naive = endtime-starttime
        print('Total time naive:',time_naive)
        print('Objective:',obj_naive)
        run_data[('naive_exact')] = (obj_naive, time_naive)

    print()
    print('run_data = ',run_data)
    pickle.dump(run_data, open(data_file, 'wb'))


if __name__ == '__main__':
    main()
