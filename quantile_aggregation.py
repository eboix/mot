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
import math
import copy

def cumargmax(a):
    # Cumulative max and argmax for 1-dimensional numpy arrays
    m = np.maximum.accumulate(a)
    argm = np.arange(a.shape[0])
    argm[1:] *= m[:-1] < m[1:]
    np.maximum.accumulate(argm, out=argm)
    return m, argm

class QuantileAggregationProblem(MOTProblem):
    def __init__(self, mus, thresh, mode, problemname='Quantile_Aggregation_Problem', log_file=None):

        # mus is a list of arrays
        mus = [np.asarray(x) for x in mus]
        self.mus = mus
        self.k = len(mus)
        self.ns = [len(x) for x in mus]
        self.thresh = thresh
        self.mode = mode
        assert(mode in ['worst_case', 'best_case'])

        super(QuantileAggregationProblem, self).__init__(mus,problemname,log_file)

    def get_tuple_cost(self, tup):

        if self.mode == 'worst_case':
            if sum(tup) >= self.thresh:
                return 0
            else:
                return 1
        elif self.mode == 'best_case':
            if sum(tup) < self.thresh:
                return 0
            else:
                return 1
        else:
            assert(False)

    def get_cutting_plane_best_case(self, dualwts):
        # pick best from each and check
        totsum = sum([np.max(x) for x in dualwts])
        argmaxtuple = tuple(np.argmax(np.asarray(dualwts), 1))
        if totsum > 1 + 1e-8: #  and argmaxtuple not in self.tups_set
            # print('totsum return more than 1',totsum)
            # assert(False)
            return [argmaxtuple]

        max_path = [0] * self.thresh
        max_path_inds = []
        # At iteration t, max_path[i] = \max_{tup : \sum_{j <= t} tup[j] == i} \sum_{j <= t} wt[tup[j]]
        # Therefore:
        # max_path[i][t+1] = max_r(wt[r] + max_path[i-r][t])

        for i in range(self.k):
            new_max_path = []
            new_max_path_ind = []
            currn = len(dualwts[i])
            print(dualwts[i])
            for j in range(self.thresh):
                currmidx = -1
                currm = -math.inf
                for r in range(currn):
                    if r > j:
                        break
                    if dualwts[i][r] + max_path[j-r] > currm:
                        currm = dualwts[i][r] + max_path[j-r]
                        currmidx = r
                new_max_path.append(currm)
                new_max_path_ind.append(currmidx)
            max_path = new_max_path
            max_path_inds.append(new_max_path_ind)
        print(max_path_inds)
        print(max_path)

        tuplist = []
        currj = np.argmax(max_path)
        for i in range(self.k-1,-1,-1):
            currval = max_path_inds[i][currj]
            currj -= currval
            tuplist.append(currval)
        assert(currj == 0)
        tuplist.reverse()
        ret_tup = tuple(tuplist)
        tup_val = sum(dualwts[i][ret_tup[i]] for i in range(self.k))

        if tup_val > 0:
            return [ret_tup]
        else:
            return []


    def get_cutting_plane_worst_case(self, dualwts):
        # pick best from each and check
        totsum = sum([np.max(x) for x in dualwts])
        argmaxtuple = tuple(np.argmax(np.asarray(dualwts), 1))
        if totsum > 1 + 1e-8: #  and argmaxtuple not in self.tups_set
            # print('totsum return more than 1',totsum)
            # assert(False)
            return [argmaxtuple]


        revthresh = sum(self.ns) - self.k - self.thresh + 1
        if revthresh <= 0:
            return []


        max_path = [0] * revthresh
        max_path_inds = []
        # At iteration t, max_path[i] = \max_{tup : \sum_{j <= t} tup[j] == i} \sum_{j <= t} wt[tup[j]]
        # Therefore:

        for i in range(self.k):
            new_max_path = []
            new_max_path_ind = []
            currn = len(dualwts[i])
            print(dualwts[i])
            for j in range(revthresh):
                currmidx = -1
                currm = -math.inf
                for r in range(currn):
                    if r > j:
                        break
                    if dualwts[i][currn-1-r] + max_path[j-r] > currm:
                        currm = dualwts[i][currn-1-r] + max_path[j-r]
                        currmidx = r
                new_max_path.append(currm)
                new_max_path_ind.append(currmidx)
            max_path = new_max_path
            max_path_inds.append(new_max_path_ind)
        print(max_path_inds)
        print(max_path)

        tuplist = []
        currj = np.argmax(max_path)
        for i in range(self.k-1,-1,-1):
            currval = max_path_inds[i][currj]
            currj -= currval
            tuplist.append(currval)
        assert(currj == 0)
        tuplist.reverse()
        ret_tup = tuple([self.ns[i] - 1 - tuplist[i] for i in range(self.k)])
        tup_val = sum(dualwts[i][ret_tup[i]] for i in range(self.k))

        if tup_val > 0:
            return [ret_tup]
        else:
            return []

        # revwts[j] is (dualwts[j][nj-1],...,dualwts[j][0])
        # Q is maximum of \sum_j dualwts[j][nj-tup[j]-1] over tups such that
        # \sum_j tup[j] < revthresh.
        # I.e., max \sum_j dualwts[j][tup[j]] over tups such that
        # \sum_j nj-1-tup[j] < revthresh
        # I.e., \sum_j tup[j] > nj-1-tup[j] - revthresh = thresh - 1
        #
        # at this point, inverted random variables sum >= thresh iff
        # self.ns - self.k - sum(invtup) >= thresh,
        # so if sum(invtup) <= self.ns - self.k - thresh,
        # so if sum(invtup) < self.ns - self.k - thresh + 1.
        #
        # revret_tups = self.get_cutting_plane_best_case(revwts, revthresh)
        # ret_tups = [tuple([self.ns[i] - 1 - tup[i] for i in range(self.k)]) for tup in revret_tups]
        # print(revret_tups)
        # print(ret_tups)
        # for tup in ret_tups:
        #     print(self.get_tuple_cost(tup))
        # return ret_tups


    def get_cutting_plane(self, dualwts):
        # dualwts is a length-k list of length-2 lists
        if self.mode == 'best_case':
            return self.get_cutting_plane_best_case(dualwts)
        elif self.mode == 'worst_case':
            return self.get_cutting_plane_worst_case(dualwts)
        else:
            assert(False), "mode is not set correctly"

    def get_cost_range(self):
        return (0,1)

    def marginalize(self, eta, p, i):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_i(P).
        """

        # TODO vectorize, and also add thresh trick if you want

        assert(i in range(self.k))
        assert(eta > 0)

        V = np.zeros(self.thresh+1)
        V[0] = 1
        for j in range(self.k):
            if j == i:
                continue

            # newV = np.zeros((self.thresh+1,self.ns[j]))
            # # Indexed by thresh, value of n used to get there
            #
            # # This code could be vectorized and perhaps some numerical precision
            # # issues could be avoided with logsumexp, but don't do it right now
            # for l in range(self.thresh+1):
            #     for m in range(self.ns[j]):
            #         newV[min(l+m, self.thresh),m] = V[l] * np.exp(eta * p[j][m])
            #
            # V = np.sum(newV, axis=1)
            newV = np.zeros(self.thresh+1)
            for m in range(self.ns[j]):
                for l in range(self.thresh+1):
                    ind = min(l+m,self.thresh)
                    newV[ind] += V[l] * np.exp(eta * p[j][m])
            V = newV

        # Now do it for i
        mi = np.zeros(self.ns[i])
        for m in range(self.ns[i]):
            for l in range(self.thresh+1):

                cost_scaling = 1
                if self.mode == 'best_case':
                    if m + l >= self.thresh:
                        cost_scaling = np.exp(-eta)
                elif self.mode == 'worst_case':
                    return self.marginalize_worst_case(eta, p, i)
                    if m + l < self.thresh:
                        cost_scaling = np.exp(-eta)
                else:
                    assert(False), "mode is not set correctly"

                mi[m] += V[l] * np.exp(eta * p[i][m]) * cost_scaling

        return mi


import matplotlib.pyplot as plt

def main():

    ##########################################################################
    ## A) Specify quantile aggregation


    for n in [100]:
        # print('\n' * 10)
        # print((str(n) + '\n') * 100)
        for k in [10]:
            for thresh in [20]:
                for mode in ['best_case']:
                    print(n,k,thresh,mode)
                    time.sleep(0.2)
                    distribs = [np.ones(n) / n] * k
                    # mode = 'best_case'

                    ##########################################################################
                    # B) Run column generation to solve the problem

                    prob=QuantileAggregationProblem(distribs,thresh, mode=mode)

                    eta=10
                    num_est_trials=1000

                    # starttime = time.time()
                    # p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic', naive=True)
                    # endtime = time.time()
                    # print(rankone)
                    # obj_sinkhorn_naive, obj_sinkhorn_naive_stddev = prob.estimate_rounded_sinkhorn_solution_cost(eta, p, rankone, num_est_trials, naive=True)
                    # time_sinkhorn_naive = endtime-starttime
                    # print('Total time sinkhorn: naive',time_sinkhorn_naive)
                    # print('Objective sinkhorn naive:',obj_sinkhorn_naive)
                    # print('Objective sinkhorn naive standard dev:',obj_sinkhorn_naive_stddev)

                    starttime = time.time()
                    p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic', naive=False)
                    print('p',p)
                    endtime = time.time()
                    time_sinkhorn = endtime-starttime
                    obj_sinkhorn, obj_sinkhorn_stddev = prob.estimate_rounded_sinkhorn_solution_cost(eta, p, rankone, num_est_trials, naive=False)
                    print('Total time sinkhorn:',time_sinkhorn)
                    print('Objective sinkhorn:',obj_sinkhorn)
                    print('Objective sinkhorn standard dev:',obj_sinkhorn_stddev)


                    # #
                    # #
                    # print('Total time sinkhorn: naive',time_sinkhorn_naive)
                    # print('Objective sinkhorn naive:',obj_sinkhorn_naive)
                    # print('Objective sinkhorn naive standard dev:',obj_sinkhorn_naive_stddev)
                    # print('Total time sinkhorn:',time_sinkhorn)
                    # print('Objective sinkhorn:',obj_sinkhorn)
                    # print('Objective sinkhorn standard dev:',obj_sinkhorn_stddev)

                    starttime = time.time()
                    obj_cg, sol_cg = prob.solve_cg()
                    endtime = time.time()
                    time_cg = endtime-starttime
                    print('Total time cg:',time_cg)
                    print('Objective:',obj_cg)
                    #
                    # starttime = time.time()
                    # obj_mw, sol_mw = prob.solve_mw(eps=0.01,subroutine_eps=0)
                    # endtime = time.time()
                    # time_mw = endtime - starttime
                    #
                    # # print('Total time cg:',time_cg)
                    # # print('Objective:',obj_cg)
                    # print('Total time mw:',time_mw)
                    # print('Returned approximate objective MW, rescaled:',obj_mw)
                    # print('MW sol cost', prob.get_sparse_sol_cost(sol_mw))
                    # print(sol_mw)


                    # starttime = time.time()
                    # obj_naive, sol_naive = prob.solve_naive()
                    # endtime = time.time()
                    # time_naive = endtime-starttime
                    # print('Total time cg:',time_cg)
                    # print('Objective:',obj_cg)
                    # print('Total time naive:',time_naive)
                    # print('Objective:',obj_naive)
                    #
                    # if not np.isclose(obj_naive, obj_cg):
                    #     print('Not close!')
                    #     print(thresh,k,n,mode)
                    #     assert(False)




if __name__ == '__main__':
    main()
