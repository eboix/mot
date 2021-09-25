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
import os
import pickle
import scipy.special


class RiskEstimationProblem(MOTProblem):
    def __init__(self, ratio_dists, appx_degree, problemname='Risk_Estimation_Problem'):

        # ratios is a list of list of pairs
        # the first item in the pair gives the ratio, and the second item gives
        # the probability of that ratio

        self.appx_degree = appx_degree

        self.k = len(ratio_dists)
        self.ns = [len(ratio_dists[i]) for i in range(self.k)]

        mus = []
        ratios = []
        for i in range(self.k):
            mus.append([ratio_dists[i][j][1] for j in range(self.ns[i])])
            ratios.append([ratio_dists[i][j][0] for j in range(self.ns[i])])
        mus = [np.asarray(x) for x in mus]
        ratios = [np.asarray(x) for x in ratios]
        self.mus = mus
        self.rankone = ratios

        for i in range(self.k):
            assert(np.all(self.rankone[i] >= 0)) # not really necessary, but helpful for logsumexp trick below
        minratio = np.prod([np.min(ratios[i]) for i in range(self.k)])
        maxratio = np.prod([np.max(ratios[i]) for i in range(self.k)])
        self._cost_range =[minratio, maxratio]

        super(RiskEstimationProblem, self).__init__(mus,problemname)

        # Cache the factorials for quick use later
        self._log_factorials = np.zeros(self.appx_degree+1)
        for i in range(1,self.appx_degree+1):
            self._log_factorials[i] = self._log_factorials[i-1] + np.log(i)

        self.cutting_plane_eta = None


    def get_tuple_cost(self, tup):
        return np.prod([self.rankone[i][tup[i]] for i in range(self.k)])

    def set_cutting_plane_eta(self, eta):
        self.cutting_plane_eta = eta

    def get_cutting_plane(self, dualwts):
        """ Returns approximate cutting plane, which is enough to run column
        generation or multiplicative weights. The degree of approximation depends on the "appx_degree"
        parameter in the constructor for this problem, and also on the parameter eta.
        """

        eta = self.cutting_plane_eta
        if eta is None:
            print('Set the cutting plane eta value with set_cutting_plane_eta()')
            assert(False)

        p = copy.deepcopy(dualwts)
        currtup = []
        for i in range(self.k):
            mi = self.marginalize(eta, p, i)
            ji = np.argmax(mi)
            currtup.append(ji)
            for j in range(self.ns[i]):
                if j == ji:
                    continue
                p[i][j] = -math.inf

        currtup = tuple(currtup)

        tupcost = self.get_tuple_cost(currtup)
        tupwt = sum([dualwts[i][currtup[i]] for i in range(self.k)])
        gap = tupcost - tupwt

        # print('currtup',currtup)
        # print('tupcost',tupcost)
        # print('tupwt',tupwt)
        # print('gap',gap)
        if gap < 0:
            return [currtup]
        else:
            return []

    def get_cost_range(self):
        """
        The range is the product of the minimum ratio values vs. the product of
        the maximum ratio values.
        """
        return self._cost_range

    def marginalize(self, eta, p, margidx):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i] for all i \in [k].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_{margidx}(P).
        """

        """
        The implementation is approximate and is done with respect to the
        degree-appx_degree polynomial. This is essentially by taking the
        Taylor series expansion and noting that it is sufficient to
        marginalize a sum of rank-one tensors, each corresponding to a term
        in the Taylor-series expansion:

        \exp(-\eta x) = \sum_{i=0}^{\infty} \frac{(-\eta)^i}{i!} x^i

        \exp(-\eta x) = \sum_{i=0}^{\infty} \frac{(-\eta)^i}{i!} (x-x0)^i
        Have to implement the factorial efficiently.
        """

        # PART 1:
        # Compute the degree-d terms \sum_{\vec{j} \in [n]^k} (C_{\vec{j}})^d
        log_deg_terms = []
        for d in range(self.appx_degree+1):
            logdterm = 0
            for i in range(self.k):
                if i == margidx: # Handle this separately, since we do not sum over it
                    continue

                # Compute \sum_{l} [u_i]_{l}^d \exp(\eta [p_i]_l) with logsumexp trick
                currdotprod = -math.inf
                for l in range(self.ns[i]):
                    if self.rankone[i][l] == 0:
                        continue
                    assert(self.rankone[i][l] > 0)
                    logcurrterm = d * np.log(self.rankone[i][l]) + eta * p[i][l]
                    currdotprod = np.logaddexp(currdotprod, logcurrterm)

                assert(currdotprod > -math.inf)

                logdterm += currdotprod

            # Do margidx
            logcurrmarg = np.ones(self.ns[margidx]) * logdterm
            for l in range(self.ns[margidx]):
                logcurrmarg[l] += d * np.log(self.rankone[margidx][l]) + eta * p[margidx][l]

            # logcurrmarg is the logarithm of
            # \prod_{i \in [k]} (\sum_{l} [u_i]_l^d \exp(\eta [p_i]_l))
            log_deg_terms.append(logcurrmarg)

        # PART 2:
        # Combine the degree-d terms via the Taylor series expansion of \sum_{jvec} exp(-\eta C_{jvec} + \eta \sum_{i} [p_i]_{j_i}) centered at x0:
        cost_min, cost_max = self.get_cost_range()
        cost_shift = cost_min + (cost_max - cost_min) / 2
        x0 = cost_shift

        totcost = np.zeros(self.ns[margidx])
        for t in range(self.appx_degree+1):
            currdegterms = 0
            for s in range(t+1):
                binomts = scipy.special.binom(t, s)
                currterm = binomts * np.exp(s * x0 + log_deg_terms[t-s])
                currterm *= (-1) ** s
                currdegterms += currterm
            currdegfactor = np.exp(t * np.log(eta) - self._log_factorials[t])
            if t % 2 == 1:
                currdegfactor *= -1
            totcost += currdegfactor * currdegterms
        totcost *= np.exp(x0)

        return totcost

    def appx_sinkhorn_cost_fast(self, eta, p, rankone):

        rankone_contrib = 1
        for i in range(self.k):
            dotprod = np.sum(self.rankone[i] * rankone[i])
            rankone_contrib *= dotprod

        pmod = copy.deepcopy(p)
        for i in range(self.k):
            pmod[i] += np.log(self.rankone[i]) / eta
        scaling_cost = np.sum(self.marginalize(eta, pmod, 0))
        print(scaling_cost)

        totcost = rankone_contrib + scaling_cost

        return totcost


import matplotlib.pyplot as plt

def main():

    data_file = 'risk_estimation_data.pkl'
    if not os.path.exists(data_file):
        pickle.dump({}, open(data_file, 'wb'))
    run_data = pickle.load(open(data_file, 'rb'))

    mw_eps_list = [0.02,0.01]
    naive_sinkhorn_eta_list = [200]
    sinkhorn_eta_list = [200]
    do_colgen = True
    do_naive = True

    appx_degree=5
    eta_cutting_plane = 1 # For implementing the AMIN oracle approximately (for MWU and COLGEN)

    # for k in [2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,25,30,40,50]:
    for k in [2,3,4,5]:

        n = 10

        # A) Set up problem
        # Dataset where the marginal return distribution on a time step is uniform in [1, 1 + 1/k]
        ratio_dists = []
        for i in range(k):
            ratio_dists.append([])
            for j in range(n):
                ratio_dists[i].append((1 + j/(k*n), 1/n))


        # B) Solve problem
        prob=RiskEstimationProblem(ratio_dists,appx_degree)
        prob.set_cutting_plane_eta(eta_cutting_plane)
        print(prob.get_cost_range())
        # assert(False)

        for eta in naive_sinkhorn_eta_list:
            starttime = time.time()
            p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic', naive=True)
            endtime = time.time()
            time_naive_sinkhorn = endtime-starttime
            # print(rankone)
            obj_naive_sinkhorn = prob.compute_rounded_sinkhorn_solution_cost_naive(eta, p, rankone)

            prob._delete_explicit_cost_tensor() # Do some garbage collection, since naive sinkhorn is very expensive.
            print('Total time sinkhorn naive:',time_naive_sinkhorn)
            print('Objective sinkhorn naive:',obj_naive_sinkhorn)
            run_data[(n,k,'naive_sinkhorn' + str(eta))] = (obj_naive_sinkhorn, time_naive_sinkhorn)

        for eta in sinkhorn_eta_list:
            starttime = time.time()
            p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic')
            endtime = time.time()
            time_sinkhorn = endtime-starttime

            print('Total time sinkhorn:',time_sinkhorn)
            obj_sinkhorn = prob.appx_sinkhorn_cost_fast(eta, p, rankone)
            print('Objective sinkhorn:',obj_sinkhorn)
            run_data[(n,k,'sinkhorn' + str(eta))] = (obj_sinkhorn, time_sinkhorn)

        if do_colgen:
            """ Column Generation """
            starttime = time.time()
            obj_cg, sol_cg = prob.solve_cg()
            endtime = time.time()
            time_cg = endtime-starttime
            print('Total time cg:',time_cg)
            print('Objective:',obj_cg)
            run_data[(n,k,'cg')] = (obj_cg, time_cg)

        """ Multiplicative weights """
        for mw_eps in mw_eps_list:
            starttime = time.time()
            obj_mw, sol_mw = prob.solve_mw(eps=mw_eps,subroutine_eps=0)
            endtime = time.time()
            time_mw = endtime - starttime
            print('Total time mw:',time_mw)
            obj_mw = prob.get_sparse_sol_cost(sol_mw)
            print('Objective MW:',obj_mw)
            run_data[(n,k,'mw' + str(mw_eps))] = (obj_mw, time_mw)

        if do_naive:
            """ Naive """
            starttime = time.time()
            obj_naive, sol_naive = prob.solve_naive()
            endtime = time.time()
            time_naive = endtime-starttime
            print('Total time naive:',time_naive)
            print('Objective:',obj_naive)
            run_data[(n,k,'naive')] = (obj_naive, time_naive)

        print('run_data = ',run_data)
        pickle.dump(run_data, open(data_file, 'wb'))

if __name__ == '__main__':
    main()
