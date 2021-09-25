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


class NetworkReliabilityProblem(MOTProblem):
    def __init__(self, v, edge_failure_dict, mode, problemname='Network_Reliability_Problem'):
        # There are v vertices, numbered 0 to v-1
        # edge_failure_dict is dict mapping pairs of vertices to an edge failure probability
        # mode is either "worst_case" or "best_case", depending on the quantity we wish to compute
        self.v = v
        self.k = len(edge_failure_dict)
        self.n = 2
        self.edges = list(edge_failure_dict.keys())
        self.mode = mode
        assert(mode in ['worst_case', 'best_case'])

        mus = []
        for e in self.edges:
            e_prob = edge_failure_dict[e]
            mus.append(np.asarray([e_prob, 1 - e_prob]))

        super(NetworkReliabilityProblem, self).__init__(mus,problemname)

    def get_tuple_cost(self, tup):
        G = nx.Graph()
        G.add_nodes_from(range(self.v))

        for e_idx in range(len(self.edges)):
            if tup[e_idx]:
                G.add_edge(self.edges[e_idx][0], self.edges[e_idx][1])

        is_conn = nx.is_connected(G)

        # In Python, 0 = False, 1 = True
        if self.mode == 'worst_case':
            return is_conn
        elif self.mode == 'best_case':
            return 1-is_conn

    def get_cost_range(self):
        return [0,1]

    def get_cutting_plane_best_case(self, dualwts):
        """ Due to round-off errors, this does not return tuples that have already been returned once"""
        # Is there a subset of edges S such that f(p,S) = \sum_{e \in S} p_{e,1} + \sum_{e \not\in S} p_{e,0} > C_S,
        # where C_S is 1-1(the subgraph with edges S is connected)?

        # First, check if there is a subset of edges such that such that f(p,S) > 1, because
        # that trumps either case.
        totsum = sum([np.max(x) for x in dualwts])
        argmaxtuple = tuple(np.argmax(np.asarray(dualwts), 1))
        if totsum > 1 + 1e-6: #  and argmaxtuple not in self.tups_set
            print('totsum return more than 1',totsum)
            return [argmaxtuple]

        # print('dualwts', dualwts)
        # print('totsum', totsum)

        # Second, check if there is a subset T of edges such that T is connected
        # graph and f(p,T) > 0. This can be solved by essentially a Kruskal algorithm
        # with a union-find data structure.

        weighted_edges = []
        for e_idx in range(len(self.edges)):
            netwt = dualwts[e_idx][0] - dualwts[e_idx][1]
            weighted_edges.append((netwt, e_idx))

        weighted_edges.sort()
        ds = DisjointSet()
        totwt = 0
        included = [0]*len(self.edges)
        for netwt,e_idx, in weighted_edges:
            wt0 = dualwts[e_idx][0]
            wt1 = dualwts[e_idx][1]
            s = self.edges[e_idx][0]
            t = self.edges[e_idx][1]
            if netwt <= 0:
                ds.union(s,t)
                included[e_idx] = 1
                totwt += wt1
            else:
                if not ds.connected(s, t):
                    ds.union(s, t)
                    included[e_idx] = 1
                    totwt += wt1
                else:
                    totwt += wt0

        # print(list(ds.itersets()))
        conncomps = len(list(ds.itersets()))
        if (conncomps > 1 and totwt > 1) or (conncomps == 1 and totwt > 0):
            return [tuple(included)]
        else:
            return []

        assert(False)


    def get_cutting_plane_worst_case(self, dualwts):
        """ Due to round-off errors, this does not return tuples that have already been returned once"""
        # Is there a subset of edges S such that f(p,S) = \sum_{e \in S} p_{e,1} + \sum_{e \not\in S} p_{e,0} > C_S,
        # where C_S is 1(the subgraph with edges S is connected)?

        # First, check if there is a subset of edges such that such that f(p,S) > 1, because
        # that trumps either case.
        totsum = sum([np.max(x) for x in dualwts])
        print(totsum)
        argmaxtuple = tuple(np.argmax(np.asarray(dualwts), 1))
        if totsum > 1 + 1e-8: #  and argmaxtuple not in self.tups_set
            # print('totsum return more than 1',totsum)
            return [argmaxtuple]

        # print('dualwts', dualwts)
        # print('totsum', totsum)

        # Second, check if there is a subset T of edges such that T is a disconnected
        # graph and f(p,T) > 0. This can be solved by min-cut, starting from the
        # graph given by the 1-value edges S in maxplist, and removing the set of
        # edges S' weighted by w(e) = p_{e,1} - p_{e,0} > 0 such that
        # \sum_{e \in S'} w(e) is minimized and T = S \ S' disconnects the nodes

        G = nx.Graph()
        G.add_nodes_from(range(self.v))
        print(G)
        for e_idx in range(len(self.edges)):
            currwt = dualwts[e_idx][1] - dualwts[e_idx][0]
            if currwt > 0:
                G.add_edge(self.edges[e_idx][0], self.edges[e_idx][1], weight=currwt)
        if not nx.is_connected(G):
            if totsum > 0:
                print('totsum return', totsum)
                return [argmaxtuple]
            else:
                return []
        else:
            cut_value, partition = nx.stoer_wagner(G)
            print(partition)
            side = [0]*self.v
            for x in partition[1]:
                side[x] = 1
            included = [0]*len(self.edges)
            totwt = 0
            for e_idx in range(len(self.edges)):
                currwt = dualwts[e_idx][1] - dualwts[e_idx][0]
                if currwt > 0:
                    s,t = self.edges[e_idx]
                    if side[s] == side[t]:
                        included[e_idx] = 1
                if included[e_idx]:
                    totwt += dualwts[e_idx][1]
                else:
                    totwt += dualwts[e_idx][0]
            # print('included totwt',included,totwt)
            if totwt > 0:
                return [tuple(included)]
            else:
                return []

    def get_cutting_plane(self, dualwts):
        # dualwts is a length-k list of length-2 lists
        if self.mode == 'best_case':
            return self.get_cutting_plane_best_case(dualwts)
        elif self.mode == 'worst_case':
            return self.get_cutting_plane_worst_case(dualwts)
        else:
            assert(False), "mode is not set correctly"

import matplotlib.pyplot as plt

def main():

    data_file = 'network_reliability_data.pkl'
    if not os.path.exists(data_file):
        pickle.dump({}, open(data_file, 'wb'))
    run_data = pickle.load(open(data_file, 'rb'))

    mw_eps_list = [0.02,0.01,0.005]
    naive_sinkhorn_eta_list = [500]
    do_colgen = True
    do_naive = True

    for v in range(2,6):

        for mode in ['worst_case', 'best_case']:
            ##########################################################################
            ## A) Generate graph test

            if mode =='best_case':
                edge_failure_prob = 0.99
            elif mode == 'worst_case':
                edge_failure_prob = 0.01
            else:
                assert(False)

            # # Path graph
            # input_graph = {(i,i+1) : edge_failure_prob for i in range(v-1)}

            # # Random Graph
            # input_graph = {tuple(x) : edge_failure_prob for x in itertools.combinations(range(v),2) if random.random() < 0.5}

            # Complete graph
            input_graph = {tuple(x) : edge_failure_prob for x in itertools.combinations(range(v),2)}


            k = len(input_graph)
            print('k = number of edges = ',k)

            ##########################################################################
            # B) Solve

            prob=NetworkReliabilityProblem(v, input_graph, mode=mode)

            for eta in naive_sinkhorn_eta_list:
                starttime = time.time()
                p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.01, method='cyclic', naive=True)
                endtime = time.time()
                time_naive_sinkhorn = endtime-starttime
                # print(rankone)
                obj_naive_sinkhorn = prob.compute_rounded_sinkhorn_solution_cost_naive(eta, p, rankone)

                prob._delete_explicit_cost_tensor() # Do some garbage collection, since naive sinkhorn is very expensive.
                print('Total time sinkhorn: naive',time_naive_sinkhorn)
                print('Objective sinkhorn naive:',obj_naive_sinkhorn)
                run_data[(v,mode, 'naive_sinkhorn' + str(eta))] = (obj_naive_sinkhorn, time_naive_sinkhorn)

            if do_colgen:
                """ Column Generation """
                starttime = time.time()
                obj_cg, sol_cg = prob.solve_cg()
                endtime = time.time()
                time_cg = endtime-starttime
                print('Total time cg:',time_cg)
                print('Objective:',obj_cg)
                run_data[(v,mode, 'cg')] = (obj_cg, time_cg)

            """ Multiplicative weights """
            for mw_eps in mw_eps_list:
                starttime = time.time()
                obj_mw, sol_mw = prob.solve_mw(eps=mw_eps,subroutine_eps=0)
                endtime = time.time()
                time_mw = endtime - starttime
                print('Total time mw:',time_mw)
                obj_mw = prob.get_sparse_sol_cost(sol_mw)
                print('Objective MW:',obj_mw)
                run_data[(v,mode,'mw' + str(mw_eps))] = (obj_mw, time_mw)

            if do_naive:
                """ Naive """
                starttime = time.time()
                obj_naive, sol_naive = prob.solve_naive()
                endtime = time.time()
                time_naive = endtime-starttime
                print('Total time naive:',time_naive)
                print('Objective:',obj_naive)
                run_data[(v,mode, 'naive')] = (obj_naive, time_naive)

            print('run_data = ',run_data)
            pickle.dump(run_data, open(data_file, 'wb'))


if __name__ == '__main__':
    main()
