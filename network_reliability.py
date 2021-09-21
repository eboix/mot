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

    mw_eps_list = []
    naive_sinkhorn_eta_list = [250]
    do_colgen = False
    do_naive = False
    run_data = {}

    # Saved results with edge_failure_prob = 0.01 for worst_case
    # and edge_failure_prob = 0.99 for best_case
    # run_data = {(2, 'worst_case', 'cg'): (0.99, 0.32831311225891113), (2, 'worst_case', 'mw0.01'): (0.99, 0.0904397964477539), (2, 'worst_case', 'naive'): (0.99, 0.0017919540405273438), (3, 'worst_case', 'cg'): (0.985, 0.0035800933837890625), (3, 'worst_case', 'mw0.01'): (0.99, 0.12365603446960449), (3, 'worst_case', 'naive'): (0.985, 0.005438089370727539), (4, 'worst_case', 'cg'): (0.98, 0.0048487186431884766), (4, 'worst_case', 'mw0.01'): (0.99, 0.1487410068511963), (4, 'worst_case', 'naive'): (0.98, 0.004221916198730469), (5, 'worst_case', 'cg'): (0.975, 0.019611120223999023), (5, 'worst_case', 'mw0.01'): (0.99, 0.17186617851257324), (5, 'worst_case', 'naive'): (0.975, 0.04087018966674805), (6, 'worst_case', 'cg'): (0.97, 0.11754798889160156), (6, 'worst_case', 'mw0.01'): (0.99, 0.19510412216186523), (6, 'worst_case', 'naive'): (0.9700000000000002, 1.766113042831421), (7, 'worst_case', 'cg'): (0.965, 0.5302259922027588), (7, 'worst_case', 'mw0.01'): (0.99, 0.21964001655578613), (7, 'worst_case', 'naive'): (0.9650000000000003, 170.3743031024933), (8, 'worst_case', 'cg'): (0.9600000000000004, 5.299043893814087), (8, 'worst_case', 'mw0.01'): (0.99, 0.25112390518188477), (9, 'worst_case', 'cg'): (0.955, 28.218992233276367), (9, 'worst_case', 'mw0.01'): (0.99, 0.2765839099884033), (10, 'worst_case', 'cg'): (0.9500000000000004, 134.6071319580078), (10, 'worst_case', 'mw0.01'): (0.9839871417969168, 0.4515340328216553), (11, 'worst_case', 'cg'): (0.9450000000000004, 629.2566766738892), (11, 'worst_case', 'mw0.01'): (0.9773215910642828, 0.6937100887298584), (12, 'worst_case', 'mw0.01'): (0.9772890503577959, 0.8318972587585449), (13, 'worst_case', 'mw0.01'): (0.9702161365199296, 1.0108661651611328), (14, 'worst_case', 'mw0.01'): (0.9631244682298886, 1.7128939628601074), (15, 'worst_case', 'mw0.01'): (0.9630838396613114, 1.926056146621704), (16, 'worst_case', 'mw0.01'): (0.9556452259731522, 2.300704002380371), (17, 'worst_case', 'mw0.01'): (0.9557593388450134, 3.008028984069824), (18, 'worst_case', 'mw0.01'): (0.9482807724044405, 4.751907825469971), (19, 'worst_case', 'mw0.01'): (0.9408955273366273, 4.362539768218994), (20, 'worst_case', 'mw0.01'): (0.9334502647105768, 8.184261083602905), (21, 'worst_case', 'mw0.01'): (0.9256422039474347, 8.477193832397461), (22, 'worst_case', 'mw0.01'): (0.9256961205419709, 9.977110147476196), (23, 'worst_case', 'mw0.01'): (0.9180752866618219, 17.388160228729248), (24, 'worst_case', 'mw0.01'): (0.9183857121746924, 21.843726873397827), (25, 'worst_case', 'mw0.01'): (0.9104662898965609, 11.802061796188354), (26, 'worst_case', 'mw0.01'): (0.9025799714829019, 22.985297918319702), (27, 'worst_case', 'mw0.01'): (0.902735613977799, 28.08206582069397), (28, 'worst_case', 'mw0.01'): (0.8946210202496002, 33.91191291809082), (29, 'worst_case', 'mw0.01'): (0.8868259136503627, 47.907634019851685), (30, 'worst_case', 'mw0.01'): (0.8866086474846441, 55.71390509605408), (31, 'worst_case', 'mw0.01'): (0.8787690003456342, 42.04777002334595), (32, 'worst_case', 'mw0.01'): (0.871140775912524, 75.8750228881836), (33, 'worst_case', 'mw0.01'): (0.8714609213536615, 89.12447905540466), (34, 'worst_case', 'mw0.01'): (0.8635174671116926, 102.7361319065094), (35, 'worst_case', 'mw0.01'): (0.8555382850427358, 153.84498596191406), (36, 'worst_case', 'mw0.01'): (0.8558552121829048, 182.83270406723022), (37, 'worst_case', 'mw0.01'): (0.8473876275282131, 77.06797099113464), (38, 'worst_case', 'mw0.01'): (0.8476974394285242, 94.60694813728333), (39, 'worst_case', 'mw0.01'): (0.8317862946929863, 169.93303203582764), (2, 'worst_case', 'mw0.005'): (0.99, 0.627539873123169), (3, 'worst_case', 'mw0.005'): (0.9899217170290878, 0.4587879180908203), (4, 'worst_case', 'mw0.005'): (0.9898663877454619, 0.5159511566162109), (5, 'worst_case', 'mw0.005'): (0.9878478550031558, 0.6182410717010498), (6, 'worst_case', 'mw0.005'): (0.9849943103360224, 0.8119139671325684), (7, 'worst_case', 'mw0.005'): (0.9818251230134042, 0.9170520305633545), (8, 'worst_case', 'mw0.005'): (0.975519215572887, 1.5047011375427246), (9, 'worst_case', 'mw0.005'): (0.9719228293812889, 2.415065050125122), (10, 'worst_case', 'mw0.005'): (0.968257555417516, 2.141869068145752), (11, 'worst_case', 'mw0.005'): (0.9613127609582092, 3.9392168521881104), (12, 'worst_case', 'mw0.005'): (0.9575002034156365, 6.530138969421387), (13, 'worst_case', 'mw0.005'): (0.9536729612872088, 3.577091693878174), (14, 'worst_case', 'mw0.005'): (0.9498237684861256, 7.162118196487427), (15, 'worst_case', 'mw0.005'): (0.9425359194348905, 13.695715188980103), (16, 'worst_case', 'mw0.005'): (0.9386316811425811, 10.444837093353271), (17, 'worst_case', 'mw0.005'): (0.9347278924541159, 18.006296157836914), (18, 'worst_case', 'mw0.005'): (0.9272942190106643, 30.996285915374756), (19, 'worst_case', 'mw0.005'): (0.923357394040624, 17.708465337753296), (20, 'worst_case', 'mw0.005'): (0.9194259029354569, 32.01845717430115), (21, 'worst_case', 'mw0.005'): (0.9157504859797112, 37.07878398895264), (22, 'worst_case', 'mw0.005'): (0.9079445471487111, 43.07141995429993), (23, 'worst_case', 'mw0.005'): (0.9039946460516584, 72.4646053314209), (24, 'worst_case', 'mw0.005'): (0.9000526946514551, 85.72529315948486), (25, 'worst_case', 'mw0.005'): (0.8924414530687039, 35.06098031997681), (26, 'worst_case', 'mw0.005'): (0.8884798828318915, 83.95556306838989), (27, 'worst_case', 'mw0.005'): (0.8845225930813336, 92.522940158844), (28, 'worst_case', 'mw0.005'): (0.8808386113733079, 183.93406009674072), (29, 'worst_case', 'mw0.005'): (0.8729093870421281, 195.16129279136658), (2, 'worst_case', 'mw0.02'): (0.99, 0.19539904594421387), (3, 'worst_case', 'mw0.02'): (0.99, 0.06293487548828125), (4, 'worst_case', 'mw0.02'): (0.99, 0.07649898529052734), (5, 'worst_case', 'mw0.02'): (0.99, 0.08864903450012207), (6, 'worst_case', 'mw0.02'): (0.99, 0.10160589218139648), (7, 'worst_case', 'mw0.02'): (0.99, 0.12518715858459473), (8, 'worst_case', 'mw0.02'): (0.99, 0.1293478012084961), (9, 'worst_case', 'mw0.02'): (0.99, 0.14352893829345703), (10, 'worst_case', 'mw0.02'): (0.99, 0.16006994247436523), (11, 'worst_case', 'mw0.02'): (0.99, 0.1806640625), (12, 'worst_case', 'mw0.02'): (0.99, 0.2099781036376953), (13, 'worst_case', 'mw0.02'): (0.99, 0.2254033088684082), (14, 'worst_case', 'mw0.02'): (0.989607300854484, 0.28037500381469727), (15, 'worst_case', 'mw0.02'): (0.989607300854484, 0.3083021640777588), (16, 'worst_case', 'mw0.02'): (0.989607300854484, 0.3927140235900879), (17, 'worst_case', 'mw0.02'): (0.9896089570129757, 0.458599328994751), (18, 'worst_case', 'mw0.02'): (0.9896089570129757, 0.5197548866271973), (19, 'worst_case', 'mw0.02'): (0.975461125425329, 0.711083173751831), (20, 'worst_case', 'mw0.02'): (0.9755179980147329, 0.894895076751709), (21, 'worst_case', 'mw0.02'): (0.9606104096278236, 1.4119789600372314), (22, 'worst_case', 'mw0.02'): (0.9606572455137756, 1.5152082443237305), (23, 'worst_case', 'mw0.02'): (0.9606567070934793, 1.6774232387542725), (24, 'worst_case', 'mw0.02'): (0.945527742404948, 1.8688852787017822), (25, 'worst_case', 'mw0.02'): (0.9457272806634652, 2.084625720977783), (26, 'worst_case', 'mw0.02'): (0.9300774574771992, 4.47345495223999), (27, 'worst_case', 'mw0.02'): (0.9303227494401075, 5.313427209854126), (28, 'worst_case', 'mw0.02'): (0.9145490831982035, 6.300863981246948), (29, 'worst_case', 'mw0.02'): (0.9146369149432431, 8.310882091522217), (30, 'worst_case', 'mw0.02'): (0.9146101417725587, 9.447694778442383), (31, 'worst_case', 'mw0.02'): (0.8993033819993095, 17.312218189239502), (32, 'worst_case', 'mw0.02'): (0.899329478272056, 20.42208504676819), (33, 'worst_case', 'mw0.02'): (0.8835172352866459, 16.644195079803467), (34, 'worst_case', 'mw0.02'): (0.8833917404808458, 16.0820529460907), (35, 'worst_case', 'mw0.02'): (0.88363052908665, 19.307225227355957), (36, 'worst_case', 'mw0.02'): (0.8676717075473335, 39.4571259021759), (37, 'worst_case', 'mw0.02'): (0.8676217793609322, 40.53036642074585), (38, 'worst_case', 'mw0.02'): (0.8677707806419537, 49.747198820114136), (39, 'worst_case', 'mw0.02'): (0.8681137442107542, 53.56041598320007)}

    run_data = {(2, 'worst_case', 'cg'): (0.99, 0.32831311225891113), (2, 'worst_case', 'mw0.01'): (0.99, 0.0904397964477539), (2, 'worst_case', 'naive'): (0.99, 0.0017919540405273438), (3, 'worst_case', 'cg'): (0.985, 0.0035800933837890625), (3, 'worst_case', 'mw0.01'): (0.99, 0.12365603446960449), (3, 'worst_case', 'naive'): (0.985, 0.005438089370727539), (4, 'worst_case', 'cg'): (0.98, 0.0048487186431884766), (4, 'worst_case', 'mw0.01'): (0.99, 0.1487410068511963), (4, 'worst_case', 'naive'): (0.98, 0.004221916198730469), (5, 'worst_case', 'cg'): (0.975, 0.019611120223999023), (5, 'worst_case', 'mw0.01'): (0.99, 0.17186617851257324), (5, 'worst_case', 'naive'): (0.975, 0.04087018966674805), (6, 'worst_case', 'cg'): (0.97, 0.11754798889160156), (6, 'worst_case', 'mw0.01'): (0.99, 0.19510412216186523), (6, 'worst_case', 'naive'): (0.9700000000000002, 1.766113042831421), (7, 'worst_case', 'cg'): (0.965, 0.5302259922027588), (7, 'worst_case', 'mw0.01'): (0.99, 0.21964001655578613), (7, 'worst_case', 'naive'): (0.9650000000000003, 170.3743031024933), (8, 'worst_case', 'cg'): (0.9600000000000004, 5.299043893814087), (8, 'worst_case', 'mw0.01'): (0.99, 0.25112390518188477), (9, 'worst_case', 'cg'): (0.955, 28.218992233276367), (9, 'worst_case', 'mw0.01'): (0.99, 0.2765839099884033), (10, 'worst_case', 'cg'): (0.9500000000000004, 134.6071319580078), (10, 'worst_case', 'mw0.01'): (0.9839871417969168, 0.4515340328216553), (11, 'worst_case', 'cg'): (0.9450000000000004, 629.2566766738892), (11, 'worst_case', 'mw0.01'): (0.9773215910642828, 0.6937100887298584), (12, 'worst_case', 'mw0.01'): (0.9772890503577959, 0.8318972587585449), (13, 'worst_case', 'mw0.01'): (0.9702161365199296, 1.0108661651611328), (14, 'worst_case', 'mw0.01'): (0.9631244682298886, 1.7128939628601074), (15, 'worst_case', 'mw0.01'): (0.9630838396613114, 1.926056146621704), (16, 'worst_case', 'mw0.01'): (0.9556452259731522, 2.300704002380371), (17, 'worst_case', 'mw0.01'): (0.9557593388450134, 3.008028984069824), (18, 'worst_case', 'mw0.01'): (0.9482807724044405, 4.751907825469971), (19, 'worst_case', 'mw0.01'): (0.9408955273366273, 4.362539768218994), (20, 'worst_case', 'mw0.01'): (0.9334502647105768, 8.184261083602905), (21, 'worst_case', 'mw0.01'): (0.9256422039474347, 8.477193832397461), (22, 'worst_case', 'mw0.01'): (0.9256961205419709, 9.977110147476196), (23, 'worst_case', 'mw0.01'): (0.9180752866618219, 17.388160228729248), (24, 'worst_case', 'mw0.01'): (0.9183857121746924, 21.843726873397827), (25, 'worst_case', 'mw0.01'): (0.9104662898965609, 11.802061796188354), (26, 'worst_case', 'mw0.01'): (0.9025799714829019, 22.985297918319702), (27, 'worst_case', 'mw0.01'): (0.902735613977799, 28.08206582069397), (28, 'worst_case', 'mw0.01'): (0.8946210202496002, 33.91191291809082), (29, 'worst_case', 'mw0.01'): (0.8868259136503627, 47.907634019851685), (30, 'worst_case', 'mw0.01'): (0.8866086474846441, 55.71390509605408), (31, 'worst_case', 'mw0.01'): (0.8787690003456342, 42.04777002334595), (32, 'worst_case', 'mw0.01'): (0.871140775912524, 75.8750228881836), (33, 'worst_case', 'mw0.01'): (0.8714609213536615, 89.12447905540466), (34, 'worst_case', 'mw0.01'): (0.8635174671116926, 102.7361319065094), (35, 'worst_case', 'mw0.01'): (0.8555382850427358, 153.84498596191406), (36, 'worst_case', 'mw0.01'): (0.8558552121829048, 182.83270406723022), (37, 'worst_case', 'mw0.01'): (0.8473876275282131, 77.06797099113464), (38, 'worst_case', 'mw0.01'): (0.8476974394285242, 94.60694813728333), (39, 'worst_case', 'mw0.01'): (0.8317862946929863, 169.93303203582764), (2, 'worst_case', 'mw0.005'): (0.99, 0.627539873123169), (3, 'worst_case', 'mw0.005'): (0.9899217170290878, 0.4587879180908203), (4, 'worst_case', 'mw0.005'): (0.9898663877454619, 0.5159511566162109), (5, 'worst_case', 'mw0.005'): (0.9878478550031558, 0.6182410717010498), (6, 'worst_case', 'mw0.005'): (0.9849943103360224, 0.8119139671325684), (7, 'worst_case', 'mw0.005'): (0.9818251230134042, 0.9170520305633545), (8, 'worst_case', 'mw0.005'): (0.975519215572887, 1.5047011375427246), (9, 'worst_case', 'mw0.005'): (0.9719228293812889, 2.415065050125122), (10, 'worst_case', 'mw0.005'): (0.968257555417516, 2.141869068145752), (11, 'worst_case', 'mw0.005'): (0.9613127609582092, 3.9392168521881104), (12, 'worst_case', 'mw0.005'): (0.9575002034156365, 6.530138969421387), (13, 'worst_case', 'mw0.005'): (0.9536729612872088, 3.577091693878174), (14, 'worst_case', 'mw0.005'): (0.9498237684861256, 7.162118196487427), (15, 'worst_case', 'mw0.005'): (0.9425359194348905, 13.695715188980103), (16, 'worst_case', 'mw0.005'): (0.9386316811425811, 10.444837093353271), (17, 'worst_case', 'mw0.005'): (0.9347278924541159, 18.006296157836914), (18, 'worst_case', 'mw0.005'): (0.9272942190106643, 30.996285915374756), (19, 'worst_case', 'mw0.005'): (0.923357394040624, 17.708465337753296), (20, 'worst_case', 'mw0.005'): (0.9194259029354569, 32.01845717430115), (21, 'worst_case', 'mw0.005'): (0.9157504859797112, 37.07878398895264), (22, 'worst_case', 'mw0.005'): (0.9079445471487111, 43.07141995429993), (23, 'worst_case', 'mw0.005'): (0.9039946460516584, 72.4646053314209), (24, 'worst_case', 'mw0.005'): (0.9000526946514551, 85.72529315948486), (25, 'worst_case', 'mw0.005'): (0.8924414530687039, 35.06098031997681), (26, 'worst_case', 'mw0.005'): (0.8884798828318915, 83.95556306838989), (27, 'worst_case', 'mw0.005'): (0.8845225930813336, 92.522940158844), (28, 'worst_case', 'mw0.005'): (0.8808386113733079, 183.93406009674072), (29, 'worst_case', 'mw0.005'): (0.8729093870421281, 195.16129279136658), (2, 'worst_case', 'mw0.02'): (0.99, 0.19539904594421387), (3, 'worst_case', 'mw0.02'): (0.99, 0.06293487548828125), (4, 'worst_case', 'mw0.02'): (0.99, 0.07649898529052734), (5, 'worst_case', 'mw0.02'): (0.99, 0.08864903450012207), (6, 'worst_case', 'mw0.02'): (0.99, 0.10160589218139648), (7, 'worst_case', 'mw0.02'): (0.99, 0.12518715858459473), (8, 'worst_case', 'mw0.02'): (0.99, 0.1293478012084961), (9, 'worst_case', 'mw0.02'): (0.99, 0.14352893829345703), (10, 'worst_case', 'mw0.02'): (0.99, 0.16006994247436523), (11, 'worst_case', 'mw0.02'): (0.99, 0.1806640625), (12, 'worst_case', 'mw0.02'): (0.99, 0.2099781036376953), (13, 'worst_case', 'mw0.02'): (0.99, 0.2254033088684082), (14, 'worst_case', 'mw0.02'): (0.989607300854484, 0.28037500381469727), (15, 'worst_case', 'mw0.02'): (0.989607300854484, 0.3083021640777588), (16, 'worst_case', 'mw0.02'): (0.989607300854484, 0.3927140235900879), (17, 'worst_case', 'mw0.02'): (0.9896089570129757, 0.458599328994751), (18, 'worst_case', 'mw0.02'): (0.9896089570129757, 0.5197548866271973), (19, 'worst_case', 'mw0.02'): (0.975461125425329, 0.711083173751831), (20, 'worst_case', 'mw0.02'): (0.9755179980147329, 0.894895076751709), (21, 'worst_case', 'mw0.02'): (0.9606104096278236, 1.4119789600372314), (22, 'worst_case', 'mw0.02'): (0.9606572455137756, 1.5152082443237305), (23, 'worst_case', 'mw0.02'): (0.9606567070934793, 1.6774232387542725), (24, 'worst_case', 'mw0.02'): (0.945527742404948, 1.8688852787017822), (25, 'worst_case', 'mw0.02'): (0.9457272806634652, 2.084625720977783), (26, 'worst_case', 'mw0.02'): (0.9300774574771992, 4.47345495223999), (27, 'worst_case', 'mw0.02'): (0.9303227494401075, 5.313427209854126), (28, 'worst_case', 'mw0.02'): (0.9145490831982035, 6.300863981246948), (29, 'worst_case', 'mw0.02'): (0.9146369149432431, 8.310882091522217), (30, 'worst_case', 'mw0.02'): (0.9146101417725587, 9.447694778442383), (31, 'worst_case', 'mw0.02'): (0.8993033819993095, 17.312218189239502), (32, 'worst_case', 'mw0.02'): (0.899329478272056, 20.42208504676819), (33, 'worst_case', 'mw0.02'): (0.8835172352866459, 16.644195079803467), (34, 'worst_case', 'mw0.02'): (0.8833917404808458, 16.0820529460907), (35, 'worst_case', 'mw0.02'): (0.88363052908665, 19.307225227355957), (36, 'worst_case', 'mw0.02'): (0.8676717075473335, 39.4571259021759), (37, 'worst_case', 'mw0.02'): (0.8676217793609322, 40.53036642074585), (38, 'worst_case', 'mw0.02'): (0.8677707806419537, 49.747198820114136), (39, 'worst_case', 'mw0.02'): (0.8681137442107542, 53.56041598320007), (2, 'best_case', 'cg'): (0.99, 0.004294157028198242), (2, 'best_case', 'mw0.02'): (0.99, 0.0435938835144043), (2, 'best_case', 'mw0.01'): (0.99, 0.08817911148071289), (2, 'best_case', 'mw0.005'): (0.99, 0.35957932472229004), (2, 'best_case', 'naive'): (0.99, 0.0007717609405517578), (3, 'best_case', 'cg'): (0.985, 0.0031740665435791016), (3, 'best_case', 'mw0.02'): (0.99, 0.061760902404785156), (3, 'best_case', 'mw0.01'): (0.99, 0.12808895111083984), (3, 'best_case', 'mw0.005'): (0.9899217170290878, 0.45632004737854004), (3, 'best_case', 'naive'): (0.985, 0.0034360885620117188), (4, 'best_case', 'cg'): (0.98, 0.0041348934173583984), (4, 'best_case', 'mw0.02'): (0.99, 0.07466602325439453), (4, 'best_case', 'mw0.01'): (0.99, 0.14864897727966309), (4, 'best_case', 'mw0.005'): (0.9899217938879611, 0.5122277736663818), (4, 'best_case', 'naive'): (0.9799999999999999, 0.0034749507904052734), (5, 'best_case', 'cg'): (0.9750000000000001, 0.011469125747680664), (5, 'best_case', 'mw0.02'): (0.99, 0.08607101440429688), (5, 'best_case', 'mw0.01'): (0.99, 0.17467212677001953), (5, 'best_case', 'mw0.005'): (0.9877951437351418, 0.6186048984527588), (5, 'best_case', 'naive'): (0.9750000000000003, 0.03927302360534668), (6, 'best_case', 'cg'): (0.9700000000000004, 0.028963088989257812), (6, 'best_case', 'mw0.02'): (0.99, 0.09827899932861328), (6, 'best_case', 'mw0.01'): (0.99, 0.19674301147460938), (6, 'best_case', 'mw0.005'): (0.9849943103360225, 0.8031308650970459), (6, 'best_case', 'naive'): (0.97, 1.720952033996582), (7, 'best_case', 'cg'): (0.9649999999999995, 0.059966325759887695), (7, 'best_case', 'mw0.02'): (0.99, 0.1099860668182373), (7, 'best_case', 'mw0.01'): (0.99, 0.22238397598266602), (7, 'best_case', 'mw0.005'): (0.9790097543653343, 1.2292578220367432), (7, 'best_case', 'naive'): (0.965, 180.70152497291565), (8, 'best_case', 'cg'): (0.9600000000000001, 0.14156222343444824), (8, 'best_case', 'mw0.02'): (0.99, 0.13262367248535156), (8, 'best_case', 'mw0.01'): (0.99, 0.25236082077026367), (8, 'best_case', 'mw0.005'): (0.9755192155728875, 1.5487051010131836), (9, 'best_case', 'cg'): (0.9550000000000002, 0.2857799530029297), (9, 'best_case', 'mw0.02'): (0.99, 0.1494462490081787), (9, 'best_case', 'mw0.01'): (0.99, 0.293302059173584), (9, 'best_case', 'mw0.005'): (0.9721206209654745, 2.512089967727661), (10, 'best_case', 'cg'): (0.9500000000000011, 0.6288361549377441), (10, 'best_case', 'mw0.02'): (0.99, 0.16405582427978516), (10, 'best_case', 'mw0.01'): (0.9839871417969168, 0.44810914993286133), (10, 'best_case', 'mw0.005'): (0.9650902501933767, 3.529460906982422), (11, 'best_case', 'cg'): (0.9449999999999973, 0.9581360816955566), (11, 'best_case', 'mw0.02'): (0.99, 0.17914223670959473), (11, 'best_case', 'mw0.01'): (0.9773853306391249, 0.7034859657287598), (11, 'best_case', 'mw0.005'): (0.9614424005901625, 4.245770215988159), (12, 'best_case', 'mw0.02'): (0.99, 0.20393109321594238), (12, 'best_case', 'mw0.01'): (0.9704774553612631, 0.8886370658874512), (12, 'best_case', 'mw0.005'): (0.957552351899257, 7.1232991218566895), (13, 'best_case', 'mw0.02'): (0.99, 0.2203531265258789), (13, 'best_case', 'mw0.01'): (0.9704727771846879, 1.038503885269165), (13, 'best_case', 'mw0.005'): (0.9503825992036787, 7.482381105422974), (14, 'best_case', 'mw0.02'): (0.9896073008544843, 0.2638061046600342), (14, 'best_case', 'mw0.01'): (0.9632516925352707, 1.789376974105835), (14, 'best_case', 'mw0.005'): (0.9465380789491702, 8.45215630531311), (15, 'best_case', 'mw0.02'): (0.9896073008544843, 0.2916851043701172), (15, 'best_case', 'mw0.01'): (0.9633389632679589, 2.016002655029297), (15, 'best_case', 'mw0.005'): (0.9425359194348905, 13.415178775787354), (16, 'best_case', 'mw0.02'): (0.975430885742509, 0.4592411518096924), (16, 'best_case', 'mw0.01'): (0.9559204550591784, 2.388062000274658), (16, 'best_case', 'mw0.005'): (0.9352295139985655, 18.593050003051758), (17, 'best_case', 'mw0.02'): (0.9753673514145769, 0.5617890357971191), (17, 'best_case', 'mw0.01'): (0.9487274041063374, 4.595771789550781), (17, 'best_case', 'mw0.005'): (0.9351642949235381, 20.287764072418213), (18, 'best_case', 'mw0.02'): (0.9755250561347933, 0.5793948173522949), (18, 'best_case', 'mw0.01'): (0.9412162883525628, 4.087914943695068), (18, 'best_case', 'mw0.005'): (0.9275343873729514, 37.18938183784485), (19, 'best_case', 'mw0.02'): (0.9756222263610499, 0.6525261402130127), (19, 'best_case', 'mw0.01'): (0.9336828798313394, 8.877375841140747), (19, 'best_case', 'mw0.005'): (0.9198683405667002, 35.66055107116699), (20, 'best_case', 'mw0.02'): (0.9609337110953705, 1.259092092514038), (20, 'best_case', 'mw0.01'): (0.9337673019457284, 9.222961187362671), (20, 'best_case', 'mw0.005'): (0.9159249929234986, 36.10061979293823), (21, 'best_case', 'mw0.02'): (0.9611536221838818, 1.4468438625335693), (21, 'best_case', 'mw0.01'): (0.9260726794154152, 11.07905101776123), (21, 'best_case', 'mw0.005'): (0.9122506590667787, 70.36021399497986), (22, 'best_case', 'mw0.02'): (0.9611896331296311, 1.6156578063964844), (22, 'best_case', 'mw0.01'): (0.9262337403237135, 13.19231915473938), (22, 'best_case', 'mw0.005'): (0.9081537643125458, 50.21665287017822), (23, 'best_case', 'mw0.02'): (0.9613486262470725, 1.7672569751739502), (23, 'best_case', 'mw0.01'): (0.9185445998293547, 25.48898983001709), (23, 'best_case', 'mw0.005'): (0.9006081763999991, 105.64404892921448), (24, 'best_case', 'mw0.02'): (0.9461908170354304, 2.357923984527588), (24, 'best_case', 'mw0.01'): (0.9106841410604611, 11.707635879516602), (24, 'best_case', 'mw0.005'): (0.8928436704288433, 51.122443199157715), (25, 'best_case', 'mw0.02'): (0.9464790066918017, 2.476701021194458), (25, 'best_case', 'mw0.01'): (0.9029495419094614, 27.495455980300903), (25, 'best_case', 'mw0.005'): (0.8889062530983269, 105.69213199615479), (26, 'best_case', 'mw0.02'): (0.9310564211639403, 5.702277183532715), (26, 'best_case', 'mw0.01'): (0.8951699938719159, 32.610620975494385), (26, 'best_case', 'mw0.005'): (0.884889138612264, 112.99748992919922), (27, 'best_case', 'mw0.02'): (0.9312076386924807, 6.448559284210205), (27, 'best_case', 'mw0.01'): (0.8953226502293731, 40.45257306098938), (27, 'best_case', 'mw0.005'): (0.8809820759342208, 181.58493423461914), (28, 'best_case', 'mw0.02'): (0.91556274936015, 9.23639702796936), (28, 'best_case', 'mw0.01'): (0.8871967701694926, 64.68812322616577), (28, 'best_case', 'mw0.005'): (0.8771653744658737, 171.16812586784363), (29, 'best_case', 'mw0.02'): (0.8997015694864079, 18.250203132629395), (29, 'best_case', 'mw0.01'): (0.8874752005434354, 75.10393977165222), (29, 'best_case', 'mw0.005'): (0.8695401069495223, 328.3083288669586), (12, 'best_case', 'cg'): (0.9400000000000006, 1.9143998622894287), (13, 'best_case', 'cg'): (0.9349999999999982, 3.8014070987701416), (14, 'best_case', 'cg'): (0.9300000000000004, 8.107293844223022), (15, 'best_case', 'cg'): (0.9250000000000002, 16.634601831436157), (16, 'best_case', 'cg'): (0.9199999999999965, 29.968029737472534), (17, 'best_case', 'cg'): (0.9149999999999974, 58.00447916984558), (18, 'best_case', 'cg'): (0.9099999999999985, 124.94270372390747), (19, 'best_case', 'cg'): (0.9050000000000021, 215.01771330833435), (2, 'best_case', 'naive_sinkhorn500'): (0.99, 0.00034308433532714844), (3, 'best_case', 'naive_sinkhorn500'): (0.9850695997660102, 0.005579948425292969), (4, 'best_case', 'naive_sinkhorn500'): (0.9812113050150274, 0.021799802780151367), (5, 'best_case', 'naive_sinkhorn500'): (0.9768503858452062, 0.0890970230102539), (6, 'best_case', 'naive_sinkhorn500'): (0.9742231511510174, 0.9916102886199951), (3, 'worst_case', 'naive_sinkhorn500'): (0.9850695997660661, 0.0057108402252197266), (4, 'worst_case', 'naive_sinkhorn500'): (0.9819895769773064, 0.022968053817749023), (5, 'worst_case', 'naive_sinkhorn500'): (0.9777376475030766, 0.16019797325134277), (6, 'worst_case', 'naive_sinkhorn500'): (0.9707826025634957, 1.443688154220581), (7, 'worst_case', 'naive_sinkhorn500'): (0.9719969969804834, 91.74867081642151), (7, 'best_case', 'naive_sinkhorn500'): (0.9703261989247922, 83.11300706863403)}

    for v in range(3,8):

        for i in range(100):
            print(v)

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

            if do_colgen and v < 20:
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

            if v < 8 and k < 25:
                if do_naive:
                    """ Naive """
                    starttime = time.time()
                    obj_naive, sol_naive = prob.solve_naive()
                    endtime = time.time()
                    time_naive = endtime-starttime
                    print('Total time naive:',time_naive)
                    print('Objective:',obj_naive)
                    run_data[(v,mode, 'naive')] = (obj_naive, time_naive)
            print('run_data',run_data)
    print(run_data)

    for l in run_data.items():
        print(l)


if __name__ == '__main__':
    main()
