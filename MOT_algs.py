# General multimarginal optimal transport problem solved using column generation.
from pulp import * ## import pulp-or functions
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from functools import partial
import scipy.special
import copy
import random
import math
import scipy.stats

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

class MOTProblem:
    """
    This class contains implementations of several methods to solve MOT problems.
    In order to use it to solve a new MOT problem, it should be extended
    and the bottleneck methods called by the solvers should be implemented.
    The organization of this file is as follows.

    0. Abstract bottlenecks. Contains the abstract methods that need to be
    implemented so that the MOT solvers can be run, (i.e., the cutting plane
    oracle [equivalent to MIN or AMIN], or marginalization oracle [equivalent to SMIN]).

    1. Column generation and naive LP solver implementations,
    called using solve_cg and solve_naive, respectively.

    2. Multiplicative weights implementation, called using solve_mw.

    3. Sinkhorn implementation, called using solve_sinkhorn (with naive flag
    set to to True for a brute-force implementation).

    4. Utility methods for sparse solution (e.g., from column generation or
    multiplicative weights). Methods to compute the cost, pairwise marginals, etc...

    5. Utility methods for Sinkhorn solutions. Sampling methods, methods to estimate
    the cost, methods to compute pairwise marginals, etc...
    """

    def __init__(self, mus, problemname='MOT_Problem'):
        """
        mus is a list of the k marginal distributions
        problemname is a string naming the problem
        log_file is the name of the file to which the cost at each iteration and timings are saved.
        """

        mus = [np.asarray(x) for x in mus]
        self.mus = mus
        self.k = len(mus)
        self.ns = [len(mus[i]) for i in range(self.k)]
        self.cumn = np.cumsum(self.ns)
        self.cumn = np.insert(self.cumn[0:-1], 0,0)
        self.tot_n = sum(self.ns)
        self.problemname = problemname
        self._explicit_cost_tensor = None

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "                       0. ABSTRACT BOTTLENECKS                           "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def get_tuple_cost(self, tup):
        """
        Cost function for column generation. Takes in a k-tuple and returns the
        corresponding cost.
        """
        raise NotImplementedError()

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



    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "         1. COLUMN GENERATION AND NAIVE LP SOLVER IMPLEMENTATIONS        "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def solve_naive(self):
        """
        Solves using explicit LP solver.
        Requires get_tuple_cost to be implemented.
        """
        t0 = time.time()

        ## Gurobi to solve the dual problem.
        model = gp.Model(self.problemname)

        # flattened array of dual variables
        ps = model.addVars(self.tot_n, lb=-float('inf'),ub=float('inf'),name='p')
        # flattened mu vectors
        flatmus = list(np.concatenate(self.mus))
        flatmusdict = {i : flatmus[i] for i in range(len(flatmus))}
        model.setObjective(ps.prod(flatmusdict), GRB.MAXIMIZE)

        # Add every single possible constraint to the program
        tups = []
        tups_set = set()
        ranges = [range(i) for i in self.ns]
        print('Adding', itertools.product(*ranges),'constraints to the linear program.')
        i = 0
        for tup in itertools.product(*ranges):
            if i % 1000 == 0:
                print(str(i) + '/' + str(np.prod(self.ns)))
            self._add_tuple(tup,model,ps,tups,tups_set)
            i += 1

        tt0 = time.time()
        model.optimize()
        t1 = time.time()
        print('Naive: Total time',t1-t0)
        print('Time init', tt0 - t0)
        print('Time LP solve',t1 - tt0)

        return model.objVal, self._get_used_tuples_cg_naive(model,tups)

    def solve_cg(self, init_method=None):
        """
        Solves using column generation.
        Requires get_tuple_cost and get_cutting_plane to be implemented.
        """

        t0 = time.time()

        ## Gurobi to solve the dual problem with lazy constraint generation.
        model = gp.Model(self.problemname)

        # flattened array of dual variables
        ps = model.addVars(self.tot_n, lb=-float('inf'),ub=float('inf'),name='p')
        # flattened mu vectors
        flatmus = list(np.concatenate(self.mus))
        flatmusdict = {i : flatmus[i] for i in range(len(flatmus))}
        model.setObjective(ps.prod(flatmusdict), GRB.MAXIMIZE)

        tups = []
        tups_set = set()

        ### Initialize the first constraints of the model
        if init_method is None:
            tups_to_start = self._init_waterfilling()
        else:
            tups_to_start = init_method()
        for new_tup in tups_to_start:
            self._add_tuple(new_tup,model,ps,tups,tups_set)
        time_cutting_plane = 0
        time_solve = 0

        ## Iteratively solve the model and add constraints (lazy constraint generation)
        iter_num = 0
        tt0 = time.time()
        model.optimize()
        tt1 = time.time()
        time_solve += (tt1 - tt0)
        while True:
            iter_num += 1
            currps = [[ps[self.cumn[i] + j].x for j in range(self.ns[i])] for i in range(self.k)]

            tt0 = time.time()
            new_tups=self.get_cutting_plane(currps)
            tt1 = time.time()
            time_cutting_plane += (tt1 - tt0)

            filtered_new_tups = set(new_tups).difference(tups_set)
            if filtered_new_tups:
                for new_tup in filtered_new_tups:
                    self._add_tuple(new_tup,model,ps,tups,tups_set)
                tt0 = time.time()
                model.optimize()
                tt1 = time.time()
                time_solve += (tt1 - tt0)
            else:
                t1 = time.time()
                tot_time = t1 - t0
                print('total time',tot_time)
                print('time LP solve',time_solve)
                print('time cutting plane',time_cutting_plane)

                usedTupList = self._get_used_tuples_cg_naive(model,tups)
                return model.objVal, usedTupList


    def _add_tuple(self,tup,model,ps,tups,tups_set):
        # Helper method
        # Add a new constraint to the model, corresponding to the tuple tup.
        tups.append(tup)
        tups_set.add(tup)

        # \sum_{i \in [k]} p[i,tup[i]] <= tup_cost
        tup_cost = self.get_tuple_cost(tup)
        model.addConstr(sum([ps[self.cumn[i] + tup[i]] for i in range(self.k)]) <= tup_cost)

    def _init_waterfilling(self):
        print("Waterfilling initialization")
        k = len(self.mus)
        ns = [len(self.mus[i]) for i in range(k)]
        curr_tup_idx = []
        for i in range(k):
            curr_tup_idx.append(0)

        waterfillingschedule = []
        for i in range(k):
            currfill = 0
            for j in range(ns[i]):
                currfill += self.mus[i][j]
                waterfillingschedule.append((currfill,i,j))
        waterfillingschedule.sort()

        tup_init_list = []
        currmass = 0
        curr_indices = [0 for i in range(k)]
        tup_init_list.append(tuple(curr_indices))
        for i in range(len(waterfillingschedule)):
            curridx = waterfillingschedule[i][1]
            currreplacement = waterfillingschedule[i][2]
            curr_indices[curridx] = min(currreplacement + 1, ns[curridx] - 1)
            if i < len(waterfillingschedule) - 1:
                if waterfillingschedule[i+1][0] == waterfillingschedule[i][0]:
                    continue
            tup_init_list.append(tuple(curr_indices))

        tup_init_list = list(set(tup_init_list))
        return tup_init_list

    def _get_used_tuples_cg_naive(self, model, tups):
        usedTupList = []
        tupwts = list(model.Pi)
        for i in range(len(tups)):
            if tupwts[i] > 1e-10:
                usedTupList.append((tupwts[i],tups[i]))
        return usedTupList


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "               2. MULTIPLICATIVE WEIGHTS IMPLEMENTATION                  "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _get_tuple_cost_mw(self, tup):
        """
        Tuple cost function for multiplicative weights.
        Returns the cost normalized to be in the range [1,2].
        """
        cost_min, cost_max = self.get_cost_range()
        cst = self.get_tuple_cost(tup)
        return (cst - cost_min) / (cost_max - cost_min) + 1

    def solve_mw(self, eps, subroutine_eps):
        """
            Outputs an approximate solution using the multiplicative weights algorithm,
            given access to the MIN (or AMIN) oracle. Two tunable parameters: eps and lam.
            Also takes in a tunable "subroutine_eps" parameter.

            This needs get_tuple_cost and mw_cost_max to be specified.
            Or don't specify mw_cost_max, and just do exponential search instead.
        """
        # Binary search over \lambda \in [C_min, C_max], which is [1,2] wlog for our case.
        assert(subroutine_eps == 0)
        # This is currently only implemented for subroutine_eps = 0: i.e., the case in which we have an exact minimization oracle.

        lo = 1
        hi = 2
        sol = None
        bestlam = None
        while hi - lo > eps:
            lam = (lo+hi)/2
            print('lam', lam)

            currsol = self._solve_mw_subroutine(eps, lam)
            if currsol:
                """
                Round the solution to obtain an upper bound on lambda
                """
                roundedcurrsol = self.round_sparse(currsol)
                for i in range(self.k):
                    mi = self.marginalize_sparse(roundedcurrsol, i)
                    print(mi)
                    assert(np.all(np.isclose(mi, self.mus[i])))
                currsol_cost = self._get_sparse_sol_cost_mw(roundedcurrsol)
                print('currsol_cost mw',currsol_cost)
                hi = min(currsol_cost,lam)
                #
                # assert(False)
                # hi = lam
                sol = roundedcurrsol
                bestlam = currsol_cost
            else:
                lo = lam
        assert(sol is not None)
        return bestlam, sol

    def _solve_mw_subroutine(self, eps, lam):
        """
        The multiplicative weights algorithm specialized to MOT.
        """
        # lam is the solution value we hope for

        print('lam',lam)
        num_constraints = self.tot_n + 1
        N = np.log(num_constraints) / (2*eps)

        cost_min, cost_max = self.get_cost_range()

        mP = [0]
        CPoverlam = 0
        mOverMu = []
        for i in range(self.k):
            mOverMu.append(np.zeros(self.ns[i]))

        tup_to_cost = {}
        sol = {}
        constraint_scalings = np.concatenate([1 / self.mus[i] for i in range(self.k)] + [np.asarray([1/lam])])
        packing_constraints = np.zeros(1 + self.tot_n)

        cached_tup_array = np.zeros((0,1+self.tot_n))
        cached_tup_list = []

        def _get_jvec_cost(jvec):
            if jvec in tup_to_cost:
                jvec_cost = tup_to_cost[jvec]
            else:
                jvec_cost = self._get_tuple_cost_mw(jvec)
                tup_to_cost[jvec] = jvec_cost
            return jvec_cost


        def _add_jvec_mw(jvec):
            jvec_cost = _get_jvec_cost(jvec)
            increment = eps # \|C mvmt\|_{\infty} = mvmt * N
            increment = min(increment, eps * lam / jvec_cost) # \|P mvmt\|_{\infty} = score * mvmt / lam
            for i in range(self.k):
                increment = min(increment, eps * self.mus[i][jvec[i]])

            if jvec in sol:
                sol[jvec] += increment
            else:
                sol[jvec] = increment

            # keep track of the marginals and of the cost
            oldlogsumexp = scipy.special.logsumexp(packing_constraints)
            packing_constraints[-1] += increment * jvec_cost / lam # <C,P> / lambda
            for i in range(self.k):
                packing_constraints[self.cumn[i] + jvec[i]] += increment / self.mus[i][jvec[i]]
            mP[0] += increment

            newlogsumexp = scipy.special.logsumexp(packing_constraints)
            packlogsumexpchange = newlogsumexp - oldlogsumexp
            mP0change = increment
            changeratio =  packlogsumexpchange / mP0change

        def _compute_jvec_deriv_mw(tup,scaled_softmax_deriv):
            curr_deriv = 0
            for i in range(self.k):
                curr_deriv += scaled_softmax_deriv[self.cumn[i] + tup[i]]
            curr_deriv += _get_jvec_cost(tup) * scaled_softmax_deriv[-1]
            return curr_deriv

        def _get_cutting_plane_mw(scaled_softmax_deriv):

            r = scaled_softmax_deriv[-1]

            if r < 1e-8: # In order to avoid round-off errors, deal with this case separately.
                jvec = []
                for i in range(self.k):
                    jvec.append(np.argmin(scaled_softmax_deriv[self.cumn[i]:self.cumn[i] + self.ns[i]]))
                jvec = tuple(jvec)
                curr_deriv = _compute_jvec_deriv_mw(jvec, scaled_softmax_deriv)

                if curr_deriv <= 1 + eps:
                    return [jvec]
                else:
                    return []

            else: # Here, appeal to the cutting plane oracle.
                # First rescale the cost.
                scaled_softmax_deriv_rescaled = scaled_softmax_deriv * (cost_max - cost_min) / r
                dualwts = [-scaled_softmax_deriv_rescaled[self.cumn[i]:self.cumn[i] + self.ns[i]] for i in range(self.k)]
                dualwts[0] -= -(cost_max - cost_min) / r - cost_min + (cost_max - cost_min)

                tups_ret = self.get_cutting_plane(dualwts)
                return tups_ret

        cache_failed = True

        print('lam',lam)

        # If lambda is feasible, guaranteed a solution such that m(P) = 1 and
        # max (<C,P> / lambda, m_i(P)_j / mu_{ij}) <= ln(m+1)/N + (1+eps)^2


        improve_mask = []
        for i in range(self.k):
            improve_mask.append(np.zeros(self.ns[i]))
        iter_num = 0
        while mP[0] < N:
            iter_num += 1
            if mP[0] > 0 and np.max(packing_constraints / mP[0]) <= (1+ eps)**2:
                print(np.max(packing_constraints) / mP[0])
                # assert(False)
                return list([sol[i] / mP[0], i] for i in sol.keys())
            if iter_num % 1000 == 0:
                print('lambda',lam)
                print('mP / N',mP[0] / N)
                print('pack/cover',np.max(packing_constraints / mP[0]))

            softmax_deriv = scipy.special.softmax(packing_constraints)
            scaled_softmax_deriv = softmax_deriv * constraint_scalings

            if not cache_failed:
                improvements = cached_tup_array @ scaled_softmax_deriv

                if np.any(improvements <= 1): # + eps


                    imp_idx = np.argmin(improvements)
                    _add_jvec_mw(cached_tup_list[imp_idx])

                    cache_failed = False
                else:
                    cache_failed = True
                continue


            print('lambda',lam)
            print('Cache failed',cache_failed)
            print('Cache size',len(tup_to_cost.keys()))
            cache_failed = False
            # Try to find a new tuple that works.

            tups_ret = _get_cutting_plane_mw(scaled_softmax_deriv)
            for tup in tups_ret:
                tup_cost = _get_jvec_cost(tup)
                curr_row = np.zeros(len(packing_constraints))
                curr_row[-1] = tup_cost
                for i in range(self.k):
                    curr_row[self.cumn[i] + tup[i]] = 1

                cached_tup_array = np.vstack([cached_tup_array, curr_row])
                cached_tup_list.append(tup)

            if not tups_ret:
                return None

        return list((sol[i] / mP[0], i) for i in sol.keys())

    def _get_sparse_sol_cost_mw(self, sol):
        cst = 0
        for pi, i in sol:
            cst += self._get_tuple_cost_mw(i) * pi
        return cst


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "                    3. SINKHORN IMPLEMENTATION                           "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def solve_sinkhorn(self, eta, tol, method='cyclic',naive=False):
        """
        eta is regularization parameter
        tol is the error tolerance for stopping
        method is "greedy", "random", or "cyclic"
        """

        p = [np.zeros(self.ns[i]) for i in range(self.k)]

        assert(method in ['greedy','random','cyclic'])

        curriter = -1
        while True:
            curriter += 1
            if curriter % self.k == 0:
                """ Every k iterations check whether we should exit """
                (worsti,worsterr) = self._sinkhorn_worst_error(eta, p, naive)

                if worsterr < tol:
                    break
            if curriter % 1000 == 0:
                print('Iteration ',curriter)
                print('worsterr',worsterr)

            """ Pick scaling index """
            scalei = -1

            if method == 'greedy':
                """ Greedy selection rule """
                (worsti, worsterr) = self._sinkhorn_worst_error(eta, p, naive)
                print('worsterr',worsterr)
                if worsterr < tol:
                    continue
                scalei = worsti

            elif method == 'random':
                """ Random selection rule """
                scalei = random.randint(0,self.k-1)

            elif method == 'cyclic':
                """ Cyclic selection rule """
                scalei = curriter % self.k

            else:
                assert(False)

            """ Rescale weights on scalei """

            if naive:
                mi = self.marginalize_naive(eta, p, scalei)
            else:
                mi = self.marginalize(eta, p, scalei)
            ratio = self.mus[scalei] / mi
            p[scalei] = p[scalei] + np.log(ratio) / eta

        p_rounded, rankone = self._round_sinkhorn(eta, p, naive)

        return p_rounded, rankone

    def marginalize_naive(self, eta, p, i):
        # assert(False)
        """
        NAIVE, BRUTE FORCE IMPLEMENTATION OF THE FOLLOWING:
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-\eta C].
        Let d_i = \exp[\eta p_i].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_i(P).
        """

        scaled_cost_tensor = copy.deepcopy(self._get_explicit_cost_tensor())
        scaled_cost_tensor = np.exp(-eta * scaled_cost_tensor)

        for scale_axis in range(self.k):
            dim_array = np.ones((1,scaled_cost_tensor.ndim),int).ravel()
            dim_array[scale_axis] = -1
            p_scale_reshaped = p[scale_axis].reshape(dim_array)
            p_scaling = np.exp(eta * p_scale_reshaped)
            # print(p_scaling.shape)
            scaled_cost_tensor = scaled_cost_tensor * p_scaling
            # print(scaled_cost_tensor)
        mi = np.apply_over_axes(np.sum, scaled_cost_tensor, [j for j in range(self.k) if j != i])
        mi = mi.flatten()
        return mi

    def _sinkhorn_worst_error(self, eta, p, naive=False):
        """
        Compute the worst error of any marginal. Used for the termination
        condition of solve_sinkhorn.
        """
        worsti = -1
        worsterr = -1
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(eta, p, i)
            else:
                mi = self.marginalize(eta, p, i)
            erri = np.sum(np.abs(mi - self.mus[i]))
            if erri > worsterr:
                worsti = i
                worsterr = erri
        return (worsti, worsterr)


    def _round_sinkhorn(self, eta, p, naive=False):
        """ Round sinkhorn solution with a rank-one perturbation """
        p = copy.deepcopy(p)
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(eta, p, i)
            else:
                mi = self.marginalize(eta, p, i)
            badratio = self.mus[i] / mi
            minbadratio = np.minimum(1, badratio)
            p[i] = p[i] + np.log(minbadratio) / eta

        rankone = []
        for i in range(self.k):
            if naive:
                mi = self.marginalize_naive(eta, p, i)
            else:
                mi = self.marginalize(eta, p, i)
            erri = self.mus[i] - mi
            assert(np.all(erri >= -1e-10))
            erri = np.maximum(0,erri)
            if i > 0 and np.sum(np.abs(erri)) > 1e-8:
                rankone.append(erri /  np.sum(np.abs(erri)))
            else:
                rankone.append(erri)

        return p, rankone

    def _get_explicit_cost_tensor(self):
        """
        Compute all cost entries and return an explicit tensor. This is used
        in naive Sinkhorn implementation.
        """
        if self._explicit_cost_tensor is not None:
            return self._explicit_cost_tensor

        self._explicit_cost_tensor = np.zeros(self.ns)
        ranges = [range(n) for n in self.ns]
        for tup in itertools.product(*ranges):
            self._explicit_cost_tensor[tup] = self.get_tuple_cost(tup)

        return self._explicit_cost_tensor

    def _delete_explicit_cost_tensor(self):
        """ To clear up memory, if the cost tensor has been stored in memory,
            free it with this method.
        """
        self._explicit_cost_tensor = None


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "      4. UTILITY METHODS FOR SOLUTIONS REPRESENTED AS SPARSE TENSOR      "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def round_sparse(self, sol):

        max_rat = 1
        for i in range(self.k):
            mi = self.marginalize_sparse(sol, i)
            max_rat = max(max_rat, np.max(mi / self.mus[i]))
        sol = [(ptup / max_rat, tup) for (ptup, tup) in sol]

        currm = []
        for i in range(self.k):
            currm.append(self.marginalize_sparse(sol, i))

        currindices = []
        for i in range(self.k):
            currindices.append(0)
        while True:
            for i in range(self.k):
                while currm[i][currindices[i]] >= self.mus[i][currindices[i]] - 1e-10:
                    currindices[i] += 1
                    if currindices[i] == self.ns[i]:
                        # print(currm)
                        for j in range(self.k):
                            assert(abs(np.sum(currm[j]) - 1) < 1e-10)
                        return sol
            currinc = 1
            # print(currindices)
            for i in range(self.k):
                currinc = min(currinc, self.mus[i][currindices[i]] - currm[i][currindices[i]])

            assert(currinc > 0)
            for i in range(self.k):
                currm[i][currindices[i]] += currinc
            sol.append((currinc, tuple([currindices[i] for i in range(self.k)])))
        assert(False)

    def marginalize_sparse(self, sol, i):
        mi = np.zeros(self.ns[i])
        for ptup, tup in sol:
            mi[tup[i]] += ptup
        return mi

    def get_sparse_sol_cost(self, sol):
        cst = 0
        for pi, i in sol:
            cst += self.get_tuple_cost(i) * pi
        return cst


    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    " 5. UTILITY METHODS FOR SINKHORN SOLUTIONS REPRESENTED AS SCALINGS + RANK-ONE "
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def sample_from_rounded_sinkhorn_solution(self, eta, p, rankone,naive=False):
        rankonel1 = 1
        for i in range(self.k):
            assert(np.all(rankone[i] >= 0))
            rankonel1 *= np.sum(rankone[i])

        def _sample_from_unnormalized(prob_dist):
            curr_seed = random.random() * np.sum(prob_dist)
            cumsum = 0
            for j in range(len(prob_dist)):
                cumsum += prob_dist[j]
                if curr_seed < cumsum:
                    return j
            assert(False)

        if random.random() < rankonel1:
            """ Sample from the rank-one term. """
            currtup = []
            for i in range(self.k):
                j = _sample_from_unnormalized(rankone[i])
                currtup.append(j)
            assert(len(currtup) == self.k)
            return tuple(currtup)

        else:
            """ Sample from the diagonally-scaled cost. """
            currtup = []
            currscalings = [copy.deepcopy(pi) for pi in p]
            for i in range(self.k):
                if naive:
                    micond = self.marginalize_naive(eta, currscalings, i)
                else:
                    micond = self.marginalize(eta, currscalings, i)
                j = _sample_from_unnormalized(micond)
                currtup.append(j)
                currscalings[i] = np.ones(len(currscalings[i])) * -np.inf
                currscalings[i][j] = p[i][j]

            assert(len(currtup) == self.k)
            return tuple(currtup)


    def estimate_rounded_sinkhorn_solution_cost(self, eta, p, rankone, num_trials, naive):
        # Determine the cost of a Sinkhorn solution by randomly sampling a total
        # of num_trials times.
        # This works since the cost is bounded.
        csts = []
        for i in range(num_trials):
            if i % 10 == 0:
                print(i)
            tup = self.sample_from_rounded_sinkhorn_solution(eta, p, rankone, naive)
            csts.append(self.get_tuple_cost(tup))
        csts = np.asarray(csts)

        conf_interval = scipy.stats.t.interval(0.95, len(csts)-1, loc=np.mean(csts), scale=scipy.stats.sem(csts))
        return (np.mean(csts), conf_interval)

    def compute_rounded_sinkhorn_solution_cost_naive(self, eta, p, rankone):
        ranges = [range(n) for n in self.ns]
        curriter = 0
        totprob = 0
        totcost = 0
        for tup in itertools.product(*ranges):
            weighting = np.sum([p[j][tup[j]] for j in range(self.k)])

            currcost = self.get_tuple_cost(tup)
            currprob = np.exp(eta * weighting - eta * currcost)
            rankonecurrprob = np.prod([rankone[i][tup[i]] for i in range(self.k)])
            currprob += rankonecurrprob
            totprob += currprob
            totcost += currcost * currprob
        assert(abs(totprob - 1) < 1e-5)
        return totcost

    def get_pairwise_marginal(self, eta, p, rankone, i1, i2):
        pcopy = copy.deepcopy(p)
        marg2 = np.zeros((self.ns[i1],self.ns[i2]))
        # print(np.sum(m1))
        # print(np.prod([np.sum(rankone[i]) for i in range(self.k)]))
        # assert(False)
        for j1 in range(self.ns[i1]):
            pcopy[i1] = np.ones(self.ns[i1]) * -math.inf
            pcopy[i1][j1] = p[i1][j1]
            m2 = self.marginalize(eta, pcopy, i2)
            marg2[j1,:] = m2
        rankonefactor = 1
        for i in range(self.k):
            if i not in [i1, i2]:
                rankonefactor *= np.sum(rankone[i])
        for j1 in range(self.ns[i1]):
            for j2 in range(self.ns[i2]):
                marg2[j1,j2] += rankone[i1][j1] * rankone[i2][j2] * rankonefactor
        return marg2
