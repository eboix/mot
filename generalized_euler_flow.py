from MOT_algs import MOTProblem
from functools import partial
import itertools
import numpy as np
from collections import Counter
import time
import math


class GeneralizedEulerFlowProblem(MOTProblem):
    def __init__(self, sigma, k, problemname='Generalized_Euler_Flow_Problem'):
        # sigma is a permutation of range(n)
        # k is the number of discretization timesteps
        self.k = k
        self.n = len(sigma)
        self.sigma = sigma
        self.cost_range = (0,self.k)
        # Sanity checks
        assert(k >= 2)
        # assert(k >= 3), "k should be at least 3"
        assert(Counter(sigma) == Counter(range(self.n))), "sigma should be a permutation of range(n)"

        self.sigma = np.asarray(sigma,dtype=int)

        mus = []
        for i in range(k):
            mus.append(np.ones(self.n) / self.n)

        n = self.n
        self.unreg_cost = np.fromfunction(lambda i, j: (i-j)**2 / n**2, (n,n), dtype=np.float64)
        self.unreg_loop_cost = np.zeros((n,n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                self.unreg_loop_cost[i,j] = (self.sigma[j] - i)**2 / n**2

        super(GeneralizedEulerFlowProblem, self).__init__(mus,problemname)

    def get_tuple_cost(self, tup):
        cst = 0
        for i in range(self.k-1):
            cst += (tup[i] - tup[i+1])**2
        cst += (self.sigma[tup[0]] - tup[self.k-1])**2
        return cst / self.n**2

    def get_cost_range(self):
        return self.cost_range

    def get_cutting_plane(self, dualwts):
        # This is a O(n^3 k)-time implementation.
        # There is a O(n^2 k)-time algorithm due to Eppstein et al. [FOCS'88], but this was faster
        # for moderate-size inputs than our implementation of that algorithm.

        # dualwts is a length-k list of length-n lists
        t0 = time.time()
        # For ease, convert weights to a n x k numpy array. Sigma is a length-n numpy array.
        wts = -np.asarray(dualwts).T
        sigma = self.sigma
        n = wts.shape[0]
        k = wts.shape[1]
        assert(k >= 3)
        assert(sigma.shape[0] == wts.shape[0])

        viol_tuples = []
        worst_error = 0
        dist_matrix = np.fromfunction(lambda i, j: (i-j)**2 / n**2, (n,n))

        for j0 in range(n):
            curr_dists = wts[j0,0] + wts[:,1] + (np.square(np.arange(n) - j0) / n**2) ## Initialize the distances to each j1, given the value of the first index, j0.

            # Store pointers to the best parent in this list
            parent_inds = []

            for i in range(2,k):
                # Compute the matrix move_{a,b} = curr_dists_{b} + dist_{b,a}
                # this matrix computes the cost of moving from b = j_{i-1} to a = j_i, taking
                # into account the weights up to j_{i-1}
                move = curr_dists + dist_matrix


                # compute the minimum distance from j0 to ji, for each ji,
                # taking into account the weights up to j_{i-1}
                new_dists = np.amin(move, axis=1)
                curr_dists = new_dists + wts[:,i]

                # store the pointer from each ji to the best j_{i-1}
                dist_inds = np.argmin(move, axis=1)
                parent_inds.append(dist_inds)

            # Close the loop by adding the cost from j_{k-1} to sigma[j0]
            curr_dists = curr_dists + (np.arange(n) - sigma[j0])**2 / n**2

            # Extract all violated tuples:
            worst_error = min(worst_error, np.min(curr_dists))
            viol_locs = np.where(curr_dists < 0)[0]

            if len(viol_locs) > 0:
                viol_locs = np.argmin(curr_dists)
                viol_locs = np.asarray([viol_locs])
                curr_viol_tup_list = []
                curr_viol_tup_list.append(viol_locs)
                curr_viol_locs = viol_locs
                for i in range(len(parent_inds)):
                    currp_inds = parent_inds[len(parent_inds)-i-1]
                    curr_viol_locs = currp_inds[curr_viol_locs]
                    curr_viol_tup_list.append(curr_viol_locs)
                curr_viol_tup_list.append([j0]*len(viol_locs))

                curr_viol_tup_list.reverse()
                new_tup_list = [tuple(x) for x in np.asarray(curr_viol_tup_list).T.tolist()]
                viol_tuples.extend(new_tup_list)

        t1 = time.time()
        print('Time elapsed: ', t1-t0)
        print('WORST_ERROR_times_n: ', worst_error*n)
        if worst_error*n > -1e-7: # The solution will be optimal up to 1e-7 additive error. <-- check this again
            return []
        # worst_tup = None
        # worst_cst = 1
        # for tup in viol_tuples:
        #     cst = 0
        #     for i in range(k-1):
        #         cst += (tup[i] - tup[i+1])**2
        #         cst += wts[tup[i],i]
        #     cst += (tup[0] - sigma[tup[k-1]])**2
        #     cst += wts[tup[k-1],k-1]
        #     if cst < worst_cst:
        #         worst_cst = cst
        #         worst_tup = tup

        # return [worst_tup]
        return viol_tuples

    ################################## Extract pairwise maps from joint distribution given as a sparse solution
    def extract_pw_map(sol,n,i1,i2):
        otmap = np.zeros((n,n))
        for val, tup in sol:
            otmap[tup[i1],tup[i2]] += val
        return otmap

    def marginalize(self, eta, p, margidx):
        """
        Given weights p = [p_1,\ldots,p_k], and regularization eta > 0,
        Let K = \exp[-C].
        Let d_i = \exp[\eta p_i] for all i \in [k].
        Let P = (d_1 \otimes \dots \otimes d_k) \odot K.
        Return m_{margidx}(P).
        """

        n = self.n
        reg_cost = np.exp(-eta * self.unreg_cost)
        reg_loop_cost = np.exp(-eta * self.unreg_loop_cost)

        """ Compute transition matrices 1 --> margidx, margidx --> k """
        trans1 = np.eye(n)
        transk = np.eye(n)

        # trans1[i,j] is the mass of cost of transitioning from i at time 1 to j at time margidx
        # include the potentials in [1,margidx)
        for i in range(margidx):
            currcolscaling = np.diag(np.exp(eta * p[i]))
            trans1 = trans1 @ currcolscaling
            trans1 = trans1 @ reg_cost

        # transk[i,j] is the mass of the cost of transitioning from i at time margidx to j at time k
        # include the potentials in (margidx,k]
        for i in range(margidx+1,self.k):
            transk = transk @ reg_cost
            currcolscaling = np.diag(np.exp(eta * p[i]))
            transk = transk @ currcolscaling

        # print('trans1',trans1)
        # print('transk',transk)

        # transk1[i,j] is the mass of the cost of transitioning from i at time margidx to j at time 1 (going through time k and looping back to time 1)
        # includes the potentials in (margidx,k]
        transk1 = transk @ reg_loop_cost

        # For each fixed l, compute trans1[:,l] \dot transk1[l,:].
        # This marginalizes over time 1, but still does not compute the potentials at margidx.
        notscaled = np.diag(transk1 @ trans1)


        scaled = notscaled * np.exp(eta * p[margidx])
        # comparison = self.marginalize_naive(eta, p, margidx)
        # assert(np.all(np.isclose(scaled, comparison)))

        return scaled

    def get_sinkhorn_objective(self, eta, p, rankone):
        """
        Compute cost of the Sinkhorn solution efficiently, using the
        fact that the treewidth of the costs is small. Here, we tailor
        to the generalized euler flow problem.
        """
        obj_sinkhorn = 0
        # print('rankone size',[np.sum(np.abs(rankone[i])) for i in range(self.k)])
        for i in range(self.k-1):
            otmap = self.get_pairwise_marginal(eta,p,rankone,i,i+1)
            assert(np.all(otmap >= 0))
            print(np.sum(otmap))
            assert(np.abs(1 - np.sum(otmap)) < 1e-8)
            pw_cost = np.sum(otmap * self.unreg_cost)
            print(pw_cost)
            obj_sinkhorn += pw_cost

        otmap = self.get_pairwise_marginal(eta,p,rankone,self.k-1,0)
        assert(np.abs(1 - np.sum(otmap)) < 1e-8)
        pw_cost = np.sum(otmap * self.unreg_loop_cost)
        print(pw_cost)
        obj_sinkhorn += pw_cost
        print(obj_sinkhorn)
        # assert(False)
        return obj_sinkhorn


from PIL import Image, ImageDraw
import aggdraw


def draw_trajectory_diagram(otmaps):

    # d = aggdraw.Draw(im)
    # p = aggdraw.Pen("black", 0.5)
    # d.line((0, 0, 500, 500), p)
    # d.flush()

    # List of otmaps from i to i+1
    # Draw

    k = len(otmaps)
    n = otmaps[0].shape[0]


    w, h = 600, 400
    img = Image.new("RGB", (w, h), color='white')
    img1 = aggdraw.Draw(img)
    pen = aggdraw.Pen("#5a86af")
    brush = aggdraw.Brush("#5a86af")

    def pos_ij(i,j):
        x = 0.025 * w + 0.95 * w * j / (n-1)
        y = 0.025 * h + 0.95 * h * i / k
        return (x,y)

    for i in range(k+1):
        for j in range(n):
            x,y = pos_ij(i,j)
            r = 3
            img1.ellipse((x-r,y-r,x+r,y+r), pen, brush)


    for i in range(k):
        for j in range(n):
            for jto in range(n):
                # print(np.max(otmaps[i][j]) * n)
                pen = aggdraw.Pen("black", opacity = int(255 * otmaps[i][j][jto] * n))
                line_coords = pos_ij(i,j) + pos_ij(i+1,jto)
                # if otmaps[i][j][jto] > 0.5:
                img1.line(line_coords, pen)
    img1.flush()
    img.show()


import matplotlib.pyplot as plt

def main():

    ##########################################################################
    ## A) Generate sigma test
    n_list = [51]
    k_list = [6]
    eta_list = [2000]
    mw_eps_list = []
    do_colgen = False
    do_naive = False
    do_traj_diagram = True
    do_maps_diagram = False

    timing_data = []

    for n, k in itertools.product(n_list,k_list):
        print('n',n,'k',k)
        assert(n % 2 == 1)
        sigma_list = []
        # sigma_list.append(range(n))
        # sigma_list.append([i for i in range(n-1,-1,-1)]) #inverted permutation
        sigma_list.append([(i + n // 2) % n for i in range(n)])
        # sigma_list.append([min(2*i,2*n-2*i-1) for i in range(n)])

        for sigma_idx in range(len(sigma_list)):
            sigma = sigma_list[sigma_idx]
            print(sigma_idx)
            prob=GeneralizedEulerFlowProblem(sigma,k)

            print('timing_data = ', timing_data)

            problem_name = 'n' + str(n) + '_k' + str(k) + '_sigma' + str(sigma_idx) + '_'

            """ Sinkhorn """
            for eta in eta_list:
                starttime = time.time()
                p, rankone = prob.solve_sinkhorn(eta=eta, tol=0.001, method='cyclic', naive=False)
                print('p',p)
                endtime = time.time()
                time_sinkhorn = endtime-starttime
                obj_sinkhorn = prob.get_sinkhorn_objective(eta, p, rankone)
                print('Total time sinkhorn:',time_sinkhorn)
                print('Unregularized objective sinkhorn:',obj_sinkhorn)

                method_name = 'sink_eta' + str(eta)

                if do_maps_diagram:
                    for i in range(0,k):
                        print('Extracting map')
                        otmap = prob.get_pairwise_marginal(eta,p,rankone,i,0)
                        print('Map extracted')
                        plt.imshow(1-otmap, cmap='gray', origin='lower')
                        plt.gca().axes.get_yaxis().set_visible(False)
                        plt.gca().axes.get_xaxis().set_visible(False)
                        plt.show()
                if do_traj_diagram:
                    otmaps = []
                    for i in range(0,k-1):
                        print('Extracting map')
                        otmaps.append(prob.get_pairwise_marginal(eta, p, rankone,i,i+1))
                        print('Map extracted')

                    final_map_pre = prob.get_pairwise_marginal(eta, p, rankone,k-1,0)
                    final_map = np.zeros(final_map_pre.shape)
                    for jkm1 in range(n):
                        for j0 in range(n):
                            final_map[jkm1][sigma[j0]] = final_map_pre[jkm1][j0]
                    otmaps.append(final_map)

                        # plt.imshow(1-otmaps[i], cmap='gray', origin='lower')
                        # plt.show()
                    draw_trajectory_diagram(otmaps)



            """ Multiplicative weights """
            for mw_eps in mw_eps_list:
                starttime = time.time()
                obj_mw, sol_mw = prob.solve_mw(eps=mw_eps,subroutine_eps=0)
                endtime = time.time()
                time_mw = endtime - starttime
                print('Total time mw:',time_mw)
                print('Objective (MW, rescaled):',obj_mw)
                print('Objective MW', prob.get_sparse_sol_cost(sol_mw))
                print(sol_mw)

                for i in range(0,k):
                    print('Extracting map')
                    otmap = GeneralizedEulerFlowProblem.extract_pw_map(sol_mw,n,i,0)
                    print('Map extracted')
                    plt.imshow(1-otmap, cmap='gray', origin='lower')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.show()


            if do_colgen:
                """ Column generation """
                starttime = time.time()
                obj_cg, sol_cg = prob.solve_cg()
                endtime = time.time()
                time_cg = endtime-starttime
                print('Total time cg:',time_cg)
                print('Objective cg:',obj_cg)


                if do_maps_diagram:
                    for i in range(0,k):
                        print('Extracting map')
                        otmap = GeneralizedEulerFlowProblem.extract_pw_map(sol_cg,n,i,0)
                        print('Map extracted')
                        plt.imshow(1-otmap, cmap='gray', origin='lower')
                        plt.gca().axes.get_yaxis().set_visible(False)
                        plt.gca().axes.get_xaxis().set_visible(False)
                        plt.show()
                    print('rel error sinkhorn', (obj_sinkhorn - obj_cg)/obj_cg)

                if do_traj_diagram:
                    otmaps = []
                    for i in range(0,k-1):
                        otmaps.append(GeneralizedEulerFlowProblem.extract_pw_map(sol_cg,n,i,i+1))
                    final_map_pre = GeneralizedEulerFlowProblem.extract_pw_map(sol_cg,n,k-1,0)
                    final_map = np.zeros(final_map_pre.shape)
                    for jkm1 in range(n):
                        for j0 in range(n):
                            final_map[jkm1][sigma[j0]] = final_map_pre[jkm1][j0]
                    otmaps.append(final_map)
                        # plt.imshow(1-otmaps[i], cmap='gray', origin='lower')
                        # plt.show()
                    draw_trajectory_diagram(otmaps)


            if do_naive:
                starttime = time.time()
                obj_naive, sol_naive = prob.solve_naive()
                endtime = time.time()
                time_naive = endtime-starttime
                print('Total time naive:',time_naive)
                print('Objective naive:',obj_naive)
                for i in range(0,k):
                    print('Extracting map')
                    otmap = GeneralizedEulerFlowProblem.extract_pw_map(sol_naive,n,i,0)
                    print('Map extracted')
                    plt.imshow(1-otmap, cmap='gray', origin='lower')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.show()


    print('timing_data = ', timing_data)


if __name__ == '__main__':
    main()
