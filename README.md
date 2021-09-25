## _Polynomial-time algorithms for Multimarginal Optimal Transport Problems with structure_

This repository contains the code from [Altschuler, Boix-Adsera](https://arxiv.org/abs/2008.03006) for solving structured MOT problems.

We implement the following algorithms:
* naive LP solver
* Sinkhorn (both naive and efficient)
* Multiplicative Weights Update (MWU)
* Column Generation (COLGEN)

We provide code for the following MOT applications:
* generalized Euler flows
* adversarial network reliability
* risk estimation

NB: Our focus in writing this code was clarity over speed. Much further optimization can certainly be done.

## Installation

1. Clone the Github repository.
<!-- ```
git clone https://github.com/eboix/mot
``` -->

2. Install Gurobi and its Python extension, `gurobipy` (https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_python.html)

3. Install disjoint set:
`python -m pip install disjoint-set`

4. To replicate the fluid dynamics plots in the paper, run `generalized_euler_flows.py`.
The IPython notebook `generate_plots.ipynb` contains instructions to replicate our scalability and accuracy comparisons on the network reliability and risk estimation problems. 

## License

This project is licensed under the LGPL-3 license. See the [LICENSE.md](LICENSE.md) file for details.
