# Code for _Polynomial-time algorithms for Multimarginal Optimal Transport Problems with structure_

This repository contains the code from [Altschuler, Boix-Adsera](https://arxiv.org/abs/2008.03006) for solving structured MOT problems.
Instead of using the Ellipsoid method, we use Column generation in our practical implementations, as described in the paper.

NB: Our focus in writing this code was clarity over speed. Much further optimization can certainly be done.

## Installation

1. Clone the Github repository.
<!-- ```
git clone https://github.com/eboix/[TODO]
``` -->

2. Using the [Anaconda](https://docs.anaconda.com/anaconda/install/) package manager, run:

```
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
```

3. Install [PuLP](https://github.com/coin-or/pulp):
`python -m pip install pulp`

4. Install disjoint set:
`python -m pip install disjoint-set`

<!-- 4. `example_barycenter_computation.py` provides example usage of our code. [TODO]-->

## License

This project is licensed under the LGPL-3 license. See the [LICENSE.md](LICENSE.md) file for details.
