

## Installation
To install the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/WCR_Levy_pub.git

2. Add path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/Path/to/Levy_wcr

## Explaination

All files starting with "generate" are used to generate data, using the discrete format of SDE; Files starting with "weak" are generally the files you need to run. These files are very simple scripts. You only need to go to the corresponding dimension and setting files to run them directly; Folders are first classified according to the dimensions of the problem. Each d-dimensional folder contains examples of each dimension couple and each dimension independently;

Adjustable parameters include but are not limited to: sample (indicates the number of samples), gauss_variance, gauss_samp_number, lam, STRidge_threshold, lhs_ratio;

The input $x$ is a three-dimensional tentor with shape [num_of_timesteps, num_of_traj, num_of_problem]; 

Other parameters: 

basis_order = N means to expand with N-order polynomial
basis_number indicates how many terms there are when expanding, for example, in one dimension, basis number = basis order; in two dimensions, basis order = 2, basis number = 6 (1, x, y, x^2, y^2, xy)


## Citation
```bibtex
@article{guo2024weak,
  title={Weak Collocation Regression for Inferring Stochastic Dynamics with L$\backslash$'$\{$e$\}$ vy Noise},
  author={Guo, Liya and Lu, Liwei and Zeng, Zhijun and Hu, Pipi and Zhu, Yi},
  journal={arXiv preprint arXiv:2403.08292},
  year={2024}
  }