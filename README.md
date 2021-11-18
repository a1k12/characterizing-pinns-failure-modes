# Characterizing possible failure modes in physics-informed neural networks

This repository contains the PyTorch source code for the experiments in the manuscript:

[Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney. Characterizing possible failure modes in physics-informed neural networks.](https://arxiv.org/abs/2109.01050), Neural Information Processing Systems (NeurIPS) 2021.

## Introduction

Recent work in scientific machine learning has developed so-called physics-informed neural network (PINN) models. The typical approach is to incorporate physical domain knowledge as soft constraints on an empirical loss function and use existing machine learning methodologies to train the model. We demonstrate that, while existing PINN methodologies can learn good models for relatively trivial problems, they can easily fail to learn relevant physical phenomena even for simple PDEs. In particular, we analyze several distinct situations of widespread physical interest, including learning differential equations with convection, reaction, and diffusion operators. We provide evidence that the soft regularization in PINNs, which involves differential operators, can introduce a number of subtle problems, including making the problem ill-conditioned. Importantly, we show that these possible failure modes are not due to the lack of expressivity in the NN architecture, but that the PINN's setup makes the loss landscape very hard to optimize. We then describe two promising solutions to address these failure modes. The first approach is to use curriculum regularization, where the PINN's loss term starts from a simple PDE regularization, and becomes progressively more complex as the NN gets trained. The second approach is to pose the problem as a sequence-to-sequence learning task, rather than learning to predict the entire space-time at once. Extensive testing shows that we can achieve up to 1-2 orders of magnitude lower error with these methods as compared to regular PINN training.

## Installation

Installation of all necessary packages can either be done via `poetry` or through requirements.txt. For example:

```
git clone git@github.com:a1k12/characterizing-pinns-failure-modes.git
cd characterizing-pinns-failure-modes
pip install .
```

## Instructions

To run the code for the convection, diffusion, reaction, or reaction-diffusion ('rd') systems with periodic boundary conditions, the following can be run within the 'pbc_examples' folder.

```
python main_pbc.py [--system] [--seed] [--N_f] [--optimizer_name] [--lr] [--L] [--xgrid] [--nu] [--rho] [--beta] [--u0_str] [--layers] [--net] [--activation] [--loss_style] [--visualize] [--save_model]

Possible arguments:
--system            system of study (default: convection; also supports diffusion, reaction, rd)
--seed              used to reproduce the results (default: 0)
--N_f               number of points to sample from the interior domain (default: 1000)
--optimizer_name    optimizer to use, currently supports L-BFGS
--lr                learning rate (default: 1.0)
--L                 multiplier on the regularization parameter (default: 1.0)
--xgrid             size of the xgrid (default: 256)
--nu                viscosity coefficient for diffusion
--rho               reaction coefficient
--beta              speed of propagation for convection
--u0_str            initial condition (default: 'sin(x)'; also supports 'gauss' for reaction/reaction-diffusion)
--layers            number of layers in the network (default: '50,50,50,50,1')
--net               net architecture (default: 'DNN')
--activation        activation for the network (default: 'tanh')
--loss_style        loss function style (default: 'mse')
--visualize         option to visualize the solution (default: False)
--save_model        option to save the model (default: False)
```

## Citation
This repository has been developed as part of the following paper. We would appreciate it if you would please cite the following paper if you found the library useful for your work:

```text
@article{krishnapriyan2021characterizing,
  title={Characterizing possible failure modes in physics-informed neural networks},
  author={Krishnapriyan, Aditi S. and Gholami, Amir and Zhe, Shandian and Kirby, Robert and Mahoney, Michael W},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
