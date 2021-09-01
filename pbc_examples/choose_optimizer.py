"""Optimizer choices."""

import torch
import torch_optimizer as optim
import numpy as np

def choose_optimizer(optimizer_name: str, *params):
    if optimizer_name == 'LBFGS':
        return LBFGS(*params)
    elif optimizer_name == 'AdaHessian':
        return AdaHessian(*params)
    elif optimizer_name == 'Shampoo':
        return Shampoo(*params)
    elif optimizer_name == 'Yogi':
        return Yogi(*params)
    elif optimizer_name == 'Apollo':
        return Apollo(*params)
    elif optimizer_name == 'Adam':
        return Adam(*params)
    elif optimizer_name == 'SGD':
        return SGD(*params)

def LBFGS(model_param,
        lr=1.0,
        max_iter=100000,
        max_eval=None,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-7,
        line_search_fn="strong_wolfe"):

    optimizer = torch.optim.LBFGS(
        model_param,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn
        )

    return optimizer

def Adam(model_param, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):

    optimizer = torch.optim.Adam(
                model_param,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad
    )
    return optimizer

def SGD(model_param, lr=1e-4, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):

    optimizer = torch.optim.SGD(
                model_param,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=False
    )

    return optimizer

def AdaHessian(model_param, lr=1.0, betas=(0.9, 0.999),
                eps=1e-4, weight_decay=0.0, hessian_power=0.5):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 0.5)
    """

    optimizer = Adahessian(model_param,
                            lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            hessian_power=hessian_power,
                            single_gpu=False)

    return optimizer

def Shampoo(model_param, lr=1e-1, momentum=0.0, weight_decay=0.0,
            epsilon=1e-4, update_freq=1):
    """
    Args:
        params: params of model
        lr: learning rate
        momentum: momentum factor
        weight_decay: weight decay (L2 penalty)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
        update_freq: update frequency to compute inverse
    """
    optimizer = optim.Shampoo(model_param,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            epsilon=epsilon,
                            update_freq=update_freq)

    return optimizer

def Yogi(model_param, lr=1e-2, betas=(0.9, 0.999), eps=1e-3, initial_accumulator=1e-6,
        weight_decay=0):

    optimizer = optim.Yogi(model_param,
                            lr=lr,
                            betas=betas,
                            eps=eps,
                            initial_accumulator=initial_accumulator,
                            weight_decay=weight_decay)

    return optimizer

def Apollo(model_param, lr=1e-2, beta=0.9, eps=1e-4, warmup=5, init_lr=0.01, weight_decay=0):
    """Apollo already includes warmup!

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        beta: coefficient used for computing
            running averages of gradient (default: 0.9)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-4)
        warmup: number of warmup steps (default: 5)
        init_lr: initial learning rate for warmup (default: 0.01)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """

    optimizer = optim.Apollo(model_param,
                            lr=lr,
                            beta=beta,
                            eps=eps,
                            warmup=warmup,
                            init_lr=init_lr,
                            weight_decay=weight_decay)
    return optimizer
