# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is from SimMIM.
# (https://github.com/microsoft/SimMIM)
# ------------------------------------------------------------------------------

import json
import torch
import torch.optim as optim

def build_optimizer(module, cfg):
    """Build an optimizer from a config dict.

    Args:
        module (:obj:`nn.Module`): The model or part of the model to be optimized.
        cfg (dict): The config dict of the optimizer.

    Returns:
        :obj:`torch.optim.Optimizer`: The initialized optimizer.
    """
    optimizer_type = cfg.pop('type')
    if optimizer_type == 'SGD':
        return optim.SGD(module.parameters(), **cfg)
    elif optimizer_type == 'Adam':
        return optim.Adam(module.parameters(), **cfg)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(module.parameters(), **cfg)
    else:
        raise ValueError(f'Unsupported optimizer type: {optimizer_type}')

def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizer will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    if all(isinstance(v, dict) for v in cfgs.values()):
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    return build_optimizer(model, cfgs)

