from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_getattr(obj: object, multiattr: str) -> Any:
    """Multi-attributes getattr."""
    attrs = multiattr.split(".")
    recur_attr = getattr(obj, attrs[0])
    for a in attrs[1:]:
        recur_attr = getattr(recur_attr, a)
    return recur_attr


def multi_setattr(obj: object, multiattr: str, value: Any) -> None:
    """Multi-attributes setattr."""
    attrs = multiattr.split(".")
    recur_attr = getattr(obj, attrs[0])
    for a in attrs[1:-1]:
        recur_attr = getattr(recur_attr, a)
    setattr(recur_attr, attrs[-1], value)


def cvt2quant(model: nn.Module, quant_str: str) -> None:
    """Given a model and quant"""
    assert isinstance(quant_str, str)
    assert all([i in "ftb" for i in quant_str])
    # 34 layers?
    targets = [
        "base.conv1_1",
        "base.conv1_2",
        "base.conv2_1",
        "base.conv2_2",
        "base.conv3_1",
        "base.conv3_2",
        "base.conv3_3",
        "base.conv4_1",
        "base.conv4_2",
        "base.conv4_3",
        "base.conv5_1",
        "base.conv5_2",
        "base.conv5_3",
        "base.conv6",
        "base.conv7",

        "aux_convs.conv8_1",
        "aux_convs.conv8_2",
        "aux_convs.conv9_1",
        "aux_convs.conv9_2",
        "aux_convs.conv10_1",
        "aux_convs.conv10_2",
        "aux_convs.conv11_1",
        "aux_convs.conv11_2",

        "pred_convs.loc_conv4_3",
        "pred_convs.loc_conv7",
        "pred_convs.loc_conv8_2",
        "pred_convs.loc_conv9_2",
        "pred_convs.loc_conv10_2",
        "pred_convs.loc_conv11_2",
        "pred_convs.cl_conv4_3",
        "pred_convs.cl_conv7",
        "pred_convs.cl_conv8_2",
        "pred_convs.cl_conv9_2",
        "pred_convs.cl_conv10_2",
        "pred_convs.cl_conv11_2",
    ]
    assert len(quant_str) == len(targets)
    all_modules = [n for n, _ in model.named_modules()]
    all_modules = list(filter(lambda x: x.find(".") > -1, all_modules))
    assert all([t in all_modules for t in targets])

    for t, q in zip(targets, quant_str):
        layer = multi_getattr(model, t)
        quant_layer = to_quant_layer(layer, q)
        multi_setattr(model, t, quant_layer)


def get_num_weight_from_name(model: nn.Module, names: List[str]) -> List[int]:
    """Get list of number of weights from list of name of modules."""
    numels = []
    for n in names:
        module = multi_getattr(model, n)
        num_weights = module.weight.numel()
        numels.append(num_weights)
    return numels


def measure_sparse(*ws) -> float:
    """Measure the sparsity of input tensors or *tensors.
    Example:

    """
    if not ws:
        # Detecting in case, empty tuple of ws (max pooling or others).
        sparse = torch.tensor(0.0)
    else:
        # In case, not empty tuple.
        total_sparity = 0
        num_params = 0
        for w in ws:
            if w is None:
                # In case of w is None.
                continue
            w = w.data
            device = w.device
            num_params += w.numel()
            total_sparity += torch.where(
                w == 0.0,
                torch.tensor(1.0).to(device),
                torch.tensor(0.0).to(device),
            ).sum()
        if num_params == 0:
            # In case, all parameters is zeros. 0/0 = ZeroDivisionError.
            sparse = torch.tensor(0.0)
        else:
            sparse = total_sparity / num_params
    return sparse.item()


class BinQuant(torch.autograd.Function):
    """BinaryConnect quantization.
    Refer:
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845/3
    """

    @staticmethod
    def forward(ctx, w):
        """Require w be in range of [0, 1].
        Otherwise, it is not in activate range.
        """
        return w.sign()

    @staticmethod
    def backward(ctx, grad_o):
        grad_i = grad_o.clone()
        return grad_i


class TerQuant(torch.autograd.Function):
    """TeraryNet quantization function."""

    @staticmethod
    def forward(ctx, w, threshold):
        ctx.save_for_backward(w, threshold)
        device = w.device
        w_ter = torch.where(
            w > threshold,
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
        )
        w_ter = torch.where(w.abs() <= -threshold, torch.tensor(0.0).to(device), w_ter)
        w_ter = torch.where(w < -threshold, torch.tensor(-1.0).to(device), w_ter)
        return w_ter

    @staticmethod
    def backward(ctx, grad_o):
        """Back propagation using same as identity function."""
        grad_i = grad_o.clone()
        return grad_i, None


def ternary_threshold(delta: float = 0.7, *ws):
    """Ternary threshold find in ws."""
    assert isinstance(delta, float)
    num_params = sum_w = 0
    if not ws:
        # In case, of all params cannot be found.
        threshold = torch.tensor(np.nan)
    else:
        for w in ws:
            num_params += w.data.numel()
            sum_w += w.abs().sum()
        threshold = delta * (sum_w / num_params)
    return threshold


class BinLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinQuant.apply(self.weight)
        y = F.linear(x, self.weight_q, self.bias)
        return y


class BinConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinQuant.apply(self.weight)
        y = F.conv2d(
            x, self.weight_q, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class TerLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = ternary_threshold(self.delta, self.weight)
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.linear(x, self.weight_q, self.bias)
        return x


class TerConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = 0.7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threshold = ternary_threshold(self.delta, self.weight)
        self.weight_q = TerQuant.apply(self.weight, threshold)
        x = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return x
#def to_quant_layer(layer: nn.Module, quant_char: str) -> nn.Module:
#    """ """
#    assert isinstance(quant_char, str)
#    if isinstance(layer, nn.Conv2d):
#        kwargs = {
#            "in_channels": layer.in_channels,
#            "out_channels": layer.out_channels,
#            "kernel_size": layer.kernel_size,
#            "stride": layer.stride,
#            "padding": layer.padding,
#            "dilation": layer.dilation,
#            "groups": layer.groups,
#            "bias": layer.bias is not None,
#            "padding_mode": layer.padding_mode,
#        }
#        if quant_char == "f":
#            quant_layer = nn.Conv2d(**kwargs)
#        elif quant_char == "t":
#            quant_layer = TerConv2d(**kwargs)
#        elif quant_char == "b":
#            quant_layer = BinConv2d(**kwargs)
#        else:
#            raise NotImplementedError(
#                f"quant_char: {quant_char} can be only `f`, `b`, and `t`."
#            )
#
#    elif isinstance(layer, nn.Linear):
#        kwargs = {
#            "in_features": layer.in_features,
#            "out_features": layer.out_features,
#            "bias": layer.bias is not None,
#        }
#        if quant_char == "f":
#            quant_layer = nn.Linear(**kwargs)
#        elif quant_char == "t":
#            quant_layer = TerLinear(**kwargs)
#        elif quant_char == "b":
#            quant_layer = BinLinear(**kwargs)
#        else:
#            raise NotImplementedError(
#                f"quant_char: {quant_char} can be only `f`, `b`, and `t`."
#            )
#    else:
#        raise NotImplementedError("Support only nn.Conv2d and nn.Linear.")
#    return quant_layer


def to_quant_layer(layer: nn.Module, quant_char: str) -> nn.Module:
    """ """
    assert isinstance(quant_char, str)
    if isinstance(layer, nn.Conv2d):
        kwargs = {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
            "bias": layer.bias is not None,
            "padding_mode": layer.padding_mode,
        }
        if quant_char == "f":
            quant_layer = nn.Conv2d(**kwargs)
        elif quant_char == "t":
            quant_layer = TerConv2d(**kwargs)
        elif quant_char == "b":
            quant_layer = BinConv2d(**kwargs)
        else:
            raise NotImplementedError(
                f"quant_char: {quant_char} can be only `f`, `b`, and `t`."
            )

    elif isinstance(layer, nn.Linear):
        kwargs = {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "bias": layer.bias is not None,
        }
        if quant_char == "f":
            quant_layer = nn.Linear(**kwargs)
        elif quant_char == "t":
            quant_layer = TerLinear(**kwargs)
        elif quant_char == "b":
            quant_layer = BinLinear(**kwargs)
        else:
            raise NotImplementedError(
                f"quant_char: {quant_char} can be only `f`, `b`, and `t`."
            )
    else:
        raise NotImplementedError("Support only nn.Conv2d and nn.Linear.")
    return quant_layer


if __name__ == "__main__":
    from functools import reduce

    import wandb
    from model_n import SSD300

    model = SSD300(11)
    # ws = ["t" for _ in range(35)]
    ws = ["b" for _ in range(35)]
    # ws = ["f" for _ in range(35)]
    ws = reduce(lambda x, y: x + y, ws)
    cvt2quant(model, ws)
    print(model)
    model.forward(torch.zeros(1, 3, 300, 300))

    # TODO: Adding hyper-tunner.
    model.base.load_pretrained_layers()
    model.pred_convs.init_conv2d()
    model.aux_convs.init_conv2d()


#    wandb.init(
#        project="mixed-ssd300",
#        notes="",
#        tags=["object-detection", "quantization"],
#        config=args,
#    )
#    wandb.watch(model)
#    wandb.log()
    # Maybe using AttrDict with wandb log



