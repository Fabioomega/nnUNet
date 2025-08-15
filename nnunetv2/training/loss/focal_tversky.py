from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SoftFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        clip_tp: float = None,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.5,
    ):
        """ """
        super(SoftFocalTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        nominator = tp
        denominator = tp + self.alpha * fn + self.beta * fp

        tversky = (nominator + self.smooth) / (
            torch.clip(denominator + self.smooth, 1e-8)
        )

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return (1 - tversky) ** (1 / self.gamma)


class MemoryEfficientSoftFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.5,
    ):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftFocalTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = (
                y_onehot.sum(axes)
                if loss_mask is None
                else (y_onehot * loss_mask).sum(axes)
            )

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes)
            fp = (x * (~y_onehot)).sum(axes)
            sum_gt = y_onehot.sum(axes)
            fn = sum_gt - tp
        else:
            tp = (x * y_onehot * loss_mask).sum(axes)
            fp = (x * (~y_onehot) * loss_mask).sum(axes)
            sum_gt = (y_onehot * loss_mask).sum(axes)
            fn = sum_gt - tp

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fp = AllGatherGrad.apply(fp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)

            tp = tp.sum(0)
            fp = fp.sum(0)
            fn = fn.sum(0)

        tversky = (tp + self.smooth) / (
            torch.clip(tp + self.alpha * fp + self.beta * fn + self.smooth, 1e-8)
        )

        tversky = tversky.mean()
        return (1 - tversky) ** (1 / self.gamma)


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(
                net_output.shape, device=net_output.device, dtype=torch.bool
            )
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(
                mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)])
            )
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == "__main__":
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftFocalTverskyLoss(
        apply_nonlin=softmax_helper_dim1,
        batch_dice=True,
        do_bg=False,
        smooth=0,
        ddp=False,
    )
    dl_new = MemoryEfficientSoftFocalTverskyLoss(
        apply_nonlin=softmax_helper_dim1,
        batch_dice=True,
        do_bg=False,
        smooth=0,
        ddp=False,
    )
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
