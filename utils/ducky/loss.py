from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from utils.loss import smooth_BCE, FocalLoss

class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        print(h)

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # CLLdistance = CumulativeLinkLoss() # TODO: Idk if this goes to correct device
        # CLLrotation = CumulativeLinkLoss()
        # CLLdistance = nn.CrossEntropyLoss()
        # CLLrotation = nn.CrossEntropyLoss()
        CLLdistance = UnimodalNet(K=5) # TODO: Idk if this goes to correct device
        CLLrotation = UnimodalNet(K=5)
        # CLLdistance = CDW_CE(K=5) # TODO: Idk if this goes to correct device
        # CLLrotation = CDW_CE(K=5)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.CLLdistance, self.CLLrotation, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, CLLdistance, CLLrotation, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.nov = m.nov  # number of ordinal variables
        self.noc = m.noc  # number of ordinal classes
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldistance = torch.zeros(1, device=self.device)  # distance loss
        lrotation = torch.zeros(1, device=self.device)  # rotation loss
        tcls, tbox, tdistance, trotation, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    number_classes = self.nc-self.nov*self.noc

                    pdistance = pcls[:, number_classes:number_classes+self.noc]
                    protation = pcls[:, number_classes+self.noc:number_classes+self.noc*2]
                    pcls = pcls[:, :number_classes] # remove ordinal classes

                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                    # ldistance += self.CLLdistance(pdistance, tdistance[i].unsqueeze(-1).to(torch.int64))
                    # lrotation += self.CLLrotation(protation, trotation[i].unsqueeze(-1).to(torch.int64))
                    ldistance += self.CLLdistance(pdistance, tdistance[i].to(torch.int64)).sum()
                    lrotation += self.CLLrotation(protation, trotation[i].to(torch.int64)).sum()

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ldistance *= self.hyp['dist']
        lrotation *= self.hyp['rot']
        bs = tobj.shape[0]  # batch size

        # TODO: Adjust weights
        return (lbox + lobj + lcls + ldistance + lrotation) * bs, torch.cat((lbox, lobj, lcls, ldistance, lrotation)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tdistance, trotation, indices, anch = [], [], [], [], [], []
        gain = torch.ones(9, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc = t[:, :2]
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            a = t[:,8]
            dist = t[:,6]
            rot = t[:, 7]
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class

            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tdistance.append(dist)  # distance class
            trotation.append(rot)   # rotation class

        return tcls, tbox, tdistance, trotation, indices, anch

###########################################################################################
## Based on https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/losses.py

def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss

    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.

    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.

    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    Returns
    -------
    loss: torch.Tensor

    """
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)
    

############################### UTILITIES ######################################
# https://github.com/rpmcruz/unimodal-ordinal-regression

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

def to_classes(probs, method=None):
    # method can be:
    # "mode" = class with highest probability (argmax) [default]
    # "mean" = expectation average of the probabilities distribution
    # "median" = median weighted by the probabilities distribution
    assert method in (None, 'mode', 'mean', 'median')
    if method == 'mean':  # also called expectation trick by Beckham
        K = probs.shape[1]
        kk = torch.arange(K, device=probs.device, dtype=torch.float32)[None]
        return torch.round(torch.sum(kk * probs, 1)).long()
    elif method == 'median':
        # the weighted median is the value whose cumulative probability is 0.5
        Pc = torch.cumsum(probs, 1)
        return torch.sum(Pc < 0.5, 1)
    else:  # default=mode
        return probs.argmax(1)

# we are using softplus instead of relu since it is smoother to optimize.
# as in http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
approx_relu = F.softplus
relu = F.relu
ce = torch.nn.CrossEntropyLoss(reduction='none')

################################# LOSSES #######################################

class OrdinalLoss(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def how_many_outputs(self):
        # how many output neurons does this loss require?
        return self.K

    def forward(self, ypred, ytrue):
        # computes the loss
        pass

    def reset_epoch(self):
        # some losses use this method to reset running statistics used, e.g., to
        # compute thresholds
        pass

    def to_proba(self, ypred):
        # output -> probabilities
        pass

    def to_classes(self, ypred, method=None):
        # output -> classes. for an explanation of the 'method' parameter, see
        # the utility function to_classes() above.
        # note: only in rare cases, you need to overload this (e.g., if your
        # loss does not produce probabilities or if it has a special means of
        # computing classes). otherwise, do not overload.
        probs = self.to_proba(ypred)
        classes = to_classes(probs, method)
        return classes

    def to_scores(self, ypred):
        # output -> scalar rank score. by default, the output (if single output)
        # ot the expected value (from the probabilities).
        if self.how_many_outputs() == 1:
            return ypred[:, 0]
        device = ypred.device
        probs = self.to_proba(ypred)
        kk = torch.arange(self.K, device=ypred.device, dtype=torch.float32)[None]
        return torch.sum(kk * probs, 1)
    
class UnimodalNet(OrdinalLoss):
    def forward(self, ypred, ytrue):
        return ce(self.activation(ypred), ytrue)

    def to_proba(self, ypred):
        return F.softmax(self.activation(ypred), 1)

    def activation(self, ypred):
        # first use relu: we need everything positive
        # for differentiable reasons, we use softplus
        ypred = approx_relu(ypred)
        # if output=[X,Y,Z] => pos_slope=[X,X+Y,X+Y+Z]
        # if output=[X,Y,Z] => neg_slope=[Z,Z+Y,Z+Y+X]
        pos_slope = torch.cumsum(ypred, 1)
        neg_slope = torch.flip(torch.cumsum(torch.flip(ypred, [1]), 1), [1])
        ypred = torch.minimum(pos_slope, neg_slope)
        return ypred
    
class CDW_CE(OrdinalLoss):
    def __init__(self, K, alpha=5):
        super().__init__(K)
        self.alpha = alpha

    def d(self, y):
        # internal function for the distance penalization. you may overload
        # this function if you want to use another.
        i = torch.arange(self.K, device=y.device)
        return torch.abs(i[None] - y[:, None])**self.alpha

    def forward(self, ypred, ytrue):
        ypred = F.softmax(ypred, 1)
        return -torch.sum(torch.log(1-ypred) * self.d(ytrue), 1)

    def to_proba(self, ypred):
        return F.softmax(ypred, 1)

class OrdinalLogLoss(CDW_CE):
    def __init__(self, K, alpha=1.5):
        super().__init__(K, alpha)