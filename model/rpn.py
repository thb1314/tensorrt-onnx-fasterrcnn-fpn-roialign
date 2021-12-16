# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops import boxes as box_ops

from model import _utils as det_utils
from model.image_list import ImageList

from torch.jit.annotations import List, Optional, Dict, Tuple

# Import AnchorGenerator to keep compatibility.
from model.anchor_utils import AnchorGenerator


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    # num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    num_anchors = torch.as_tensor(int(ob.shape[1]), device=ob.device, dtype=torch.int64).unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    将 N,A,C,H,W 转换为  N,H*W*A,C
    :param layer:
    :param N:
    :param A:
    :param C:
    :param H:
    :param W:
    :return:
    """
    layer = layer.view(-1, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(-1, A*H*W, C)
    return layer

def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = int(boxes.dim())
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    reshapes = [int(item) for item in boxes.shape]
    reshapes[0] = -1
    return clipped_boxes.reshape(reshapes)

def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = list(map(int,box_cls_per_level.shape))
        Ax4 = int(box_regression_per_level.shape[1])
        A = Ax4 // 4
        C = AxC // A
        # N, H*W*A, C
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )

        box_cls_flattened.append(box_cls_per_level)

        # N, H*W*A, 4
        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    # box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0,-2)
    box_cls = torch.cat(box_cls_flattened, dim=1)
    box_regression = torch.cat(box_regression_flattened, dim=1)
    return box_cls, box_regression

@torch.jit._script_if_tracing
def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = box_ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep

class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN. 大于该阈值可以视为正样本
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN. 小于该阈值可以视为负样本
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss 每一个image所设定的参与训练的anchor的数量
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN 每一个image所设定的参与训练的正样本的比例
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation 参与nms之前保留的样本数量
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation 参与nms之后保留的样本数量
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals NMS的iou设定

    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """

        :param proposals: N,-1,4的anchor
        :param objectness:
        :param image_shapes: 图像形状列表
        :param num_anchors_per_level: fpn每一个level的anchor的个数
        :return:
        """
        num_images = int(proposals.shape[0])
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()

        num = 1
        for item in objectness.shape[0:]:
            num *= int(item)
        num //= num_images
        objectness = objectness.reshape(-1, num)

        levels = [
            torch.full((n,), idx, dtype=torch.float32, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]

        levels = torch.cat(levels, 0)
        # 如果想支持动态batch size就去掉
        b = int(objectness.shape[0])
        ele_size = int(levels.size(0))
        levels = levels.unsqueeze(dim=0)
        if b > 1:
            levels = levels.expand((b, ele_size))

        # levels = torch.full_like(objectness, 0)

        # iter_num_anchors_per_level = [0] + num_anchors_per_level
        # index_start = 0
        # for idx, element in enumerate(num_anchors_per_level):
        #     start = index_start
        #     end = index_start + element
        #     index_start = end
        #     levels[:,start:end] = idx

        # objectness: N x anchor_num
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # image_range = torch.arange(num_images, device=device)
        # batch_idx = image_range[:, None]

        new_objectness = list()
        new_levels = list()
        new_proposals = list()
        for idx,ob,lvl,proposal in zip(top_n_idx, objectness, levels, proposals):
            new_objectness.append(ob.index_select(index=idx, dim=0))
            new_levels.append(lvl.index_select(index=idx, dim=0))
            new_proposals.append(proposal.index_select(index=idx, dim=0))
        # top_n_idx Nx5000

        # NxK
        objectness = torch.stack(new_objectness, dim=0)
        # objectness = torch.gather(objectness, dim=1, index=top_n_idx)
        # NxK
        levels = torch.stack(new_levels, dim=0)
        # levels = torch.gather(levels, dim=1, index=top_n_idx)
        # NxKx4
        # top_n_idxes = torch.stack([top_n_idx,] * 4, dim=-1)
        # proposals = torch.gather(proposals, dim=1, index=top_n_idxes)
        proposals = torch.stack(new_proposals, dim=0)


        if torchvision._is_tracing():
            proposals = clip_boxes_to_image(proposals, image_shapes[0])
            final_score_boxes = torch.cat([torch.sigmoid(objectness.unsqueeze(dim=-1)), proposals, levels.unsqueeze(dim=-1)], dim=-1)
            return final_score_boxes
        final_score_boxes = list()
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_score_boxes.append(torch.cat([scores.unsqueeze(dim=-1),boxes], dim=-1))
        return final_score_boxes

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Arguments:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.view(-1)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,       # type: ImageList
                features,     # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        #  logits, bbox_reg List[Tensor], List[Tensor]
        # objectness 5个 Tensor size为 b*anchor_num*h*w
        # pred_bbox_deltas 5个 Tensor size为 b*anchor_num x 4*h*w
        objectness, pred_bbox_deltas = self.head(features)
        # features: List[Tensor] len = 5
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # 计算每个level anchor的数量
        num_anchors_per_level = [int(s[0] * s[1] * s[2]) for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        if torchvision._is_tracing():
            boxes = anchors[0].detach()

            pred_bbox_deltas = pred_bbox_deltas.to(boxes.dtype)

            # 1, anchor_num
            widths = boxes[:, 2] - boxes[:, 0]
            widths = widths.unsqueeze(dim=0)
            heights = boxes[:, 3] - boxes[:, 1]
            heights = heights.unsqueeze(dim=0)
            ctr_x = boxes[:, 0].unsqueeze(dim=0) + 0.5 * widths
            # ctr_x = ctr_x.unsqueeze(dim=0)
            ctr_y = boxes[:, 1].unsqueeze(dim=0) + 0.5 * heights
            # ctr_y = ctr_y.unsqueeze(dim=0)

            wx, wy, ww, wh = self.box_coder.weights
            dx = pred_bbox_deltas[:, :, 0] / wx
            dy = pred_bbox_deltas[:, :, 1] / wy
            dw = pred_bbox_deltas[:, :, 2] / ww
            dh = pred_bbox_deltas[:, :, 3] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=self.box_coder.bbox_xform_clip)
            dh = torch.clamp(dh, max=self.box_coder.bbox_xform_clip)

            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w = torch.exp(dw) * widths
            pred_h = torch.exp(dh) * heights

            pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
            pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
            pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
            pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
            proposals = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=-1)
        else:
            anchor_num = int(pred_bbox_deltas.size(1))
            proposals = self.box_coder.decode(pred_bbox_deltas.view(-1, 4), anchors).view(-1,anchor_num,4)
        # proposals = torch.stack(new_proposals, dim=0).squeeze(dim=2)
        score_boxes = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)


        losses = {}
        if self.training:
            assert targets is not None
            # 为每个anchor分配一个label和gt_box
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return score_boxes, losses
