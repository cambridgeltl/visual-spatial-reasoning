
# reference: https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing

import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
#from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

class FastRCNNOutputs:
     """
     An internal implementation that stores information about outputs of a Fast R-CNN head,
     and provides methods that are used to decode the outputs of a Fast R-CNN head.
     """

     def __init__(
         self,
         box2box_transform,
         pred_class_logits,
         pred_proposal_deltas,
         proposals,
         smooth_l1_beta=0.0,
         box_reg_loss_type="smooth_l1",
     ):
         """
         Args:
             box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                 box2box transform instance for proposal-to-detection transformations.
             pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                 logits for all R predicted object instances.
                 Each row corresponds to a predicted object instance.
             pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                 class-specific or class-agnostic regression. It stores the predicted deltas that
                 transform proposals into final box detections.
                 B is the box dimension (4 or 5).
                 When B is 4, each row is [dx, dy, dw, dh (, ....)].
                 When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
             proposals (list[Instances]): A list of N Instances, where Instances i stores the
                 proposals for image i, in the field "proposal_boxes".
                 When training, each Instances must have ground-truth labels
                 stored in the field "gt_classes" and "gt_boxes".
                 The total number of all instances must be equal to R.
             smooth_l1_beta (float): The transition point between L1 and L2 loss in
                 the smooth L1 loss function. When set to 0, the loss becomes L1. When
                 set to +inf, the loss becomes constant 0.
             box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
         """
         self.box2box_transform = box2box_transform
         self.num_preds_per_image = [len(p) for p in proposals]
         self.pred_class_logits = pred_class_logits
         self.pred_proposal_deltas = pred_proposal_deltas
         self.smooth_l1_beta = smooth_l1_beta
         self.box_reg_loss_type = box_reg_loss_type

         self.image_shapes = [x.image_size for x in proposals]

         if len(proposals):
             box_type = type(proposals[0].proposal_boxes)
             # cat(..., dim=0) concatenates over all images in the batch
             self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
             assert (
                 not self.proposals.tensor.requires_grad
             ), "Proposals should not require gradients!"

             # "gt_classes" exists if and only if training. But other gt fields may
             # not necessarily exist in training for images that have no groundtruth.
             if proposals[0].has("gt_classes"):
                 self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                 # If "gt_boxes" does not exist, the proposals must be all negative and
                 # should not be included in regression loss computation.
                 # Here we just use proposal_boxes as an arbitrary placeholder because its
                 # value won't be used in self.box_reg_loss().
                 gt_boxes = [
                     p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                 ]
                 self.gt_boxes = box_type.cat(gt_boxes)
         else:
             self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
         self._no_instances = len(self.proposals) == 0  # no instances found

     def softmax_cross_entropy_loss(self):
         """
         Deprecated
         """
         _log_classification_stats(self.pred_class_logits, self.gt_classes)
         return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

     def box_reg_loss(self):
         """
         Deprecated
         """
         if self._no_instances:
             return 0.0 * self.pred_proposal_deltas.sum()

         box_dim = self.proposals.tensor.size(1)  # 4 or 5
         cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
         device = self.pred_proposal_deltas.device

         bg_class_ind = self.pred_class_logits.shape[1] - 1
         # Box delta loss is only computed between the prediction for the gt class k
         # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
         # for non-gt classes and background.
         # Empty fg_inds should produce a valid loss of zero because reduction=sum.
         fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

         if cls_agnostic_bbox_reg:
             # pred_proposal_deltas only corresponds to foreground class for agnostic
             gt_class_cols = torch.arange(box_dim, device=device)
         else:
             # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
             # where b is the dimension of box representation (4 or 5)
             # Note that compared to Detectron1,
             # we do not perform bounding box regression for background classes.
             gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                 box_dim, device=device
             )

         if self.box_reg_loss_type == "smooth_l1":
             gt_proposal_deltas = self.box2box_transform.get_deltas(
                 self.proposals.tensor, self.gt_boxes.tensor
             )
             loss_box_reg = smooth_l1_loss(
                 self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                 gt_proposal_deltas[fg_inds],
                 self.smooth_l1_beta,
                 reduction="sum",
             )
         elif self.box_reg_loss_type == "giou":
             fg_pred_boxes = self.box2box_transform.apply_deltas(
                 self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                 self.proposals.tensor[fg_inds],
             )
             loss_box_reg = giou_loss(
                 fg_pred_boxes,
                 self.gt_boxes.tensor[fg_inds],
                 reduction="sum",
             )
         else:
             raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

         loss_box_reg = loss_box_reg / self.gt_classes.numel()
         return loss_box_reg

     def losses(self):
         """
         Deprecated
         """
         return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

     def predict_boxes(self):
         """
         Deprecated
         """
         pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
         return pred.split(self.num_preds_per_image, dim=0)

     def predict_probs(self):
         """
         Deprecated
         """
         probs = F.softmax(self.pred_class_logits, dim=-1)
         return probs.split(self.num_preds_per_image, dim=0)



def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    #cfg['MODEL']['DEVICE']='cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg


def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

def prepare_image_inputs(cfg, img_list, model):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]).cuda() for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
    
    return images, batched_inputs

def get_features(model, images):
    features = model.backbone(images.tensor)
    return features

def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    batch_size = features_list[0].shape[0]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    num_proposals = box_features.shape[0]
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    #if num_proposals>1000: num_proposals = 1000
    box_features = box_features.reshape(batch_size, num_proposals, 1024) # depends on your config and batch size
    return box_features, features_list, num_proposals

def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes

def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

def select_boxes(cfg, output_boxes, scores, num_proposals=1000):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(num_proposals,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0])).cuda()
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        #keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        keep = nms(det_boxes, cls_scores, test_nms_thresh)
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf.detach().cpu()).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf.detach().cpu()).numpy()[::-1][:max_boxes]
    return keep_boxes

def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]



def extract_visual_features(cfg, model, images): 
    images, batched_inputs = prepare_image_inputs(cfg, images, model)
    features = get_features(model, images)
    proposals = get_proposals(model, images, features)
    box_features, features_list, num_proposals = get_box_features(model, features, proposals)
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)
    boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
    temp = [select_boxes(cfg, output_boxes[i], scores[i], num_proposals=num_proposals) for i in range(len(scores))]
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)
    MIN_BOXES=10
    MAX_BOXES=100
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
    visual_embeds = [get_visual_embeds(box_feature, keep_box).detach().cpu() for box_feature, keep_box in zip(box_features, keep_boxes)]
    del images, batched_inputs, features, proposals, box_features, pred_class_logits, pred_proposal_deltas, features_list
    return visual_embeds

