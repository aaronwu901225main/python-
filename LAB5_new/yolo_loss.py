import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: predictions after sigmoid, shape [N, *]
        targets: ground truth labels (0 or 1), shape [N, *]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4] (raw predictions)
        target_boxes: [bsz, grid, grid, anchors, 4] (encoded targets)
        anchors: list of (w, h) for the anchors at this scale (normalized 0-1)
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        anchors = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)

        # coordinate offset for each grid cell
        grid_range = torch.arange(grid, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid_x = grid_x.view(1, grid, grid, 1, 1)
        grid_y = grid_y.view(1, grid, grid, 1, 1)

        if self.type == 'giou':
            ##################YOUR CODE HERE##########################
            # 1. predicted centre (cell offset) and size
            # 2. target centre still stored as cell offset, convert to same system
            # 3. Convert both to image-normalised coordinates
            # 4. boxes to corner format
            # 5. Intersection box
            # 6. union area
            # 7. smallest enclosing box
            ##########################################################
            eps = 1e-9
            # anchors [1,1,1,A,2]
            ax = anchors[..., 0]
            ay = anchors[..., 1]

            # Decode predictions
            px = torch.sigmoid(pred_boxes[..., 0:1])
            py = torch.sigmoid(pred_boxes[..., 1:2])
            pw = torch.exp(torch.clamp(pred_boxes[..., 2:3], -10, 10)) * ax
            ph = torch.exp(torch.clamp(pred_boxes[..., 3:4], -10, 10)) * ay

            # Decode targets
            tx = target_boxes[..., 0:1]
            ty = target_boxes[..., 1:2]
            tw = target_boxes[..., 2:3]
            th = target_boxes[..., 3:4]

            # Normalize to image space [0,1]
            px = (px + grid_x) / grid
            py = (py + grid_y) / grid
            tx = (tx + grid_x) / grid
            ty = (ty + grid_y) / grid

            # Convert to corners
            px1 = px - pw / 2
            py1 = py - ph / 2
            px2 = px + pw / 2
            py2 = py + ph / 2

            tx1 = tx - tw / 2
            ty1 = ty - th / 2
            tx2 = tx + tw / 2
            ty2 = ty + th / 2

            # Intersection
            ix1 = torch.max(px1, tx1)
            iy1 = torch.max(py1, ty1)
            ix2 = torch.min(px2, tx2)
            iy2 = torch.min(py2, ty2)
            iw = torch.clamp(ix2 - ix1, min=0)
            ih = torch.clamp(iy2 - iy1, min=0)
            inter = iw * ih

            # Areas
            pa = torch.clamp(px2 - px1, min=0) * torch.clamp(py2 - py1, min=0)
            ta = torch.clamp(tx2 - tx1, min=0) * torch.clamp(ty2 - ty1, min=0)
            union = pa + ta - inter + eps
            iou = inter / union

            # Enclosing box
            cx1 = torch.min(px1, tx1)
            cy1 = torch.min(py1, ty1)
            cx2 = torch.max(px2, tx2)
            cy2 = torch.max(py2, ty2)
            cw = torch.clamp(cx2 - cx1, min=0)
            ch = torch.clamp(cy2 - cy1, min=0)
            c_area = cw * ch + eps

            giou = iou - (c_area - union) / c_area
            giou_loss = 1.0 - giou

            return giou_loss.squeeze(-1)

        elif self.type == 'mse':
            ##################YOUR CODE HERE##########################
            #### MSE box loss ####
            ## hints: decode predicted boxes, compute MSE loss on box coordinates
            ## Don't forget to caculate w, h mse in log space.
            ##########################################################
            grid = grid
            # Decode predictions
            px = torch.sigmoid(pred_boxes[..., 0])
            py = torch.sigmoid(pred_boxes[..., 1])
            pw = pred_boxes[..., 2]
            ph = pred_boxes[..., 3]

            tx = target_boxes[..., 0]
            ty = target_boxes[..., 1]
            tw = torch.clamp(target_boxes[..., 2], min=1e-6)
            th = torch.clamp(target_boxes[..., 3], min=1e-6)

            # Compare centers (cell offsets)
            loss_xy = (px - tx) ** 2 + (py - ty) ** 2
            # Compare sizes in log space
            loss_wh = (pw - torch.log(tw)) ** 2 + (ph - torch.log(th)) ** 2
            return loss_xy + loss_wh
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
        lambda_class=1.0,
        anchors=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors  # List of anchor boxes per scale
    # Check for NaNs in any of the loss scalars and print which one is NaN
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 75]
        targets: list of 3 scales, each [batch, grid, grid, 3, 25]
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0
        batch_size = predictions[0].size(0)

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, _ = gt.shape
            # Reshape prediction: [B, H, W, 75] -> [B, H, W, 3, 25]
            pred = pred.view(bsz, grid, grid, num_anchors, -1)
            ##################YOUR CODE HERE##########################
            # 分離各成分
            pred_box = pred[..., 0:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            tgt_box = gt[..., 0:4]
            tgt_obj = gt[..., 4]
            tgt_cls = gt[..., 5:]

            obj_mask = tgt_obj > 0.5
            noobj_mask = ~obj_mask

            # Box loss (僅計算正樣本)
            box_l = self.box_loss(pred_box, tgt_box, anchors)
            box_l = box_l[obj_mask]
            if box_l.numel() > 0:
                total_box_loss = total_box_loss + box_l.sum()

            # Objectness loss
            obj_pos = self.bce_loss(pred_obj[obj_mask], tgt_obj[obj_mask]) if obj_mask.any() else torch.tensor(0.0, device=device)
            obj_neg = self.bce_loss(pred_obj[noobj_mask], tgt_obj[noobj_mask]) if noobj_mask.any() else torch.tensor(0.0, device=device)
            total_obj_loss_pos = total_obj_loss_pos + obj_pos.sum()
            total_obj_loss_neg = total_obj_loss_neg + obj_neg.sum()

            # Class loss (僅正樣本)
            if obj_mask.any():
                cls_l = self.bce_loss(pred_cls[obj_mask], tgt_cls[obj_mask])
                total_cls_loss = total_cls_loss + cls_l.sum()

            total_num_pos += obj_mask.sum().item()
            total_num_neg += noobj_mask.sum().item()
            ##########################################################
            

        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        total_box_loss = total_box_loss / pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom
        total_cls_loss = total_cls_loss / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom

        # Combined loss
        
        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_cls_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'noobj': total_noobj_loss,
            'cls': total_cls_loss,
        }
        
        return loss_dict
