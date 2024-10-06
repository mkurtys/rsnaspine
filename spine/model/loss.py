import torch
import torch.nn as nn
import torch.nn.functional as F


def F_grade_loss(grade, truth, mask):
    eps = 1e-5
    weight = torch.FloatTensor([1,2,4]).to(grade.device)
    
    g = grade.reshape(-1,3)
    t = truth.reshape(-1)
    mask = mask.reshape(-1)

    # print(f" mask {mask.sum()}")

    # go for log softmax instead?
    # loss = F.nll_loss( torch.clamp(g, eps, 1-eps).log(), t,weight=weight, ignore_index=-1)
    loss = F.cross_entropy(g[mask], t[mask], weight=weight, ignore_index=-1)
    if torch.isnan(loss):
        # print("grade loss is nan")
        return torch.tensor(0.0).to(grade.device)
    return loss

 
def F_zxy_loss(coords, gt, mask, heatmap):
    # heatmap_size = torch.stack([torch.tensor(h.shape[-3:]).float().to(coords.device) for h in heatmap])
    # heatmap_size = heatmap_size.unsqueeze(1)
    # print("heatmap_size ",heatmap_size.shape)
    # print("coords ",coords.shape)
    
    # loss = F.mse_loss((coords/heatmap_size)[mask], (gt/heatmap_size)[mask])
    loss = F.mse_loss(coords[mask], gt[mask])/10
    return loss


# consider adding mask for layers without label
def F_focal_heatmap_loss(heatmap, gt, D, coords_mask, grades):
    # heatmap =  torch.split_with_sizes(heatmap, D, 0)
    # print("heatmap type",type(heatmap))
    # print("gt type",type(gt))

    # gt (B*D, num_point*num_grade, hh, ww)
    gt = torch.split(gt, D.tolist(), 0)
    num_image = len(heatmap)

    grades_mask = grades >= 0
    coords_mask = coords_mask > 0

    for i in range(num_image):
        # (D, num_point*num_grade, H, W) -> (num_point*num_grade, D, H, W)
        # print(gt[i].shape)
        # print(heatmap[i].shape)
        gt_i = gt[i].permute(1,0,2,3)
        gt_i = gt_i.reshape(*heatmap[i].shape)
        
        pos_inds = gt_i>=1
        neg_inds = gt_i<1

        #print(f"pos_inds {pos_inds.sum()}")
        #print(f"max {gt_i.max()}")
        #print(f"f gt {gt_i.sum()}")
        # print(coords_mask[i])
        # print(grades_mask[i])
        pos_inds = pos_inds & coords_mask[i].reshape(-1,1,1,1,1) & grades_mask[i].reshape(-1,1,1,1,1)
        neg_inds = neg_inds & coords_mask[i].reshape(-1,1,1,1,1) & grades_mask[i].reshape(-1,1,1,1,1)
        #print(f"AFTER pos_inds {pos_inds.sum()}")

        neg_weights = torch.pow(1 - gt_i[neg_inds], 4)

        loss = 0.0
        heatmap_i=heatmap[i]
        # TODO standardize, num_point*num_grade -> num_point, num_grade
        # heatmap_i = heatmap[i].reshape(-1, *heatmap[i].shape[2:])

        # shapes -> (num_point*num_grade, dd, hh, ww)
        # print("gt_i ",gt_i.shape)
        # print("heatmap_i ",heatmap_i.shape)
        # print("pos_inds ",pos_inds.shape)
        pos_pred = heatmap_i[pos_inds]
        neg_pred = heatmap_i[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        # print("num_pos ",num_pos)
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()/(gt_i.shape[0]*gt_i.shape[2])
        # print(f" gt_i {gt_i.shape} heatmap_i {heatmap_i.shape} pos_pred {pos_pred.shape} neg_pred {neg_pred.shape}")

        # print(f"pos_loss {pos_loss} neg_loss {neg_loss} num_pos {num_pos} num_neg {neg_inds.sum()}")

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss/num_image


#F_JS_divergence_loss
def F_JS_heatmap_loss(heatmap, truth, D):
    # heatmap =  torch.split_with_sizes(heatmap, D, 0)  
    # truth =  torch.split_with_sizes(truth, D, 0)
    truth = torch.split(truth, D.tolist(), 0)
    num_image = len(heatmap)

    loss = 0.0
    for i in range(num_image):
        p,q = truth[i], heatmap[i]
        # D,num_point*num_grade, H,W 
        p = p.transpose(1,0).reshape(*q.shape).flatten(1)
        q = q.flatten(1)
        eps = 1e-8
        p = torch.clamp(p,eps,1-eps)
        q = torch.clamp(q,eps,1-eps)
        m = (0.5 * (p + q)).log()
        #  first arg - Tensor of arbitrary shape in log-probabilities
        kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    loss = loss/num_image
    return loss