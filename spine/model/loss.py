import torch
import torch.nn as nn
import torch.nn.functional as F


def F_grade_loss(grade, truth):
    eps = 1e-5
    weight = torch.FloatTensor([1,2,4]).to(grade.device)

    t = truth.reshape(-1)
    g = grade.reshape(-1,3)

    loss = F.nll_loss( torch.clamp(g, eps, 1-eps).log(), t,weight=weight, ignore_index=-1)
    #loss = F.cross_entropy(g, t,weight=weight, ignore_index=-1)
    return loss

 
def F_zxy_loss(z, xy,  z_truth, xy_truth):
    m = z_truth!=-1
    z_truth = z_truth.float()
    loss = (
        F.mse_loss(z[m], z_truth[m]) + F.mse_loss(xy[m], xy_truth[m])
    )
    return loss


# consider adding mask for layers without label
def F_focal_heatmap_loss(heatmap, gt, D):
    heatmap =  torch.split_with_sizes(heatmap, D, 0)  
    truth =  torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    for i in range(num_image):
        gt_i = gt[i]
        pos_inds = gt_i.eq(1)
        neg_inds = gt_i.lt(1)

        neg_weights = torch.pow(1 - gt_i[neg_inds], 4)

        loss = 0
        for pred in heatmap[i]:
            pos_pred = pred[pos_inds]
            neg_pred = pred[neg_inds]

            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

            num_pos  = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if pos_pred.nelement() == 0:
                loss = loss - neg_loss
            else:
                loss = loss - (pos_loss + neg_loss) / num_pos
    return loss/num_image


#F_JS_divergence_loss
def F_JS_heatmap_loss(heatmap, truth, D):
    heatmap =  torch.split_with_sizes(heatmap, D, 0)  
    truth =  torch.split_with_sizes(truth, D, 0)
    num_image = len(heatmap)

    loss = 0.0
    for i in range(num_image):
        p,q = truth[i], heatmap[i]
        D,num_point,num_grade, H,W = p.shape

        eps = 1e-8
        p = torch.clamp(p.transpose(1,0).flatten(1),eps,1-eps)
        q = torch.clamp(q.transpose(1,0).flatten(1),eps,1-eps)
        m = (0.5 * (p + q)).log()

        kl = lambda x,t: F.kl_div(x,t, reduction='batchmean', log_target=True)
        loss += 0.5 * (kl(m, p.log()) + kl(m, q.log()))
    loss = loss/num_image
    return loss