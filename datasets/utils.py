import torch
import torch.nn.functional as F
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def structure_loss(pred,mask):
    """
    loss function (ref:F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask,kernel_size=31,stride=1,padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred,mask,reduction='mean')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2,3))
    union = ((pred + mask) * weit).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
