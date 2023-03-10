import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from datasets.data import get_loader, test_dataset
from datasets.utils import clip_gradient, adjust_lr, structure_loss
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from model.mit2_trans import mit_trans as Net
# from model.no_crosstrans import mit_trans as Net
# from model.trans_nocross import mit_trans as Net
# from model.mit_edge import mit_trans as Net
# from model.resnet_edge import mit_trans as Net

from model.smallmodel.cross_transformer  import CONFIGS as CONFIGS_ViT_seg
# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# build the model
config_vit = CONFIGS_ViT_seg[opt.vit_name]
config_vit.n_classes = opt.num_classes
config_vit.n_skip = opt.n_skip

model = Net(config_vit, opt.img_size, opt)
# model.load_checkpoint(config_vit.pretrained_path)

if (opt.load_mit is not None):
    model.encoder1.init_weights(opt.load_mit)
    model.encoder2.init_weights(opt.load_mit)
    print('load model from ', opt.load_mit)

def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
print_network(model, 'mit_trans')
# set the path
image_root = opt.image_root
gt_root = opt.gt_root
edge_root = opt.edge_root
test_image_root = opt.test_image_root
test_gt_root = opt.test_gt_root
test_edge_root = opt.test_edge_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_edge_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Net-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,  save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            bgts = 1 - gts
            edges = edges.cuda()
            # cod, coee= model(images)
            # cod, coee, s_coee, s_cod= model(images)
            cod, s_foreg, s_backg, edge4, edge3, edge2, edge1 = model(images)
            # loss = CE(coee, edges) + CE(cod, gts)
            loss_foreground = structure_loss(s_foreg, gts)
            loss_background = structure_loss(s_backg, bgts)
            loss_edge = F.binary_cross_entropy_with_logits(edge4, edges, reduction='mean') \
                        + F.binary_cross_entropy_with_logits(edge3, edges, reduction='mean')\
                        + F.binary_cross_entropy_with_logits(edge2, edges, reduction='mean')\
                        + F.binary_cross_entropy_with_logits(edge1, edges, reduction='mean')
            loss_saliency = structure_loss(cod, gts)
            loss = loss_foreground + loss_edge + loss_saliency + loss_background
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                             format(epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Camouflaged', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = cod[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('COD', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, edge, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            edge = np.asarray(edge, np.float32)
            gt /= (gt.max() + 1e-8)
            edge /= (edge.max() + 1e-8)
            image = image.cuda()
            # res, _ = model(image)
            # res, _, _, _ = model(image)
            res, _, _, _, _, _, _ = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
