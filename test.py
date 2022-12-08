import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from model.mit2_trans import mit_trans as Net
from model.smallmodel.cross_transformer import CONFIGS as CONFIGS_ViT_seg
from datasets.data import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--vit_name', type=str, default='ViT-B_16')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--pretrained', type=bool, default=False, help='train from checkpoints')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='I:/camouflaged/Dataset/TestDataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
config_vit = CONFIGS_ViT_seg[opt.vit_name]
config_vit.n_classes = opt.num_classes
config_vit.n_skip = opt.n_skip
model = Net(config_vit, opt.img_size, opt)
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('./checkpoint/dtit.pth'))
model.cuda()
model.eval()
test_datasets = ['CAMO', 'COD10K', 'NC4K']
for dataset in test_datasets:
    save_path1 = './result_best/' + dataset + '/cod/'
    save_path2 = './result_best/' + dataset + '/coee/'
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
        os.makedirs(save_path2)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    edge_root=dataset_path + dataset + '/Edge/'
    test_loader = test_dataset(image_root, gt_root, edge_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, edge, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        cod_pred, coee_pred, _, _, _, _, _, _, _, _ = model(image)
        cod_pred = F.interpolate(cod_pred, size=gt.shape, mode='bilinear', align_corners=False)
        coee_pred = F.interpolate(coee_pred, size=gt.shape, mode='bilinear', align_corners=False)

        cod_pred = cod_pred.sigmoid().data.cpu().numpy().squeeze()
        coee_pred = coee_pred.sigmoid().data.cpu().numpy().squeeze()

        cod_pred = (cod_pred - cod_pred.min()) / (cod_pred.max() - cod_pred.min() + 1e-8)
        coee_pred = (coee_pred - coee_pred.min()) / (coee_pred.max() - coee_pred.min() + 1e-8)


        print('save img to: ', save_path1+name)
        cv2.imwrite(save_path1+name, cod_pred*255)
        cv2.imwrite(save_path2+name, coee_pred*255)
    print('Test Done!')
