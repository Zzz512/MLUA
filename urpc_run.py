import random
import os
import numpy as np
import torch
import argparse
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms as T
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import random
from PIL import Image
from torch.utils.data.sampler import Sampler
import itertools
from util.utils import mean_metric, DiceLoss, mse_loss, sigmoid_rampup, get_current_consistency_weight, sigmoid_mse_loss
from evaluate.utils import recompone_overlap, metric_calculate
from dataset import TrainDataset, ValDataset
from dataloader import TwoStreamBatchSampler
from medpy import metric

gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)


parser = argparse.ArgumentParser(description='CariesNet')
parser.add_argument('--sigma', '-s', type=float, default=5)
args = parser.parse_args()

from model.smpFPN import FPNnet

class CariesSSLNet(LightningModule):
    def __init__(self, lr: float = 0.001, l_batch_size: int = 8, theta: float = 0.99, SSL: bool = True):
        super().__init__()
        self.semi_train = SSL
        self.learning_rate = lr
        self.p = theta
        self.max_epoch = 200
        self.l_batch_size = l_batch_size
        self.glob_step = 0

        # networks
        self.model = FPNnet(in_c=1, c=1)

        self.dice_loss = DiceLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.mse_loss = sigmoid_mse_loss
        self.kl_dist = nn.KLDivLoss(reduction='none')
        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        volume_batch, label_batch = batch
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        _, [outputs_aux3, outputs_aux2, outputs_aux1, outputs] = self.model(volume_batch)
        outputs_soft = torch.sigmoid(outputs)
        outputs_aux1_soft = torch.sigmoid(outputs_aux1)
        outputs_aux2_soft = torch.sigmoid(outputs_aux2)
        outputs_aux3_soft = torch.sigmoid(outputs_aux3)

        loss_ce = self.bce_loss(outputs[:self.l_batch_size], label_batch[:self.l_batch_size][:])
        loss_ce_aux1 = self.bce_loss(outputs_aux1[:self.l_batch_size], label_batch[:self.l_batch_size][:])
        loss_ce_aux2 = self.bce_loss(outputs_aux2[:self.l_batch_size], label_batch[:self.l_batch_size][:])
        loss_ce_aux3 = self.bce_loss(outputs_aux3[:self.l_batch_size], label_batch[:self.l_batch_size][:])

        loss_dice = self.dice_loss(outputs_soft[:self.l_batch_size], label_batch[:self.l_batch_size])
        loss_dice_aux1 = self.dice_loss(
            outputs_aux1_soft[:self.l_batch_size], label_batch[:self.l_batch_size])
        loss_dice_aux2 = self.dice_loss(
            outputs_aux2_soft[:self.l_batch_size], label_batch[:self.l_batch_size])
        loss_dice_aux3 = self.dice_loss(
            outputs_aux3_soft[:self.l_batch_size], label_batch[:self.l_batch_size])

        supervised_loss = (loss_ce+loss_ce_aux1+loss_ce_aux2+loss_ce_aux3 +
                            loss_dice+loss_dice_aux1+loss_dice_aux2+loss_dice_aux3)/8

        preds = (outputs_soft+outputs_aux1_soft +
                    outputs_aux2_soft+outputs_aux3_soft)/4

        variance_main = torch.sum(self.kl_dist(
            torch.log(outputs_soft[:self.l_batch_size:]), preds[:self.l_batch_size:]), dim=1, keepdim=True)
        exp_variance_main = torch.exp(-variance_main)

        variance_aux1 = torch.sum(self.kl_dist(
            torch.log(outputs_aux1_soft[:self.l_batch_size:]), preds[:self.l_batch_size:]), dim=1, keepdim=True)
        exp_variance_aux1 = torch.exp(-variance_aux1)

        variance_aux2 = torch.sum(self.kl_dist(
            torch.log(outputs_aux2_soft[:self.l_batch_size:]), preds[:self.l_batch_size:]), dim=1, keepdim=True)
        exp_variance_aux2 = torch.exp(-variance_aux2)

        variance_aux3 = torch.sum(self.kl_dist(
            torch.log(outputs_aux3_soft[:self.l_batch_size:]), preds[:self.l_batch_size:]), dim=1, keepdim=True)
        exp_variance_aux3 = torch.exp(-variance_aux3)

        consistency_dist_main = (
            preds[:self.l_batch_size:] - outputs_soft[:self.l_batch_size:]) ** 2

        consistency_loss_main = torch.mean(
            consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

        consistency_dist_aux1 = (
            preds[:self.l_batch_size:] - outputs_aux1_soft[:self.l_batch_size:]) ** 2
        consistency_loss_aux1 = torch.mean(
            consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

        consistency_dist_aux2 = (
            preds[:self.l_batch_size:] - outputs_aux2_soft[:self.l_batch_size:]) ** 2
        consistency_loss_aux2 = torch.mean(
            consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

        consistency_dist_aux3 = (
            preds[:self.l_batch_size:] - outputs_aux3_soft[:self.l_batch_size:]) ** 2
        consistency_loss_aux3 = torch.mean(
            consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

        consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                            consistency_loss_aux2 + consistency_loss_aux3) / 4
        consistency_weight = get_current_consistency_weight(self.current_epoch, 200)
        loss = supervised_loss + consistency_weight * consistency_loss

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        imgs, gt = batch
        imgs = imgs.permute(1, 0, 2, 3)
        with torch.no_grad():
            outputs = self(imgs)[1][-1]
        pred = torch.sigmoid(outputs)
        pred_imgs = recompone_overlap(pred.cpu().numpy(), 768, 1536, 192, 192)
        pred_imgs = np.array(pred_imgs > 0.5).squeeze()
        gt = (gt > 0.5).cpu().numpy().squeeze()
        dice = metric.binary.dc(pred_imgs, gt)
        jc = metric.binary.jc(pred_imgs, gt)
        sen = metric.binary.sensitivity(pred_imgs, gt)
        pre = metric.binary.precision(pred_imgs, gt)
        spe = metric.binary.specificity(pred_imgs, gt)
        self.eval_dict["iou"].append(jc)
        self.eval_dict["dice"].append(dice)
        self.eval_dict["pre"].append(pre)
        self.eval_dict["sen"].append(sen)
        self.eval_dict["spe"].append(spe)

    def on_validation_epoch_end(self):
        mean_iou = sum(self.eval_dict["iou"]) / 100
        mean_dice = sum(self.eval_dict["dice"]) / 100
        mean_spe = sum(self.eval_dict["spe"]) / 100
        mean_pre = sum(self.eval_dict["pre"]) / 100
        mean_sen = sum(self.eval_dict["sen"]) / 100
        self.log('val_mean_iou', mean_iou)
        self.log('val_mean_dice', mean_dice)
        self.log('val_mean_spe', mean_spe)
        self.log('val_mean_pre', mean_pre)
        self.log('val_mean_sen', mean_sen)
        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]        


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def train_process(model, train_loader, val_loader, max_epochs, labeled_rate):
    if labeled_rate == "0.1":
        model_name = "URPC10"
    elif labeled_rate == "0.2":
        model_name = "URPC20"
    elif labeled_rate == "0.5":
        model_name = "URPC50"
    tb_logger = pl_loggers.TensorBoardLogger('.\\Cariouslog\\URPC')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_mean_dice',
                                        filename=model_name + '-{epoch:02d}-{val_mean_iou:.4f}-{val_mean_dice:.4f}-{val_mean_spe:.4f}-{val_mean_sen:.4f}-{val_mean_pre:.4f}',
                                        save_top_k=5,
                                        mode='max',
                                        save_weights_only=True)

    trainer = Trainer(max_epochs=max_epochs, logger=tb_logger, gpus=[0, ],
                    precision=16, check_val_every_n_epoch=1, benchmark=True,
                    callbacks=[lr_monitor, checkpoint_callback])  # 使用单卡
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_dataloaders=val_loader)

def main():
    SSL_flag = True
    learning_rate = 1e-4
    theta = 0.99
    labeled_ratio = {"0.1": 265, "0.2": 530, "0.5": 1325}
    labeld_rate = "0.1"
    if SSL_flag:
        batch_size, l_batch_size = 8, 4
    else:
        batch_size, l_batch_size = 4, 4

    pwd = os.getcwd()

    file_path = pwd + "\\data"
    image_path = os.path.join(file_path, "train\\images")
    mask_path = os.path.join(file_path, "train\\labels")
    unlable_path = os.path.join(file_path, "train\\unlabel_images\\images")
    train_image_list = [os.path.join(image_path, file_name) for file_name in os.listdir(image_path)]
    train_label_list = [os.path.join(mask_path, file_name) for file_name in os.listdir(mask_path)]
    train_image_list = sorted(train_image_list,key=lambda x: int(x.split('\\')[-1][:-4]), reverse=False)
    train_label_list = sorted(train_label_list,key=lambda x: int(x.split('\\')[-1][:-4]), reverse=False)

    ul_image_list = [os.path.join(unlable_path, file_name) for file_name in os.listdir(unlable_path)]
    train_data = TrainDataset(train_image_list, train_label_list, ul_image_list)

    f_pwd = os.path.abspath(os.path.dirname(pwd) + os.path.sep + '.')
    panorama_img_path = os.path.join(f_pwd, "caries_data\\Max100Dice\\images_cut")
    panorama_gt_path = os.path.join(f_pwd, "caries_data\Max100Dice\labels_cut")
    panorama_img_path_list = [os.path.join(panorama_img_path, file_name) for file_name in os.listdir(panorama_img_path)]
    panorama_gt_path_list = [os.path.join(panorama_gt_path, file_name) for file_name in os.listdir(panorama_gt_path)]
    val_data = ValDataset(panorama_img_path_list, panorama_gt_path_list)

    model = CariesSSLNet(learning_rate, l_batch_size, theta, SSL_flag)

    idxs = list(range(len(train_image_list + ul_image_list)))
    labeled_len = labeled_ratio[labeld_rate]
    labeled_idxs = idxs[:labeled_len]
    unlabeled_idxs = list(set(idxs) - set(labeled_idxs))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, l_batch_size)

    train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, pin_memory=True) # batch_size must be 1

    max_epoch = 200
    train_process(model, train_loader, val_loader, max_epoch, labeld_rate)

if __name__ == '__main__':
    seed_everything()
    main()
