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


class Net(smp.Unet):
    def __init__(self, in_c: int = 1, out_c: int = 1):
        super().__init__(in_channels=in_c, classes=out_c)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks


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
        self.model_tea = Net()
        self.model_stu = Net()
        for para in self.model_stu.parameters():
            para.detach_()

        self.dice_loss = DiceLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        # self.mse_loss = mse_loss
        self.mse_loss = sigmoid_mse_loss

        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def forward(self, l_x):
        return self.model_tea(l_x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        self.data, self.label = batch
        self.sup_data = self.data[:self.l_batch_size]
        self.sup_label = self.label[:self.l_batch_size]
        # train teacher network
        # if optimizer_idx == 0:

        self.pred = self.model_tea(self.data)
        consistency_loss = 0
        if self.semi_train:
            ul_data = self.data[self.l_batch_size:]
            noise = torch.clamp(torch.randn_like(ul_data) * 0.1, -0.2, 0.2)
            ul_data += noise
            with torch.no_grad():
                ul_pred = self.model_stu(ul_data)  
            T = 8
            volume_batch_r = ul_data.repeat(2, 1, 1, 1)
            (b, c, h, w) = self.label.shape
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, c, h, w]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = self.model_stu(ema_inputs)
            preds = preds.reshape(T, stride, c, h, w)
            preds = torch.mean(preds, dim=0).sigmoid()  #
            uncertainty = -2.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True) #
            consistency_dist = self.mse_loss(self.pred[self.l_batch_size:], ul_pred) #
            threshold = (0.75 + 0.25 * sigmoid_rampup(self.glob_step, 4480)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_loss = torch.sum(mask * consistency_dist)/(2 * torch.sum(mask)+1e-16)

        acc, iou, dice, spe, sen = mean_metric(self.pred[:self.l_batch_size], self.sup_label[:self.l_batch_size])
        self.log('train_mean_acc', acc, on_step=False, on_epoch=True)
        self.log('train_mean_iou', iou, on_step=False, on_epoch=True)
        self.log('train_mean_dice', dice, on_step=False, on_epoch=True)
        self.log('train_mean_spe', spe, on_step=False, on_epoch=True)
        self.log('train_mean_sen', sen, on_step=False, on_epoch=True)
        bce_loss = self.bce_loss(self.pred[:self.l_batch_size], self.sup_label)
        dice_loss = self.dice_loss(self.pred[:self.l_batch_size], self.sup_label)
        seg_loss = 0.5 * (bce_loss + dice_loss)

        # consistency_weight = 0.1
        # if self.semi_train:
        #     consistency_loss = self.mse_loss(ul_pred, self.pred[self.l_batch_size:])
        # else:
        #     consistency_loss = 0
        consistency_weight = get_current_consistency_weight(self.current_epoch, 200)

        self.log('train_consistency_loss', consistency_loss, on_step=False, on_epoch=True)
        
        self.log('train_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True)
        return seg_loss + consistency_weight * consistency_loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch > 150:
            self.eval()
            imgs, gt = batch
            imgs = imgs.permute(1, 0, 2, 3)
            with torch.no_grad():
                outputs = self(imgs)
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
        if self.current_epoch > 150:
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
        else:
            self.log('val_mean_iou', 0)
            self.log('val_mean_dice', 0)
            self.log('val_mean_spe', 0)
            self.log('val_mean_pre', 0)
            self.log('val_mean_sen', 0)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]        

    def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
        alpha = min(1 - 1 / (self.glob_step + 1), self.p)
        for para1, para2 in zip(self.model_stu.parameters(), self.model_tea.parameters()):
            para1 = alpha * para1 + (1 - alpha) * para2  


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def train_process(model, train_loader, val_loader, max_epochs):
    tb_logger = pl_loggers.TensorBoardLogger('.\\Cariouslog\\UAMT')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_mean_dice',
                                        filename='UAMT10-{epoch:02d}-{val_mean_iou:.4f}-{val_mean_dice:.4f}-{val_mean_spe:.4f}-{val_mean_sen:.4f}-{val_mean_pre:.4f}',
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
    learning_rate = 1e-3
    theta = 0.99
    labeled_ratio = {"0.1": 265, "0.2": 530, "0.5": 1325}
    labeld_rate = "0.2"
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
    train_process(model, train_loader, val_loader, max_epoch)

if __name__ == '__main__':
    seed_everything()
    main()
