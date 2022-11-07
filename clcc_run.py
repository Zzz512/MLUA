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
from evaluate.utils import recompone_overlap, metric_calculate, get_data_test_overlap, recompone_overlap, rgb2gray
from medpy import metric

gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)


parser = argparse.ArgumentParser(description='CariesNet')
parser.add_argument('--sigma', '-s', type=float, default=5)
args = parser.parse_args()


class TrainDataset(Dataset):
    def __init__(self, image_list, label_list, ul_image_list = None, transize = 384):
        self.transize = transize
        self.data_list = []
        for image_path, label_path in zip(image_list, label_list):
            self.data_list.append([image_path, label_path])
        if ul_image_list is not None:
            for ul_image_path in ul_image_list:
                self.data_list.append([ul_image_path, None])
        self.img_transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5),
        ])
        self.both_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(45),
        ])
        self.resize_transform = T.Resize((self.transize, self.transize))
        self.nomalize_transform = T.ToTensor()
        print("data set num:", len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        [image_path, label_path] = self.data_list[index]
        image = Image.open(image_path)
        if label_path is not None:
            label = Image.open(label_path)
        else:
            label = Image.fromarray(np.zeros((self.transize, self.transize)))
        seed = random.randint(0, 10000)
        torch.random.manual_seed(seed)
        image = self.both_transform(image)
        torch.random.manual_seed(seed)
        label = self.both_transform(label)
        image = self.img_transform(image)
        image = self.resize_transform(image)
        label = self.resize_transform(label)
        image = self.nomalize_transform(image)
        label = self.nomalize_transform(label)
        image = torch.tensor(np.array(image), dtype=torch.float32)
        label = torch.tensor(np.array(label), dtype=torch.float32)
        return image, label

class ValDataset(Dataset):
    # (1, 768, 1536)  (21, 384, 384)
    def __init__(self, img_path_list, gt_path_list):
        self.img_path_list = img_path_list
        self.gt_path_list = gt_path_list
        self.resize_transform = T.Resize((384, 384))
        self.nomalize_transform = T.ToTensor()

    def __len__(self):
        return len(self.img_path_list)
    
    def normalize(self, inputs):
        return (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
    
    def __getitem__(self, index):
        # 两个都是 (1, 768, 1536) 
        img_path = self.img_path_list[index]
        gt_path = self.gt_path_list[index]
        imgs_patch, _, _, gt = get_data_test_overlap(img_path, gt_path, 384, 384, 192, 192)
        assert imgs_patch.shape[0] == 21, "没有切成21个patch"
        imgs_patch = rgb2gray(imgs_patch)
        final_img = torch.zeros(21, 384, 384)
        for i in range(imgs_patch.shape[0]):
            image = Image.fromarray(np.uint8(imgs_patch[i].squeeze()))
            # 然后都变成  (384,384) 
            image = self.resize_transform(image)
            image = self.nomalize_transform(image)
            final_img[i] = torch.tensor(np.array(image).squeeze(), dtype=torch.float32)  # torch.Size([384, 384])
        gt = self.normalize(gt)
        final_gt = torch.tensor(gt.squeeze(), dtype=torch.float32)
    
        return final_img, final_gt

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, l_indices, ul_indices, batch_size, l_batch_size):
        self.l_indices = l_indices  # * self.cfg.DATA.REPEAT
        self.ul_indices = ul_indices
        self.l_batch_size = l_batch_size
        self.ul_batch_size = batch_size - l_batch_size
        assert len(self.l_indices) >= self.l_batch_size > 0
        assert len(self.ul_indices) >= self.ul_batch_size >= 0

    def __iter__(self):
        label_iter = iterate_once(self.l_indices)
        unlabel_iter = iterate_eternally(self.ul_indices)
        if self.ul_batch_size == 0:
            return (l_batch + l_batch for (l_batch, l_batch)in zip(grouper(label_iter, self.l_batch_size), grouper(label_iter, self.l_batch_size)))
        return (l_batch + ul_batch for (l_batch, ul_batch) in zip(grouper(label_iter, self.l_batch_size), grouper(unlabel_iter, self.ul_batch_size)))

    def __len__(self):
        return len(self.l_indices) // self.l_batch_size


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, pred, mask):
        # mask = mask.unsqueeze(dim=1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        mask_flat = mask.view(size, -1)
        intersection = pred_flat * mask_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + mask_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return (wbce + dice_loss).mean()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, patch_outputs, output):
        bs = output.shape[0]
        cls = output.shape[1]
        psz = patch_outputs.shape[-1]
        cn = output.shape[-1] // psz

        patch_outputs = patch_outputs.reshape(bs, cn, cn, cls, psz, psz)
        output = output.reshape(bs, cls, cn, psz, cn, psz).permute(0, 2, 4, 1, 3, 5)

        p_output_soft = torch.sigmoid(patch_outputs)
        outputs_soft = torch.sigmoid(output)

        loss = torch.mean((p_output_soft - outputs_soft) ** 2, dim=(0, 3, 4, 5)).sum()

        return loss

class Net(smp.Unet):
    def __init__(self, in_c: int = 1, out_c: int = 1):
        super().__init__(in_channels=in_c, classes=out_c)
        self.seg_loss = BCEDiceLoss()
        self.aux_proj = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        feature_map = self.aux_proj(decoder_output)
        masks = self.segmentation_head(decoder_output)
        return feature_map, masks


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
        self.model = Net()

        # loss
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.dice_loss = DiceLoss()
        self.contrast_loss = ContrastiveLoss()
        self.consist_loss = ConsistencyLoss()

        #evaluate result
        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def cropImage(self, image):
        # input torch.Size([8, 1, 384, 384])
        patch_size = 128
        # torch.Size([288, 1, 64, 64])    
        image_patch = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).permute(
            0, 2, 3, 1, 4, 5).reshape(-1, 1, patch_size, patch_size) 
        
        return image_patch

    def reshapeFeatMap(self, feat_map_patch , batch_size):
        # proj_final torch.Size([288, 128, 1, 1]
        num = 3
        feat_map_patch = feat_map_patch.reshape(batch_size, num, num, 128 , 1, 1).permute(
            0, 3, 1, 4, 2, 5).reshape(8, 128, num, num)   # torch.Size([8, 128, 6, 6])
        return feat_map_patch


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        image, label = batch
        batch_size = image.shape[0]
        feat_map, pred = self.model(image)
        image_patch = self.cropImage(image)
        feat_map_patch, pred_patch = self.model(image_patch)
        feat_map_patch_reshape = self.reshapeFeatMap(feat_map_patch, batch_size)

        bce_loss = self.bce_loss(pred[:self.l_batch_size], label[:self.l_batch_size])
        dice_loss = self.dice_loss(pred[:self.l_batch_size], label[:self.l_batch_size])
        seg_loss = 0.5 * (bce_loss + dice_loss)
        contrast_loss = 0
        consist_loss = 0
        if self.semi_train:
            if self.current_epoch < 100:
                contrast_loss = self.contrast_loss(feat_map, feat_map_patch_reshape)
            else:
                consist_loss = self.consist_loss(pred_patch, pred)
        weight = sigmoid_rampup(self.current_epoch, 200)
        self.log('train_contrast_loss', contrast_loss, on_step=False, on_epoch=True)
        self.log('train_consist_loss', consist_loss, on_step=False, on_epoch=True)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True)
        return seg_loss + weight * (contrast_loss + consist_loss)

    def validation_step(self, batch, batch_idx):
        self.eval()
        imgs, gt = batch
        imgs = imgs.permute(1, 0, 2, 3)
        with torch.no_grad():
            _, outputs = self(imgs)
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
    
    # def on_train_epoch_end(self):
    #     sample_imgs = self.sup_data[0]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("train image", grid)
    #     sample_imgs = self.pred.sigmoid()[0]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("train label", grid)

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
    #     alpha = min(1 - 1 / (self.glob_step + 1), self.p)
    #     for para1, para2 in zip(self.model_stu.parameters(), self.model_tea.parameters()):
    #         para1 = alpha * para1 + (1 - alpha) * para2  

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.generator.model[0].weight)
    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def train_process(model, train_loader, val_loader, max_epochs):
    tb_logger = pl_loggers.TensorBoardLogger('Cariouslog/')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_mean_dice',
                                        filename='CLCC10-{epoch:02d}-{val_mean_dice:.4f}',
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
    train_process(model, train_loader, val_loader, max_epoch)


if __name__ == '__main__':
    seed_everything()
    main()

