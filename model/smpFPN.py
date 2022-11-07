import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base.heads import SegmentationHead
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from torchvision.utils import _log_api_usage_once
import logging
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import random
from evaluate.utils import recompone_overlap, metric_calculate,  recompone_overlap

def mean_metric(preds: torch.Tensor, target: torch.Tensor):
    
    assert preds.shape == target.shape

    preds = (preds > 0.5).float().view(-1)
    target = (target > 0).float().view(-1)

    TP = (preds * target).sum()
    FN = ((1 - preds) * target).sum()
    TN = ((1 - preds) * (1 - target)).sum()
    FP = (preds * (1 - target)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou = TP / (TP + FP + FN + 1e-4)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-4)
    pre = TP / (TP + FP + 1e-4)
    spe = TN / (FP + TN + 1e-4)
    sen = TP / (TP + FN + 1e-4)

    return acc, iou, dice, pre, spe, sen

class FPN_ToothDataset(Dataset):
    def __init__(self, mode, image_list, label_list):
        self.mode = mode
        self.data_list = []
        for image_path, label_path in zip(image_list, label_list):
            self.data_list.append([image_path, label_path])
        self.img_transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5),
        ])
        self.both_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(45),
        ])
        self.resize_transform = T.Resize((384, 384))
        self.nomalize_transform = T.ToTensor()

        print("data set num:", len(self.data_list))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        [image_path, label_path] = self.data_list[index]
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.mode == "train":
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

logging.basicConfig(filename='train_log.log',
                    level=logging.INFO, filemode='w')

def seg_loss(pred, mask):
    # weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = 1
    wbce = F.binary_cross_entropy_with_logits_with_logits(pred, mask, reduce='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return wbce + (wiou).mean()



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth, block, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        self._depth = depth
        self.out_channels = out_channels
        self.in_channels = in_channels

        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = self.out_channels[1]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.out_channels[2], layers[0])
        self.layer2 = self._make_layer(block, self.out_channels[3], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.out_channels[4], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.out_channels[5], layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            # self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class SkipBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, skip):
        return skip

# class FPNBlock(nn.Module):
#     def __init__(self, pyramid_channels, skip_channels, top_layer=False):
#         super().__init__()
#         self.top_layer = top_layer
#         self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
#         self.gate_conv = nn.Conv2d(pyramid_channels, 1, kernel_size=1)

#     def forward(self, x, skip=None):
#         if self.top_layer:
#             x = self.skip_conv(x)
#             alpha = self.gate_conv(x)
#         else:
#             x = F.interpolate(x, scale_factor=2, mode="nearest")
#             skip = self.skip_conv(skip)
#             alpha = self.gate_conv(skip)
#             x = x + skip
#             # x =  (1 - 0.5 * alpha.sigmoid()) * x + (1 + 0.5 * alpha.sigmoid()) * skip
#         return x, alpha

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, top_layer=False):
        super().__init__()
        self.top_layer = top_layer
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(pyramid_channels, 1, kernel_size=1)

    def forward(self, x, skip=None):
        if self.top_layer:
            x = self.skip_conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = self.skip_conv(skip)
            x = x + skip
        return x



class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, encoder_depth=5, pyramid_channels=256, segmentation_channels=128, dropout=0.2, merge_policy="add"):
        super().__init__()
        self.pyramid_channels = pyramid_channels
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.skip_blocks = nn.ModuleList([SkipBlock() for _ in range(4)])

        self.p5 = FPNBlock(pyramid_channels, encoder_channels[0], top_layer=True)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        [c2, c3, c4, c5] = [skip_block(c) for skip_block, c in zip(self.skip_blocks, features[-4:])]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x, feature_pyramid

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class FPN(SegmentationModel):
    def __init__(self, encoder_depth=5, decoder_pyramid_channels=256, decoder_segmentation_channels=128, \
        decoder_merge_policy="add", decoder_dropout=0.2, in_channels=1, classes=1, activation: Optional[str] = None, upsampling=4, aux_params=None):
        super().__init__()

        self.encoder=ResNetEncoder(
            in_channels=in_channels,
            out_channels=(1, 64, 64, 128, 256, 512),
            depth=encoder_depth,
            block=BasicBlock,
            layers=[3, 4, 6, 3]
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.upsampler =[nn.UpsamplingBilinear2d(scale_factor=factor) for factor in [4, 8, 16, 32]]

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.aux_segmentation_head_list = nn.Sequential(*[SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            kernel_size=1,
            upsampling=4
        ) for _ in range(encoder_depth - 1)])

        self.classification_head = None

        self.name = "fpn-resnet34"
        self.initialize()

    def forward(self, x):
        features = self.encoder(x)
        decoder_output, pyramid_features = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        aux_masks_list = [aux_head(f_p) for aux_head, f_p in zip(self.aux_segmentation_head_list, pyramid_features)]
        return masks, aux_masks_list

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class FPNnet(FPN, pl.LightningModule):
    def __init__(self, in_c: int = 1, c: int = 1):
        super().__init__(in_channels=in_c, classes=c)
        self.learning_rate = 1e-3
        self.max_epoch = 200
        self.dice_loss = BinaryDiceLoss()
        # 这个是内部自己实现 sigmoid的
        self.bce_loss = F.binary_cross_entropy_with_logits
        
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc, iou, dice, pre, spe, sen = mean_metric(y_hat.sigmoid(), y)
        self.log('train_mean_acc', acc, on_epoch=True)
        self.log('train_mean_iou', iou, on_epoch=True)
        self.log('train_mean_dice', dice, on_epoch=True)
        self.log('train_mean_pre', pre, on_epoch=True)
        self.log('train_mean_spe', spe, on_epoch=True)
        self.log('train_mean_sen', sen, on_epoch=True)
        bce_loss = self.bce_loss(y_hat, y)
        dice_loss = self.dice_loss(y_hat.sigmoid(), y)
        self.log('train_bce_loss', bce_loss, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_epoch=True)
        return bce_loss + dice_loss

    def validation_step(self, batch, batch_idx):
        imgs, gt = batch
        imgs = imgs.permute(1, 0, 2, 3)
        outputs = self(imgs)
        pred = torch.sigmoid(outputs)
        pred_imgs = recompone_overlap(pred.cpu().numpy(), 768, 1536, 192, 192)
        acc, iou, dice, pre, spe, sen = metric_calculate(gt.cpu().numpy(), pred_imgs)

        self.log('val_mean_acc', acc, on_step=False, on_epoch=True)
        self.log('val_mean_iou', iou, on_step=False, on_epoch=True)
        self.log('val_mean_dice', dice, on_step=False, on_epoch=True)
        self.log('val_mean_spe', spe, on_step=False, on_epoch=True)
        self.log('val_mean_pre', pre, on_step=False, on_epoch=True)
        self.log('val_mean_sen', sen, on_step=False, on_epoch=True)
    
    # def test_step(self, batch, batch_idx):
    #     x, y, y1, y2, y3, y4 = batch
    #     y_hat, [y1_hat, y2_hat, y3_hat, y4_hat] = self(x)
    #     main_loss_seg = F.binary_cross_entropy_with_logits(y_hat, y)
    #     gate_loss_seg = F.binary_cross_entropy_with_logits(y1_hat, y1) + F.binary_cross_entropy_with_logits(y2_hat, y2) +\
    #          F.binary_cross_entropy_with_logits(y3_hat, y3) + F.binary_cross_entropy_with_logits(y4_hat, y4)
    #     p = 0.5 * self.current_epoch / self.max_epoch
    #     test_loss_seg = (1-p) * main_loss_seg + p * gate_loss_seg
    #     # test_mean_iou = mean_iou(y_hat, y)
    #     self.log('test_loss_seg', test_loss_seg, on_epoch=True)
    #     acc, iou, dice, spe, sen = mean_metric(y_hat, y)
    #     self.log('test_mean_acc', acc, on_epoch=True)
    #     self.log('test_mean_iou', iou, on_epoch=True)
    #     self.log('test_mean_dice', dice, on_epoch=True)
    #     self.log('test_mean_spe', spe, on_epoch=True)
    #     self.log('test_mean_sen', sen, on_epoch=True)
    #     return test_loss_seg
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]
