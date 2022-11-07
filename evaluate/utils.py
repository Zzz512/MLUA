from evaluate.common import readImg, readLabel
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import torch

class ToothDatasetTest(Dataset):
    # (21, 3, 384, 384)  (21, 1, 384, 384)
    def __init__(self, mode, patches_image, patches_label):
        self.mode = mode
        self.patches_image = rgb2gray(patches_image)
        self.patches_label = patches_label
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

    def __len__(self):
        return self.patches_image.shape[0]
    
    def normalize(self, input):
        input = (input - input.min()) /(input.max() - input.min() + 1e-6)
        return input
    
    def __getitem__(self, index):
        # 两个都是 (1, 384, 384) 
        image = self.patches_image[index,...]
        image = Image.fromarray(np.uint8(image.squeeze(0)))
        label = self.patches_label[index,...]
        label = Image.fromarray(np.uint8(label.squeeze(0)))
        
        # 然后都变成  (384,384) 
        image = self.resize_transform(image)
        label = self.resize_transform(label)
        image = self.nomalize_transform(image)
        # cv2.imshow("img", np.uint8(image[0]* 255))
        # cv2.waitKey(0)
        image = torch.tensor(np.array(image), dtype=torch.float32)  # torch.Size([1, 384, 384])
        # 二分类
        label = torch.tensor(np.array(label)[None, :, :]/255, dtype=torch.float32)  # torch.Size([1, 384, 384])

        return image, label

    def binary_loader(self,path):
        with open(path,'rb') as f:
            img = Image.open(f)
            return img.convert('L')

# 读取test数据,test数据切分按重复 一半的方式 切分 将patch存储进json 并。
def get_data_test_overlap(test_img, test_gt, patch_height = 384, patch_width = 384 , stride_height = 192, stride_width = 192):
    imgs = None
    groundTruth = None
    pic_index = test_img.split('/')[-1]
    img = np.asarray(readImg(test_img))  # PIL 读取 rgb格式图像
    gt = np.asarray(readLabel(test_gt))
    if len(gt.shape)==3:
        gt = gt[:,:,0]
    # imgs (1, 784, 1536, 3)  groundTruth (1, 784, 1536)
    imgs = np.expand_dims(img,0) 
    groundTruth = np.expand_dims(gt,0) 

    
    if(np.max(groundTruth) == 1):
        groundTruth = groundTruth * 255
    #Convert the dimension of imgs to [N,C,H,W]
    imgs = np.transpose(imgs,(0,3,1,2))  # imgs.shape (1, 3, 768, 1536)
    groundTruth = np.expand_dims(groundTruth,1)  # groundTruth.shape (1, 1, 768, 1536)  max = 255

    # Test images : [271.png], shape: (1, 3, 768, 1536), vaule range (4 - 255):
    # Test gts : [271.png], shape: (1, 1, 768, 1536), vaule range (0 - 255):
    test_imgs = paint_border_overlap(imgs,patch_height ,patch_width, stride_height, stride_width)
    test_gts = paint_border_overlap(groundTruth,patch_height ,patch_width, stride_height, stride_width)

    # extract the test patches from the all test images 开始切块  按 384×384 的patch  宽高都为 192 的重复
    test_imgs_patches = extract_ordered_overlap(test_imgs,patch_height ,patch_width, stride_height, stride_width)
    test_gts_patches = extract_ordered_overlap(test_gts,patch_height ,patch_width, stride_height, stride_width)
    
    # test_imgs_patches[0].shape (3, 384, 384)  test_gts_patches[0].shape (1, 384, 384)
    return test_imgs_patches, test_gts_patches, test_imgs, test_gts

# 滑动窗口裁剪成测试数据集
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the image  768
    img_w = full_imgs.shape[3] # width of the image  1536
    leftover_h = (img_h-patch_h) % stride_h  #leftover on the h dim  2  =》0
    leftover_w = (img_w-patch_w) % stride_w  #leftover on the w dim  6  =》0
    if (leftover_h != 0):  #change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " +str(stride_h))
        # print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " +str(leftover_h))
        print("So the H dim will be padded with additional " +str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],img_h+(stride_h-leftover_h),img_w))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:img_h,0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        print("the side W is not compatible with the selected stride of " +str(stride_w))
        # print("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " +str(leftover_w))
        print("So the W dim will be padded with additional " +str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0],full_imgs.shape[1],full_imgs.shape[2],img_w+(stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0],0:full_imgs.shape[1],0:full_imgs.shape[2],0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    # print("new padded images shape: " +str(full_imgs.shape))
    return full_imgs


# Extract test image patches in order and overlap
def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):  # test_imgs.shape (20, 1, 592, 576)
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[1]==1 or full_imgs.shape[1]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[2]  #height of the full image
    img_w = full_imgs.shape[3] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers 34×33 = 1122
    N_patches_tot = N_patches_img*full_imgs.shape[0] # 1122 × 20 = 22440
    patches = np.empty((N_patches_tot,full_imgs.shape[1],patch_h,patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches


# recompone the prediction result patches to images
# 768 1536 192 192
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):  # (21, 1, 384, 384)
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1  # 3
    N_patches_w = (img_w-patch_w)//stride_w+1  # 7
    N_patches_img = N_patches_h * N_patches_w  # 21
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img  # 1


    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    # 他是用了一个计数器来记录每个像素点被预测了多少次 然后所有的patch预测加起来/对应的计算次数矩阵就可以了
    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1): # 34
            for w in range((img_w-patch_w)//stride_w+1): # 33
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k] # Accumulate predicted values
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1  # Accumulate the number of predictions
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0) 
    final_avg = full_prob/full_sum # Take the average
    # print(final_avg.shape)
    final_avg[final_avg > 1.0] = 1.0
    final_avg[final_avg < 0.0] = 0.0
    # final_avg.shape (1, 1, 768, 1536)
    return final_avg

#convert RGB image in black and white  他是用纯原生的办法来这么做的 用cv2 或者 PIL 库等都可以的
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs


def metric_calculate(target: np.ndarray, prediction: np.ndarray):

    target = np.uint8(target.flatten() > 0.5)
    prediction = np.uint8(prediction.flatten() > 0.5)
    TP = (prediction * target).sum()
    FN = ((1 - prediction) * target).sum()
    TN = ((1 - prediction) * (1 - target)).sum()
    FP = (prediction * (1 - target)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-4)
    iou = TP / (TP + FP + FN + 1e-4)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-4)
    pre = TP / (TP + FP + 1e-4)
    spe = TN / (FP + TN + 1e-4)
    sen = TP / (TP + FN + 1e-4)

    return acc, iou, dice, pre, spe, sen
