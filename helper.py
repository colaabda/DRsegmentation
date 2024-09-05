import sys
sys.path.append('/home/abir/Documents/seg-folder')
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def show_image(image,mask,pred_image = None):

    if pred_image == None:

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')

    elif pred_image != None :

        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))

        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')

        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(),cmap = 'gray')




def inspect_dataset(idx,data):
    index=idx
    if data=='train':
        row = train_df.iloc[index]
    if data=='test':
        row = test_df.iloc[index]

    image_path= row['Images']  #use[] instead of (row.'mask')
    mask_path = row['mask']
    if image_path[-16:]==mask_path[-16:]:
        print("match")
    else:
        print(f"No match{image_path[-16:]} {mask_path[-16:]}")
    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR image read by cv2 to RGB
    mask=cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)     #/255.0
   
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.set_title('IMAGE')
    ax1.imshow(image)
    
    ax2.set_title('GROUND TRUTH')
    ax2.imshow(mask,cmap = 'gray')



def get_train_augs():
  return A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE,always_apply=True,),
                   A.HorizontalFlip(p=0.5),
                   A.VerticalFlip(p=0.5),],is_check_shapes=False) # is_check_shapes=False
def get_val_augs():
  return A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE,always_apply=True,),],is_check_shapes=False) #is_check_shapes=False


def check_dataloaders(index,loader):

    idx2=index
    if loader=='trainset':
        image,mask=trainset[idx2] 
    if loader=='validet':
        image, mask=validset[idx2]
    # plt.imshow(image,mask)

    f, (ay1, ay2) = plt.subplots(1, 2, figsize=(10,5))

    ay1.set_title('image')
    ay1.imshow(image.permute(1,2,0).squeeze(),cmap='gray')

    ay2.set_title('ground_truth')
    ay2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')


def precision_score_(mask, pred_mask):
    
    intersect = torch.sum(pred_mask*mask)
    total_pixel_pred = torch.sum(pred_mask)
    precision = torch.mean(intersect/total_pixel_pred)
    # print(precision)
    return round(precision.item(), 3)

def recall_score_(mask, pred_mask):
    intersect = torch.sum(pred_mask*mask)
    total_pixel_truth = torch.sum(mask)
    if total_pixel_truth == 0:
        return 0.0
    recall = torch.mean(intersect/total_pixel_truth)
    return round(recall.item(), 3)

def accuracy(mask, pred_mask):
    intersect = torch.sum(pred_mask*mask)
    union = torch.sum(pred_mask) + torch.sum(mask) - intersect
    xor = torch.sum(mask==pred_mask)
    acc = torch.mean(xor/(union + xor - intersect))
    return round(acc.item(),3)

def dice_coef(mask, pred_mask):
    intersect = np.sum(pred_mask*mask)
    total_sum = np.sum(pred_mask) + np.sum(mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3) #round up to 3 decimal places

def iou_(mask, pred_mask):
    intersect = torch.sum(pred_mask*mask)
    union = torch.sum(pred_mask) + torch.sum(mask) - intersect
    if union == 0:
        return 1.0 
    iou = torch.mean(intersect/union)
    return round(iou.item(), 3)

def f1_score_(mask,pred_mask):
    precision=precision_score_(mask,pred_mask)
    recall=recall_score_(mask,pred_mask)
    F1_score = 2 * (precision * recall) / (precision + recall)
    F1_score=round(F1_score, 3)
    return F1_score