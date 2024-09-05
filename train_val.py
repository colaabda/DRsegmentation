import torch
import cv2
import random
import numpy as np
import pandas as pd
import tqdm


CSV_FILE='/home/abir/Documents/PROJECTS/seg_folder/pycodes/aug_data/micro_train.csv'
DATA_DIR='/home/abir/Documents/seg-folder'
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
EPOCHS = 100
# LR= 0.001
IMAGE_SIZE=512
BATCH_SIZE=1
ENCODER ='vgg16'                     
WEIGHTS='imagenet'
TEST_CSV='/home/abir/Documents/PROJECTS/seg_folder/pycodes/aug_data/micro_val.csv'

train_df = pd.read_csv(CSV_FILE,encoding='UTF-16')      
train_df=train_df[['Images','mask']]
train_df = train_df.fillna('').astype(str) 

test_df=pd.read_csv(TEST_CSV,encoding='UTF-16')
test_df=test_df[['Images','mask']]



def train_net(trainloader, model, optimizer):
  model.train()
  total_loss = 0.0
  for images, masks in tqdm(trainloader):
    images =images.to(DEVICE)
    masks = masks.to(DEVICE)
    optimizer.zero_grad()
    model_pred=model(images) 
    loss_fn=DiceLoss(sigmoid=True)
    loss=loss_fn(model_pred, masks)
    loss.backward()
    optimizer.step()
    total_loss +=loss.item()
  return total_loss/len(trainloader)

def eval_net(validloader, model):
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for images, masks in tqdm(validloader):
      images =images.to(DEVICE)
      masks = masks.to(DEVICE)
      model_pred=model(images) 
      loss_fn=DiceLoss(sigmoid=True)
      loss=loss_fn(model_pred, masks)
      total_loss +=loss.item()
  return total_loss/len(validloader)