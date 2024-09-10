def is_black(px_l,p_xl):
    count=0
    for i in range(len(px_l)):
        n=px_l[i];m=p_xl[i]
        if n==m and n==0:
            count+=1
        else:
            n!=m;count=0

    if count==len(px_l):
        return True
    else:
        return False
      
def is_clear_background(pxl2):
    count=0
   
    for i in range(len(pxl2)):
        m=pxl2[i]
        if m==0:
            count+=1
        else:
             m!=0
             count=0
        # break
    if count==len(pxl2):
        return True
    else:
        count!=len(pxl2)
        return False

def binary2multinary(masks):
    msk=np.zeros((1024,1024,3),dtype=np.uint8)
    # h,w=512,512
    clr=[0,0,0]
    # current=0
    for m in range(len(masks)):
        clrs=[[255,255,255],[255,255,0],[255,100,0],[0,255,0]]
        if m == 0: clr=clrs[0]
        if m == 1: clr=clrs[1]
        if m == 2: clr=clrs[2]
        if m == 3: clr=clrs[3]
        # print(len(masks))
        mask=masks[m]
        mask=cv2.resize(mask,(1024,1024))
        h=mask.shape[0]
        w=mask.shape[1]
        # print(f'{m} {h} {w} {clr}')
        for y in range(h):
            for x in range(w):
                clr1=mask[y,x]
                clr2=msk[y,x]
                back_ground=is_black(px_l=clr1,p_xl=clr2)
                if back_ground is True:
                    continue
                else:
                    com=is_clear_background(pxl2=clr2)
                # print('echo')
                    if com is True:
                        msk[y,x]=clr
                        # print('echo')
                    # else:
                    #     com is False
                        
                    #     msk[y,x]=[255,0,0]
    return msk


if __main__=="__name__":
  import pandas as pd
  import cv2
  import numpy as np
  import os
  import tqdm
  mask_list=[]
  mul_df=pd.read_csv('/home/abir/Documents/PROJECTS/seg_folder/train_idrid.csv',encoding='UTF-16')
  img_no=0
  mul_df=mul_df[['Hard_Exudate','Haemorrhages','Microaneurysms','Soft_Exudates']]
  mul_df=mul_df.fillna('').astype(str)
  for i in range(2):
      row = mul_df.iloc[i]
      mask1_path=row['Hard_Exudate']
      mask2_path=row['Haemorrhages']
      mask3_path=row['Microaneurysms']
      mask4_path=row['Soft_Exudates']
      # print(mask3_path)
      # num=000
      if os.path.exists(mask1_path): 
          mk1=cv2.imread(mask1_path);mk1=cv2.cvtColor(mk1,cv2.COLOR_BGR2RGB);mask_list.append(mk1)
              
      if os.path.exists(mask2_path): 
          mk2=cv2.imread(mask2_path);mk2=cv2.cvtColor(mk2,cv2.COLOR_BGR2RGB);mask_list.append(mk2)
  
      if os.path.exists(mask3_path): 
              mk3=cv2.imread(mask3_path);mk3=cv2.cvtColor(mk3,cv2.COLOR_BGR2RGB);mask_list.append(mk3)
  
      if os.path.exists(mask4_path): 
          mk4=cv2.imread(mask4_path);mk4=cv2.cvtColor(mk4,cv2.COLOR_BGR2RGB);mask_list.append(mk4)  
  
      # mk1=cv2.resize(mask2_path,(512,512));mask_list.append(mk1)
      # mk2=cv2.resize(mk2,(512,512));mask_list.append(mk2)
      print(len(mask_list))
      combined_mask=binary2multinary(masks=mask_list)
  
      cv2.imwrite('/home/abir/Documents/PROJECTS/seg_folder/multinary_train/IDRiD_'+str(img_no)+'.png',combined_mask)
      del combined_mask
      img_no+=1;print('file saved')
  
      # cv2.imwrite(f'/home/abir/Documents/PROJECTS/seg_folder/multi_mask/mulmsk',finalmsk)
      # break
