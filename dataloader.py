from torch.utils.data import Dataset
import cv2

class PersonSegmentationDataset(Dataset):
  def __init__(self, df, augmentations):
    self.df=df
    self.augmentations=augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = train_df.iloc[index]
    image_path = row['Images']
    mask_path = row['mask']

    image= cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR image read by cv2 to RGB

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #mask is in [ h, w] form add channel to it [h, w,c]
    mask = np.expand_dims(mask, axis=-1)

    # Apply albumentations is applicable
    if self.augmentations:
      data=self.augmentations(image=image, mask=mask)
      image = data['image']
      mask = data['mask']

    # convert  image and mask from [h,w,c]--> [c,h,w]
    image= np.transpose(image,(2,0,1))
    mask= np.transpose(mask,(2,0,1))

    # convert them to torch tensor and normalise

    image = torch.Tensor(image)/255.0
    mask = torch.Tensor(mask)/255.0

    return image, mask