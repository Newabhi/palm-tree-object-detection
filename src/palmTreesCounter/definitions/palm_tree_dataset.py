from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class PalmTreeDataset(Dataset):
    def __init__(self, dataframe, img_dir,  target_size: tuple, transforms=None, train=True):
        self.data = dataframe
        self.img_dir = img_dir
        self.transforms = transforms
        self.target_size = target_size
        self.train = train

        if not self.train:
            palm_df = self.data[self.data['class'] == 'Palm']
            self.unique_image_counts = palm_df.groupby('filename').size().reset_index(name='count')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert("RGB") 
        
        # Resize the image
        img = img.resize(self.target_size, Image.BILINEAR)
        width, height = self.target_size
        
        if self.train:
            # Bounding box normalization
            xmin = (row['xmin'] / row['width']) * width 
            xmax = (row['xmax'] / row['width']) * width
            ymin = (row['ymin'] / row['height']) * height
            ymax = (row['ymax'] / row['height']) * height

            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.tensor([1 if row['class'] == 'Palm' else 2], dtype=torch.int64)

            target = {'boxes': boxes, 'labels': labels}

            if self.transforms:
                img = self.transforms(img)

            return img, target
        
        if self.transforms:
            img = self.transforms(img)
        
        # Get the count for the filename
        filename = row['filename']
        palm_trees_count_series = self.unique_image_counts[self.unique_image_counts['filename'] == filename]['count']
        
        # Extract the count value, default to 0 if not found
        palm_trees_count = torch.tensor(palm_trees_count_series.iloc[0] if not palm_trees_count_series.empty else 0, dtype=torch.int64)
        
        return img, palm_trees_count