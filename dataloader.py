import glob
import os
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import torchvision.transforms as T

class Echo_RV_Image(data.Dataset):

	def __init__(self, folder_path="/workspace/data/NAS/RV-Milos/RV_images", split="train"):
		super(Echo_RV_Image, self).__init__()

		df = pd.read_csv(os.path.join(folder_path, 'FileList.csv'))
		self.folder_path = folder_path	
		self.file_names = list(df[df['Split']==split]['FileName'])


	def __getitem__(self, index):
			img_path = os.path.join(self.folder_path, "Images" ,self.file_names[index])
			mask_path = os.path.join(self.folder_path, "Masks", self.file_names[index])

			data = Image.open(img_path)
			label = Image.open(mask_path).convert('L')

			return self.transform_input(data),self.transform_mask(label)

	transform_input = T.Compose([
			T.Resize((224,224)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

	transform_mask = T.Compose([
			T.Resize((224,224)),
			T.ToTensor()])

	def __len__(self):
		return len(self.file_names)