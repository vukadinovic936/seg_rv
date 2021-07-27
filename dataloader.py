import glob
import os
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import torchvision.transforms as T
from ast import literal_eval
from utils import *

class Echo_RV_Image(data.Dataset):

	def __init__(self, folder_path="/home/dockeruser/Documents/NAS/RV-Milos/RV_images", split="train"):
		super(Echo_RV_Image, self).__init__()

		df = pd.read_csv(os.path.join(folder_path, 'FileList.csv'))
		self.folder_path = folder_path	
		self.file_names = list(df[df['Split']==split]['FileName'])
		self.split = split


	def __getitem__(self, index):
			img_path = os.path.join(self.folder_path, "Images" ,self.file_names[index])
			mask_path = os.path.join(self.folder_path, "Masks", self.file_names[index])

			data = Image.open(img_path)
			label = Image.open(mask_path).convert('L')

			if self.split=="train":
				data, label = augment(data, label)

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



class Echo_FAC(data.Dataset):
	""" Returns pairs of frames and their labeled FAC """
	def __init__(self, folder_path="/home/dockeruser/Documents/NAS/RV-Milos/RV_images", split="test"):
		super(Echo_FAC, self).__init__()
		self.folder_path = folder_path
		self.images_path = os.path.join(self.folder_path, "Images")
		self.mask_path = os.path.join(self.folder_path, "Masks")
		self.split = split
		df = pd.read_csv(f"{folder_path}/VideoList.csv")

		# get as array
		df['Tracings'] = df['Tracings'].apply(literal_eval)

		# get relevant frames
		pair_frames = []
		for i, row in df.iterrows():
			if len(row['Tracings']) == 2:
				name = row['FileName']
				frame1 = row['Tracings'][0]
				frame2 = row['Tracings'][1]
				if(row['Split']==split):
					pair_frames.append([ f"{name}_{frame1}" , f"{name}_{frame2}"])
		
		self.pair_frames = pair_frames

	def __getitem__(self, index):

		pair = self.pair_frames[index]

		mask1_path = os.path.join(self.mask_path, f"{pair[0]}.png")
		mask2_path = os.path.join(self.mask_path, f"{pair[1]}.png")
		mask1 = Image.open(mask1_path)
		mask2 = Image.open(mask2_path)
		area1 = np.sum((np.array(mask1)>0).astype('uint8'))
		area2 = np.sum((np.array(mask2)>0).astype('uint8'))
		real_FAC = ((max(area1,area2)-min(area1,area2))/max(area1,area2))

		frame1_path = os.path.join(self.images_path, f"{pair[0]}.png")
		frame2_path = os.path.join(self.images_path, f"{pair[1]}.png")

		frame1 = Image.open(frame1_path)
		frame2 = Image.open(frame2_path)

		return self.transform_input(frame1), self.transform_input(frame2), real_FAC

	transform_input = T.Compose([
		T.Resize((224,224)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def __len__(self):
		return len(self.pair_frames)
