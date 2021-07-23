from dataloader import Echo_RV_Image
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger 
from utils import *

class DeepLab(pl.LightningModule):
	def __init__(self):
		super().__init__()

		model = models.segmentation.deeplabv3_resnet101(pretrained=True,progress=True)
		model.classifier = DeepLabHead(2048,1)
		self.deeplab = model

	def forward(self,x):
		return self.deeplab(x)['out']

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		return optimizer

	def training_step(self,train_batch,batch_idx):
		x,y = train_batch
		preds = self.forward(x)
		loss = F.binary_cross_entropy_with_logits(preds,y)

		self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_dice', dice_metric( (preds>0.).float(),y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x,y = val_batch
		preds = self.forward(x)
		loss = F.binary_cross_entropy_with_logits(preds,y)

		self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_dice', dice_metric( (preds>0.).float(),y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return loss
		

if __name__ == "__main__":

	train_dataset = Echo_RV_Image()
	train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=8,drop_last=True)

	val_dataset = Echo_RV_Image(split="val")
	val_loader = DataLoader(dataset=val_dataset,batch_size=32,shuffle=False,num_workers=8,drop_last=True)

	test_dataset = Echo_RV_Image(split="test")
	test_loader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=8)

	
	wandb.init(project="rv-image-segmentation",reinit=True)
	wandb_logger = WandbLogger()
	model = DeepLab()
	trainer = pl.Trainer(gpus = [0],
						 precision=16,
						 logger= wandb_logger,
#						 limit_train_batches=0.01,
#						 limit_val_batches=0.0,
						 max_epochs=40)
	trainer.fit(model,train_loader,val_loader)

	# visualize results
	sample, mask = iter(train_loader).next()
	sample_preds = model(sample)

	viz_table = wandb.Table(columns=["image", "predicted_mask","real_mask"])
	for idx in range(len(sample[0])):
		viz_table.add_data(wandb.Image(sample[idx]), wandb.Image( (sample_preds[idx]>0.).float() ), wandb.Image(mask[idx]))
	wandb.log({"deeplab_predictions": viz_table})