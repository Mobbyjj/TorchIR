# conda activate midir

import numpy as np
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchir.networks import DIRNet, AIRNet
from torchir.metrics import NCC
from torchir.transformers import BsplineTransformer, AffineTransformer
from torchir.dataset import ACDCTrainDataset, ACDCValDataset
from pathlib import Path
from torchir.dlir_framework import DLIRFramework

# define the dataloader of ACDC

train_path = '/media/ssd/fanwen/ACDC/training/'
val_path = '/media/ssd/fanwen/ACDC/validation/'
batch_size = 20

train_loader = DataLoader(ACDCTrainDataset(train_path), batch_size)
val_loader = DataLoader(ACDCValDataset(val_path), batch_size)

# This is single-layer B-spline
class LitDIRNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        grid_spacing = (8, 8)
        self.dirnet = DIRNet(kernels=16, grid_spacing=grid_spacing)
        self.bspline_transformer = BsplineTransformer(ndim=2, upsampling_factors=grid_spacing)
        self.metric = NCC()
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = torch.optim.Adam(self.dirnet.parameters(), lr=lr, amsgrad=True)
        return optimizer

    def forward(self, fixed, moving):
        params = self.dirnet(fixed, moving)
        warped = self.bspline_transformer(params, fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/validation', loss)
        return loss  
    
# This is the affine network
class LitAIRNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.airnet = AIRNet(kernels=16)
        self.global_transformer = AffineTransformer(ndim=2)
        self.metric = NCC()
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = torch.optim.Adam(self.airnet.parameters(), lr=lr, amsgrad=True)
        return optimizer

    def forward(self, fixed, moving):
        parameters = self.airnet(fixed, moving)
        warped  = self.global_transformer(parameters, fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/validation', loss)
        return loss  


# This is coarse-to-fine network
class LitDLIRFramework(pl.LightningModule):
    def __init__(self, only_last_trainable=True):
        super().__init__()
        self.dlir_framework = DLIRFramework(only_last_trainable=only_last_trainable)
        self.add_stage = self.dlir_framework.add_stage
        self.metric = NCC()
    
    def configure_optimizers(self):
        lr = 0.001
        weight_decay = 0
        optimizer = torch.optim.Adam(self.dlir_framework.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        return {'optimizer': optimizer}

    def forward(self, fixed, moving):
        warped = self.dlir_framework(fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/validation', loss)
        return loss  
    
DEST_DIR = Path('/media/ssd/fanwen/ACDC/DIRNet')
# initialize the model and add an affine registration layer. 
# model = LitDIRNet()
model = LitDLIRFramework()
model.add_stage(network=AIRNet(kernels=16), transformer=AffineTransformer(ndim=2))
model.add_stage(network=DIRNet(grid_spacing=(8, 8), kernels=16, num_conv_layers=5, num_dense_layers=2),
                transformer=BsplineTransformer(ndim=2, upsampling_factors=(8, 8)))
model.add_stage(network=DIRNet(grid_spacing=(4, 4), kernels=16, num_conv_layers=5, num_dense_layers=2),
                transformer=BsplineTransformer(ndim=2, upsampling_factors=(4, 4)))

trainer = pl.Trainer(default_root_dir=DEST_DIR,
                     log_every_n_steps = 20,
                     val_check_interval = 20,
                     max_epochs = 100,
                     gpus = 1)
trainer.fit(model, train_loader, val_loader)

trainer.save_checkpoint(DEST_DIR / 'ACDC.ckpt')






