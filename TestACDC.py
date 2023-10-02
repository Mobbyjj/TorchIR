import numpy as np
import torch
import pytorch_lightning as pl
from torchir.dlir_framework import DLIRFramework
from torchir.metrics import NCC
from torchir.transformers import BsplineTransformer, AffineTransformer
from torchir.networks import DIRNet, AIRNet
from pathlib import Path
from torchir.dataset import ACDCValDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



DEST_DIR = Path('/media/ssd/fanwen/ACDC/DIRNet')


class LitDLIRFramework(pl.LightningModule):
    def __init__(self, only_last_trainable=True):
        super().__init__()
        self.dlir_framework = DLIRFramework(only_last_trainable=only_last_trainable)
        self.add_stage = self.dlir_framework.add_stage
        self.metric = NCC()
        self.saved_test_images = []
    
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
    
    def test_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        self.saved_test_images.append(warped)
        return {"warped": warped}
    
    def test_epoch_end(self, outputs):
        all_warped_images = [o["warped"] for o in outputs]
        self.all_warped = torch.cat(all_warped_images, 0)
    
model = LitDLIRFramework()
model.add_stage(network=AIRNet(kernels=16), transformer=AffineTransformer(ndim=2))
model.add_stage(network=DIRNet(grid_spacing=(8, 8), kernels=16, num_conv_layers=5, num_dense_layers=2),
                transformer=BsplineTransformer(ndim=2, upsampling_factors=(8, 8)))
model.add_stage(network=DIRNet(grid_spacing=(4, 4), kernels=16, num_conv_layers=5, num_dense_layers=2),
                transformer=BsplineTransformer(ndim=2, upsampling_factors=(4, 4)))
model.load_state_dict(torch.load(DEST_DIR / 'ACDC.ckpt')['state_dict'])

test_path = '/media/ssd/fanwen/ACDC/testing/'
test_loader = DataLoader(ACDCValDataset(test_path))

trainer = pl.Trainer(gpus=1)
result = trainer.test(model, test_loader)
warped_img = model.all_warped

plt.imshow(warped_img[0, 0].cpu().numpy(), cmap='gray')





