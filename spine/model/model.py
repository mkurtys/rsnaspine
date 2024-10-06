from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchmetrics import MetricCollection
# from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryStatScores
from torcheval.metrics import MulticlassAccuracy
import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
import numpy as np
from .enc2d3d import Enc2d3d
from .enc2d3d_retina import Enc2d3dRetina


class RSNASpineLightningModule(pl.LightningModule):
    def __init__(self, lr: float, num_training_steps: int, num_cycles: float):
        super().__init__()
        # save_hyperparameters() is used to specify which init arguments should 
        # be saved in the checkpoint file to be used to instantiate the model
        # from the checkpoint later.
        self.save_hyperparameters()
        self.model =  Enc2d3dRetina(pretrained=True) # Enc2d3d(pretrained=True) # Enc2d3dRetina(pretrained=True) # Enc2d3d(pretrained=True)
        self.validation_step_outputs = []
        self.valid_accuracy= MulticlassAccuracy(num_classes=3)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.hparams.num_training_steps,
            num_cycles=self.hparams.num_cycles, num_warmup_steps=0
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, batch, output_types=('infer', 'loss')):
        return self.model(batch, output_types)
        
    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, 'train')
        
    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, 'val')
        
    def __share_step(self, batch, mode: str):
        if mode == 'train':
            output_type = ('loss' )
        else:
            output_type = ('infer', 'loss')

        output = self.model(batch, output_type)
        loss: torch.Tensor = output["loss"]
        to_log = {
            f'{mode}/loss': output["loss"].detach().item(),
            f'{mode}/heatmap_loss': output["heatmap_loss"].detach().item(),
            f'{mode}/grade_loss': output["grade_loss"].detach().item(),
            f'{mode}/zxy_loss': output["zxy_loss"].detach().item(),
            }
        
        if mode == 'val':
            self.valid_accuracy.update(output["grade"].detach().flatten(end_dim=1),
                                       batch["grade"].detach().flatten())
            self.validation_step_outputs.append(
                (
                    output["grade"].detach().cpu().numpy(),
                )
            )
        
        self.log_dict(
            to_log,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=len(batch["D"])
        )
        
        return loss
        
    def on_validation_epoch_end(self):
        grades = np.concatenate([x[0] for x in self.validation_step_outputs])
        # accuracy = 
        #self.validation_step_outputs.clear()
        self.log("val/accuracy", self.valid_accuracy.compute().cpu().item())
        self.validation_step_outputs.clear()
        self.valid_accuracy.reset()

    def predict_step(self, batch, batch_idx):
        return self.model(batch, ('infer',))["grade"].detach().cpu()