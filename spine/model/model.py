from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryStatScores
import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
import numpy as np
from .enc2d3d import Enc2d3d


class RSNASpineLightningModule(pl.LightningModule):
    def __init__(self, lr: float, max_steps: int):
        super().__init__()
        # save_hyperparameters() is used to specify which init arguments should 
        # be saved in the checkpoint file to be used to instantiate the model
        # from the checkpoint later.
        self.save_hyperparameters()


        # self.base_model = base_model
        # self.neck = neck
        # self.classification_head  = AbdTraumaClassificationHead(out_features)
        # self.loss_module = AbdTraumaLoss()
        self.model = Enc2d3d(pretrained=True)
        self.validation_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.hparams.max_steps, num_warmup_steps=0
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
            output_type = ('loss',)
        else:
            output_type = ('infer', 'loss')

        output = self.model(batch, output_type)
        loss: torch.Tensor = output["loss"]
        to_log = {
            f'{mode}/loss': output["loss"].detach().item(),
            f'{mode}/heatmap_loss': output["heatmap_loss"].detach().item(),
            f'{mode}/grade_loss': output["grade_loss"].detach().item(),
            }
        
        if mode == 'val':
            self.validation_step_outputs.append(
                (
                    output["grade"].detach().cpu().numpy(),
                )
            )

        print(to_log)
        
        self.log_dict(
            to_log,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        
        return loss
        
    def on_validation_epoch_end(self):
        grades = np.concatenate([x[0] for x in self.validation_step_outputs])
        # preds = np.concatenate([x[1] for x in self.validation_step_outputs])
        # losses = np.array([x[2] for x in self.validation_step_outputs])
        # loss = losses.mean()        
        self.validation_step_outputs.clear()

    def predict(self, batch):
        return self.model(batch, ('infer',))["grade"].detach().cpu()