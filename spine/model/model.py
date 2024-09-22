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

class RSNASpineLightningModule(pl.LightningModule):
    def __init__(self,
                 base_model,
                 neck,
                 loss
                ):
        super().__init__()
        # save_hyperparameters() is used to specify which init arguments should 
        # be saved in the checkpoint file to be used to instantiate the model
        # from the checkpoint later.
        self.save_hyperparameters(ignore=['base_model'])


        self.base_model = base_model
        self.neck = neck
        # self.classification_head  = AbdTraumaClassificationHead(out_features)
        # self.loss_module = AbdTraumaLoss()

        multiclass_metrics = MetricCollection([
            MulticlassAccuracy(3), MulticlassPrecision(3), MulticlassRecall(3)
        ])
        binary_metrics = MetricCollection([
            BinaryAccuracy(), BinaryPrecision(), BinaryRecall()
        ])

        self.train_metrics = binary_metrics.clone(prefix='train/injury_')
        self.valid_metrics = binary_metrics.clone(prefix='val/injury_')
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        
    def forward(self, x):
        x = self.base_model(x)
        if self.neck:
            x = self.neck(x)
        bowel, extravasation, kidney, liver, spleen = self.classification_head(x)
        return  bowel, extravasation, kidney, liver, spleen


    def predict(self, x):
        bowel, extravasation, kidney, liver, spleen = self.forward(x)
        return F.sigmoid(bowel), F.sigmoid(extravasation), F.softmax(kidney), F.softmax(liver), F.softmax(spleen)

    
    def training_step(self, batch, batch_idx):
        
        imgs = batch["image"]
        batch_size = imgs.shape[0]
        bowel_target, extravasation_target, kidney_target, liver_target, spleen_target = batch["bowel_target"], batch["extravasation_target"], batch["kidney_target"], batch["liver_target"], batch["spleen_target"]
        anyinjury_target = batch["any_injury_target_extra"]
        bowel, extravasation, kidney, liver, spleen = self(imgs)
        # print(bowel.shape, extravasation.shape, kidney.shape, liver.shape, spleen.shape)
        (b_loss, e_loss, k_loss, l_loss, s_loss), total_loss = \
            self.loss_module(bowel, extravasation, kidney, liver, spleen,
                             bowel_target, extravasation_target, kidney_target, liver_target, spleen_target)

        probs = F.sigmoid(bowel), F.sigmoid(extravasation), F.softmax(kidney), F.softmax(liver), F.softmax(spleen)
        any_injury = self._compute_any_injury_probability(probs)
        any_injury_metrics_out = self.train_metrics_anyinjury(any_injury, anyinjury_target)

        # true_positives, false_positives, true_negatives, false_negatives, sup = self.train_stat_scores(predictions, target)
        # train_loss = self.train_loss(out, target)
        to_log = {'train/bowel_loss': b_loss.item(),
                  'train/extravasation_loss': e_loss.item(),
                  'train/kidney_loss': k_loss.item(),
                  'train/liver_loss': l_loss.item(),
                  'train/spleen_loss': s_loss.item()
                 }
        self.log('train/total_loss', total_loss.item(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(to_log, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_dict(any_injury_metrics_out, on_epoch=True, batch_size=batch_size)
        return total_loss 
    
    def on_train_epoch_end(self):
        # log epoch metric
        # self.log_dict({**self.train_metrics_anyinjury}, on_epoch=True)
        pass
    
    def validation_step(self, batch, batch_idx):
        imgs = batch["image"]
        batch_size = imgs.shape[0]
        bowel_target, extravasation_target, kidney_target, liver_target, spleen_target = batch["bowel_target"], batch["extravasation_target"], batch["kidney_target"], batch["liver_target"], batch["spleen_target"]
        anyinjury_target = batch["any_injury_target_extra"]
        bowel, extravasation, kidney, liver, spleen = self(imgs)
        
        (b_loss, e_loss, k_loss, l_loss, s_loss), total_loss = \
            self.loss_module(bowel, extravasation, kidney, liver, spleen,
                             bowel_target, extravasation_target, kidney_target, liver_target, spleen_target)

        probs = F.sigmoid(bowel), F.sigmoid(extravasation), F.softmax(kidney), F.softmax(liver), F.softmax(spleen)
        
        any_injury = self._compute_any_injury_probability(probs)
        any_injury_metrics_out = self.valid_metrics_anyinjury(any_injury, anyinjury_target)

        to_log = {
            'valid/bowel_loss': b_loss.item(),
            'valid/extravasation_loss': e_loss.item(),
            'valid/kidney_loss': k_loss.item(),
            'valid/liver_loss': l_loss.item(),
            'valid/spleen_loss': s_loss.item(),
            'valid/total_loss': total_loss.item()
            }

        # probs = F.sigmoid(bowel), F.sigmoid(extravasation), F.softmax(kidney), F.softmax(liver), F.softmax(spleen)
        self.log_dict(to_log, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log_dict(any_injury_metrics_out, on_epoch=True, batch_size=batch_size)
        return total_loss