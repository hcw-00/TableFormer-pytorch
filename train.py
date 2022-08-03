from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import glob
import os

from PIL import Image
from tableformer.models.models import TableFormer

class TableFormerLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """inference onlys"""
        _ = self.model(x)
        return None    

    def configure_optimizers(self):
        # 3 x Adam optimizer each for,
        # CNN backbone network
        # structure decoder
        # cell bbox decoder

        # parameter setup (=> PubTabNet)
        # 1st stage. lr 0.001 for 12 epoch with batch size 24, and lambda 0.5
        # 2nd stage. lr 0.0001 for 12 epoch with batch size 18.
        opt1 = Adam(...)
        opt2 = Adam(...)
        opt3 = Adam(...)
        optimizers = [opt1, opt2, opt3]
        lr_schedulers = {"scheduler": ReduceLROnPlateau(opt1, ...), "monitor": "metric_to_track"}
        return optimizers, lr_schedulers
        
    def training_step(self, batch, batch_idx): 
        x, y, z = batch
        enc_out, pred_tags, pred_boxes, pred_clses = self.model(x, y)
        loss = 'loss'
        metrics = {}
        metrics["train_loss"] = loss.tolist()
        if self.global_step % self.trainer.log_every_n_steps == 0 and self.logger:
            self.logger.log_metrics(metrics)
        pass

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        enc_out, pred_tags, pred_boxes, pred_clses = self.model(x, y)
        loss = 'loss'
        try:
            eval_result = self.evaluator.evaluate(y, preds, verbose= batch_idx == 0)
            return { **eval_result , "loss": loss}
        except Exception as e:
            print(e)
            return { "correct": 0, "cer": 0, "length": 0 , "loss": loss}

def get_args():
    parser = argparse.ArgumentParser(description='TableFormer')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--input_size', default=224)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))
    trainer = pl.Trainer.from_argparse_args(args, \
        default_root_dir=os.path.join(args.project_root_path, args.category), \
        max_epochs=args.num_epochs, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    
    model = TableFormer(hparams=args)
    
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)