import argparse
import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

from torchvision.ops import generalized_box_iou_loss
from tableformer.models.models import TableFormer
from tableformer.datas.datas import DataModule
from tableformer.datas.utils import PREDEFINED_SET

class TableFormerLit(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.model = TableFormer()
        self.batch_size = args.batch_size
        self.l1_loss = nn.L1Loss()
        self.vocab_size = len(PREDEFINED_SET)

    def forward(self, x):
        """inference onlys"""
        _ = self.model(x)
        return None    

    def configure_optimizers(self):
        opt1 = optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_schedulers = {"scheduler": ReduceLROnPlateau(opt1, ...), "monitor": "metric_to_track"}
        return opt1, #optimizers, None

    def training_step(self, batch, batch_idx): 
        opt = self.optimizers()
        image, tags, boxes, classes = batch
        enc_out, pred_tags, pred_boxes, pred_clses = self.model(image, tags, boxes, classes)
        # structure decode loss : cross-entropy
        l_s = F.cross_entropy(pred_tags.permute(1,2,0), tags[:,1:]) # (b, vocab-1) : vocab : len([<start>,<end>,...])
        # bbox loss
        # import pdb;pdb.set_trace()
        l_iou = generalized_box_iou_loss(pred_boxes.permute(1,0,2), boxes, reduction='mean') # TODO match size of pred and gt
        l_l1 = self.l1_loss(pred_boxes.permute(1,0,2), boxes)
        l_box = 0.5 * l_iou + 0.5 * l_l1
        # total loss
        loss = 0.5*l_s + 0.5*l_box
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        pass

def get_args():
    parser = argparse.ArgumentParser(description='TableFormer')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--num_epochs', default=5)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--input_size', default=448)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--lr', default=0.0001)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dm = DataModule(args)
    model = TableFormerLit(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.num_epochs, 
        gpus=1
        ) 
        # default_root_dir=os.path.join(args.project_root_path, args.category), 
        #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    
    trainer.fit(model, dm)

    # if args.phase == 'train':
    #     trainer.fit(model)
    #     trainer.test(model)
    # elif args.phase == 'test':
    #     trainer.test(model)