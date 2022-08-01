from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import glob
import os

from PIL import Image
from tableformer.models.models import FeatureExtractor, Encoder, Decoder

class CustomDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

class TableFormerModule(pl.LightningModule):
    def __init__(self, hparams):
        super(TableFormerModule, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cell_bbox_decoder = CellBboxDecoder()

        self.criterion = torch.nn.MSELoss(reduction='sum')

        #self.data_transforms = transforms.Compose(...)

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def train_dataloader(self):
        image_datasets = CustomDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = CustomDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True)
        return test_loader

    def configure_optimizers(self):
        # 3 x Adam optimizer each for,
        # CNN backbone network
        # structure decoder
        # cell bbox decoder

        # parameter setup (=> PubTabNet)
        # 1st stage. lr 0.001 for 12 epoch with batch size 24, and lambda 0.5
        # 2nd stage. lr 0.0001 for 12 epoch with batch size 18.
        return None

    def on_train_start(self):
        
        pass
    
    def on_test_start(self):
        pass
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        pass

    def training_epoch_end(self, outputs): 
        pass

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        pass

    def test_epoch_end(self, outputs):
        pass

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'./MVTec')
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path', default=r'./test')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    trainer = pl.Trainer.from_argparse_args(args, \
        default_root_dir=os.path.join(args.project_root_path, args.category), \
        max_epochs=args.num_epochs, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = TableFormerModule(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)