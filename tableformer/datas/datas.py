import json
from typing import List
from torch.utils.data import Dataset

PREDEFINED_SET = {
    'tags': ['<thead>', '<tr>', '<td>', '</td>', '<td', ' colspan="6"', '>', '</tr>', '</thead>', '<tbody>', '</tbody>', ' rowspan="2"', ' rowspan="3"', ' colspan="4"', ' colspan="3"', ' colspan="2"', ' colspan="5"', ' colspan="9"', ' colspan="7"', ' colspan="10"', ' colspan="8"']
}

def parse_tags(tags:List) -> List:
    idxs = []
    for tag in tags:
        assert tag in PREDEFINED_SET['tags'], f"{tag} not in predefined sets"
        idxs.append(PREDEFINED_SET['tags'].index(tag))
    return idxs

class SynthTabNetDataset(Dataset):
    def __init__(self, img_dir:str, anno_file:str):
        self.img_dir = img_dir
        
        with open(anno_file, 'r') as json_file:
            json_list = list(json_file)
        
        for j in json_list:
            anno = json.loads(j)
    
    def __len__(self):
        return len()
    
    def __getitem__(self, idx):

        return img, label