import json
from typing import List
from torch.utils.data import Dataset

PREDEFINED_SET = {
    'tags': [' colspan="10"', ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"', ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"', ' rowspan="2"', ' rowspan="3"', '</tbody>', '</td>', '</thead>', '</tr>', '<tbody>', '<td', '<td>', '<thead>', '<tr>', '>']
}

class Parser:
    def __init__(self, char_set=PREDEFINED_SET):
        tokens = ['[BOS]', '[EOS]']
        char_set = char_set + tokens
        self.char_dict = dict()
        self.char_dict.update({k: v for k, v in enumerate(char_set)})

    
    

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