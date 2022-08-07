from typing import List, Union
import torch

PREDEFINED_SET = [" colspan=\"10\"", " colspan=\"2\"", " colspan=\"3\"", " colspan=\"4\"", " colspan=\"5\"",
" colspan=\"6\"", " colspan=\"7\"", " colspan=\"8\"", " colspan=\"9\"", " rowspan=\"10\"",
" rowspan=\"2\"", " rowspan=\"3\"", " rowspan=\"4\"", " rowspan=\"5\"", " rowspan=\"6\"",
" rowspan=\"7\"", " rowspan=\"8\"", " rowspan=\"9\"", "</tbody>", "</td>", "</thead>",
"</tr>", "<end>", "<pad>", "<start>", "<tbody>", "<td", "<td>", "<thead>", "<tr>", "<unk>", ">"]


class TagConverter:
    def __init__(self, char_set=PREDEFINED_SET, max_len=512):
        # tokens = ['[sos]', '[eos]']
        char_set = char_set # + tokens
        self.char_dict = dict()
        self.char_dict.update({v: idx for idx, v in enumerate(char_set)})
        self.max_len = max_len

    def encode(self, src:List):
        assert isinstance(src, List)
        # add <sos>, <eos>, <pad>
        src = src + ["<end>"]
        while len(src) < self.max_len+1:
            src.append("<pad>")
        src = ["<start>"] + src
        return self._encode(src)

    def _encode(self, input: Union[str, List]):
        if isinstance(input, List):
            encoded = []
            for i in input:
                encoded.append(self.char_dict[i])
            return torch.tensor(encoded, dtype=torch.long)
        return torch.tensor(self.char_dict[input], dtype=torch.long)

    def decode(self):
        pass
    

def parse_tags(tags:List) -> List:
    idxs = []
    for tag in tags:
        assert tag in PREDEFINED_SET, f"{tag} not in predefined sets"
        idxs.append(PREDEFINED_SET.index(tag))
    return idxs