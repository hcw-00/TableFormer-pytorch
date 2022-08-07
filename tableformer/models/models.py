import math
import torch
from torch import nn, Tensor
import torchvision
from ..datas.utils import PREDEFINED_SET

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim] # [6, 4, ]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FeatureExtractor(nn.Module):
    def __init__(self, encoded_image_size=28, device='cuda'):
        super(FeatureExtractor, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet18(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules).to(device)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        """
        args:
            images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        return: encoded images
        """
        images = images.float() # TODO remove this after add normalize in transforms
        out = self.resnet(images)  
        out = self.adaptive_pool(out) 
        out = out.permute(0, 2, 3, 1) 
        return out

class MLP(nn.Module):
    def __init__(self, in_features=512, out_features=4, hidden_dim=512):
        super().__init__()
        op_list = []
        op_list.append(nn.Linear(in_features, hidden_dim))
        op_list.append(nn.ReLU())
        op_list.append(nn.Linear(hidden_dim, hidden_dim))
        op_list.append(nn.ReLU())
        op_list.append(nn.Linear(hidden_dim, out_features))
        op_list.append(nn.ReLU())
        self.mlp = nn.Sequential(*op_list)
    def forward(self, x):
        return self.mlp(x)

class TableFormer(nn.Module):
    def __init__(
        self, 
        d_model=512,
        enc_nhead=4,
        enc_dim_feedforward=1024,
        enc_num_layers=2,
        dec_nhead=4,
        dec_dim_feedforward=1024,
        dec_num_layers=4,
        max_len=512,
        device='cuda'
        ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(device=device)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=enc_nhead, dim_feedforward=enc_dim_feedforward, dropout=0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=dec_nhead, dim_feedforward=dec_dim_feedforward, dropout=0.5)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_num_layers)
        self.linear = nn.Linear(in_features=d_model, out_features=len(PREDEFINED_SET))
        self.softmax = nn.Softmax(dim=2) # TODO (l, b, len(vocab))

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, dropout=0.5)

        self.mlp = MLP(in_features=d_model, out_features=4, hidden_dim=512)
        self.cls = nn.Linear(in_features=d_model, out_features=2)

        self.src_pe = PositionalEncoding(d_model)
        self.target_pe = PositionalEncoding(d_model)

        self.src_embed = nn.Embedding(num_embeddings=len(PREDEFINED_SET), embedding_dim=d_model) 
        self.tgt_embed = nn.Embedding(num_embeddings=len(PREDEFINED_SET), embedding_dim=d_model) 

        self.tgt_mask = generate_square_subsequent_mask(max_len+1)

        self.pad_idx = PREDEFINED_SET.index("<pad>")
        self.td_a_idx = PREDEFINED_SET.index("<td>")
        self.td_b_idx = PREDEFINED_SET.index("<td")

    def forward(self, img, y_tags, y_bboxes, y_classes):
        x_f = self.feature_extractor(img) # x_f : (b, h, w, c)
        x_f = x_f.permute(1,2,0,3) # (b, h, w, c) => (h, w, b, c)
        x_f = x_f.view(x_f.shape[0]*x_f.shape[1], -1, x_f.shape[3]) # (l, b, c) (784, 1, 512)
        enc_out = self.encoder(self.src_pe(x_f)) 

        if y_tags is None:
            # TODO: greedy | beam
            pass
        else:
            y_src = y_tags[:, :-1].permute(1,0)
            padding_mask = self.get_padding_mask(y_src)
            dec_out = self.decoder(
                tgt=self.target_pe(self.tgt_embed(y_src)),
                memory=enc_out, 
                tgt_mask=self.tgt_mask.to(enc_out.device),
                tgt_key_padding_mask=padding_mask.permute(1,0), 
                memory_mask=None)
            dec_out_ = self.get_tds(y_src, dec_out)
            pred_tags = self.softmax(self.linear(dec_out))
            attn_out, _ = self.attention(dec_out_, enc_out, enc_out) # q, k, v
            pred_boxes, pred_clses = self.mlp(attn_out), self.cls(attn_out)
            return enc_out, pred_tags, pred_boxes, pred_clses

    def get_tds(self, src, dec_out):
        mask_a = src == self.td_a_idx
        mask_b = src == self.td_b_idx
        mask = mask_a + mask_b
        for i in range(dec_out.shape[1]):
            if i == 0:
                dec_out_ = dec_out[:-1,0:1,:][mask[1:,0]]
            else:
                dec_out_ = torch.cat([dec_out_, dec_out[:-1,i:i+1,:][mask[1:,i]]], dim=1)
        return dec_out_


    def get_padding_mask(self, src):
        padding_mask = torch.zeros(src.shape, dtype=torch.bool).to(src.device)
        padding_mask[src==self.pad_idx] = True
        return padding_mask

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)