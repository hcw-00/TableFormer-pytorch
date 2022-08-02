import torch
from torch import nn
import torchvision

class FeatureExtractor(nn.Module):
    """
    FeatureExtractor.
    """

    def __init__(self, encoded_image_size=14):
        super(FeatureExtractor, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet18(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

class MLP(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        op_list = []
        for _ in range(num_layers):
            op_list.append(nn.Linear(4))
            op_list.append(nn.ReLU())
        self.mlp = nn.Sequential(*op_list)
    def forward(self):
        return self.mlp

class TableFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.embedding = Embedding()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=0.5)
        # attn_output, attn_output_weights = self.attention(query, key, value)

        self.linear = nn.Linear(in_features=1, out_features=1)
        self.softmax = nn.Softmax(dim=4)

        self.mlp = MLP(num_layers=3)
        self.cls = nn.Linear(in_features=1, out_features=2)

        self.src_pe = PositionalEncoder()
        self.target_pe = PositionalEncoder()

    def forward(self, img, y_tags, y_bboxes):
        x_f = self.feature_extractor(img)
        enc_out = self.encoder(self.src_pe(x_f))

        ys = torch.zeros(ys_size, dtype=torch.long, device=img.device)
        ys[0, :] = 1

        if y_tags in None: # predict greedy
            for i in range(self.max_len + 1):
                out = self.decoder(self.target_pe(self.target_embed(ys[:i+1])), x_e, mask)

                probs[:, i, :] = self.generator(out[-1])
                _, ys[i+1, :] = torch.max(probs[:, i, :], dim=-1)
                return
        else:
            dec_out = self.decoder(tgt=y_tags, memory=enc_out, tgt_mask=None, memory_mask=None)
            pred_tags = self.softmax(self.linear(dec_out))
            attn_out, attn_outpuot_weights = self.attention(out, out, enc_out)
            pred_boxes, pred_clses = self.mlp(attn_out), self.cls(attn_out)
            return enc_out, pred_tags, pred_boxes, pred_clses

def _greedy_decode(self, enc_input):
    batch_size = enc_input.size(1)
    pred_size = (batch_size, self.batch_max_length+1, self.num_classes)
    ys_size = (self.batch_max_length+2, batch_size)
    
    memory = self.model.encoder(self.src_pe(enc_input))
    ys = torch.zeros(ys_size, dtype=torch.long, device=enc_input.device)
    # TODO: Fix Dynamic BOS Token Index
    ys[0, :] = 1
    probs = torch.zeros(
        pred_size, dtype=torch.float, device=enc_input.device)
    target_mask = self.target_mask.to(enc_input.device)

    for i in range(self.batch_max_length+1):
        out = self.model.decoder(
            self.target_pe(self.taraget_embed(ys[:i+1])),
            memory,
            target_mask[:i+1, :i+1]
        )

        probs[:, i, :] = self.generator(out[-1])
        _, ys[i+1, :] = torch.max(probs[:, i, :], dim=-1)
    
    return probs

def _generate_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(
        mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask