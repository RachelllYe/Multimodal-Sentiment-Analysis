import torch
import torch.nn as nn

from TextEncoder import BertEncoder
from ImageEncoder import ImageEncoder


class MultimodalConcatBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBertClf, self).__init__()
        self.args = args
        if args.model_type == "multimodal":
            self.txtenc = BertEncoder(args)
            self.imgenc = ImageEncoder(args)
            last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        elif args.model_type == 'unimodal_img':
            self.imgenc = ImageEncoder(args)
            last_size = args.img_hidden_sz * args.num_image_embeds
        elif args.model_type == 'unimodal_text':
            self.txtenc = BertEncoder(args)
            last_size = args.hidden_sz
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, 100))
        self.clf.append(nn.ReLU())
        self.clf.append(nn.Linear(100, args.n_classes))

    def forward(self, txt, mask, segment, img):
        if self.args.model_type == "multimodal":
            txt = self.txtenc(txt, mask, segment)
            img = self.imgenc(img)
            img = torch.flatten(img, start_dim=1)
            out = torch.cat([txt, img], -1)
        elif self.args.model_type == 'unimodal_img':
            img = self.imgenc(img)
            img = torch.flatten(img, start_dim=1)
            out = img
        elif self.args.model_type == 'unimodal_text':
            txt = self.txtenc(txt, mask, segment)
            out = txt
        for layer in self.clf:
            out = layer(out)
        return out

