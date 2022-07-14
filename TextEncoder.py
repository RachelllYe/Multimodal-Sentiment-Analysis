import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return out

