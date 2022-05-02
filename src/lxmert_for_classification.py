
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from transformers.modeling_outputs import SequenceClassifierOutput

class LxmertForBinaryClassification(nn.Module):
    def __init__(self, lxmert):
        super(LxmertForBinaryClassification, self).__init__()
        self.lxmert = lxmert

    def forward(self, input_ids, attention_mask, visual_feats, visual_pos, token_type_ids, labels=None):

        outputs = self.lxmert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                visual_feats = visual_feats,
                visual_pos = visual_pos,
                token_type_ids = token_type_ids)
        reshaped_logits = outputs["cross_relationship_score"]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))
        else:
            loss = None

        return SequenceClassifierOutput(
                loss=loss,
                logits=reshaped_logits,
        )


