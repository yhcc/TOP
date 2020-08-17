
from fastNLP import MetricBase
from fastNLP import seq_len_to_mask


class ExactMatch(MetricBase):
    def __init__(self, eos_index):
        super().__init__()
        self.total = 0
        self.em = 0
        self.intent = 0
        self.eos_index = eos_index

    def evaluate(self, tgt_tokens, pred, tgt_seq_len):
        """

        :param tgt_tokens: bsz x max_len, [sos] + [tokens] + [eos]
        :param pred: bsz x max_len' x vocab_size
        :param tgt_seq_len: bsz
        :return:
        """

        if pred.dim()==3:
            pred = pred.argmax(dim=-1)

        self.total += pred.size(0)
        self.intent += pred[:, 1].eq(tgt_tokens[:, 1]).sum().item()

        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_index).cumsum(dim=1).long()
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz

        if pred.size(1)>tgt_tokens.size(1):
            tmp = tgt_tokens.new_zeros(tgt_tokens.size(0), pred.size(1))
            tmp[:, :tgt_tokens.size(1)] = tgt_tokens
            tgt_tokens = tmp
        elif pred.size(1)<tgt_tokens.size(1):
            tmp = tgt_tokens.new_zeros(tgt_tokens.size(0), tgt_tokens.size(1))
            tmp[:, :pred.size(1)] = pred
            pred = tmp

        length_eq = pred_seq_len.eq(tgt_seq_len)[:, None]  # bsz x 1 相等的地方才有必要继续对比

        pred = pred.masked_select(length_eq).view(-1, pred.size(1))
        tgt_tokens = tgt_tokens.masked_select(length_eq).view(-1, pred.size(1))
        length = tgt_seq_len.masked_select(length_eq.squeeze(1))
        mask = seq_len_to_mask(length, max_len=pred.size(1)).eq(0)

        self.em += pred.eq(tgt_tokens).masked_fill(mask, 0).sum(dim=1).eq(length).sum().item()

    def get_metric(self, reset=True):
        res = {'em': round(self.em/self.total*100, 2), 'i_acc': round(self.intent/self.total*100, 2)}
        if reset:
            self.total = 0
            self.em = 0
            self.intent = 0
        return res
