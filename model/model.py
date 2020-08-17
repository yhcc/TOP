
from torch import nn
import torch
from fastNLP.models import Seq2SeqModel
from fastNLP.modules import TransformerSeq2SeqDecoder
from fastNLP.modules.decoder.seq2seq_decoder import TransformerState
from fastNLP.modules import TransformerSeq2SeqEncoder
from fastNLP.embeddings import get_embeddings, get_sinusoid_encoding_table
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class CopyTransformerState(TransformerState):
    def __init__(self, encoder_output, encoder_mask, num_decoder_layer, src_token_embeds, src_seq_len):
        """
        与TransformerSeq2SeqDecoder对应的State，

        :param torch.FloatTensor encoder_output: bsz x encode_max_len x encoder_output_size, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x encode_max_len 为1的地方需要attend
        :param int num_decoder_layer: decode有多少层
        :param src_token_embeds: bsz x encode_max_len x embed_size
        :param src_seq_len: bsz
        """
        super().__init__(encoder_output, encoder_mask, num_decoder_layer)
        self.copy_src_len = encoder_output.new_zeros(encoder_output.size(0)).long()
        self.src_token_embeds = src_token_embeds
        self.src_seq_len = src_seq_len

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.copy_src_len = self.copy_src_len.index_select(index=indices, dim=0)
        self.src_token_embeds = self.src_token_embeds.index_select(index=indices, dim=0)
        self.src_seq_len = self.src_seq_len.index_select(index=indices, dim=0)


class CopyTransformerSeq2SeqDecoder(TransformerSeq2SeqDecoder):
    def forward(self, tokens, state, return_attention=False):
        """

        :param torch.LongTensor tokens: batch x tgt_len，decode的词
        :param CopyTransformerState state: 用于记录encoder的输出以及decode状态的对象，可以通过init_state()获取
        :param bool return_attention: 是否返回对encoder结果的attention score
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """

        encoder_output = state.encoder_output
        encoder_mask = state.encoder_mask
        all_tokens = tokens

        assert state.decode_length<tokens.size(1), "The decoded tokens in State should be less than tokens."
        tokens = tokens[:, state.decode_length:]
        device = tokens.device

        x = []
        if self.training:
            tgt_seq_len = tokens.size(1)-1
        else:
            tgt_seq_len = tokens.size(1)
        for i in range(tgt_seq_len):
            # 这里需要设计一下
            embed = self.get_real_embed(tokens[:, i], state.copy_src_len, state.src_seq_len, state.src_token_embeds)
            x.append(embed)

        x = torch.stack(x, dim=1)
        x = self.embed_scale * x
        if self.pos_embed is not None:
            position = torch.arange(state.decode_length+1, state.decode_length+1+tgt_seq_len).long().to(device)[None]
            x += self.pos_embed(position)
        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        batch_size, max_tgt_len = tokens.size()

        if max_tgt_len>1:
            triangle_mask = self._get_triangle_mask(tokens[:, :tgt_seq_len])
        else:
            triangle_mask = None

        for layer in self.layer_stacks:
            x, attn_weight = layer(x=x,
                                   encoder_output=encoder_output,
                                   encoder_mask=encoder_mask,
                                   self_attn_mask=triangle_mask,
                                   state=state)
        x = self.layer_norm(x)  # batch, tgt_len, dim
        x = self.output_fc(x)
        feats = self.output_layer(x)

        if not self.training:
            mask = state.copy_src_len.ge(state.src_seq_len)
            # TODO 这一条有点作弊，因为强制format了合法的bracket
            no_need_bracket = all_tokens.gt(4).sum(dim=1).le(all_tokens.eq(4).sum(dim=1))
            feats[:, -1, 4].data.masked_fill_(no_need_bracket, -1000)
            feats[:, -1, 1].data.masked_fill_(mask, -1000)  # 不能copy了

        if return_attention:
            return feats, attn_weight
        return feats

    def get_real_embed(self, cur_token, copy_src_len, src_seq_len, src_token_embeds):
        """

        :param cur_token: bsz, 当前的token
        :param copy_src_len: bsz, copy到哪个位置了，如果发生了copy会in-place增长1的
        :param src_seq_len: bsz, source的长度
        :param src_token_embeds: bsz x max_len x embed_size
        :return: bsz x embed_size
        """
        copy_token_flag = cur_token.eq(1)  # 为1地方需要copy前面的, bsz

        copied_embed = src_token_embeds.gather(index=copy_src_len.view(-1, 1, 1).repeat(1, 1, src_token_embeds.size(-1)), dim=1).squeeze(1)  # bsz x embed_size

        top_embed = self.embed(cur_token[:, None]).squeeze(1)  # bsz x embed_size

        cur_real_tokens = torch.where(copy_token_flag.view(-1, 1), copied_embed, top_embed)  # bsz x 1 x embed_size

        copy_src_len.add_(copy_token_flag.masked_fill(copy_src_len.ge(src_seq_len-1), 0).long())  # 超过能够copy的长度就结束了

        return cur_real_tokens

    def init_state(self, encoder_output, encoder_mask, src_token_embeds, src_seq_len):
        """
        初始化一个TransformerState用于forward

        :param torch.FloatTensor encoder_output: bsz x max_len x d_model, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为1的位置需要attend。
        :param src_token_embeds: bsz x max_len x embed_size
        :param src_seq_len: bsz
        :return: TransformerState
        """
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = CopyTransformerState(encoder_output, encoder_mask, num_decoder_layer=self.num_layers,
                                     src_token_embeds=src_token_embeds, src_seq_len=src_seq_len)

        return state

    @staticmethod
    def _get_triangle_mask(tokens):
        tensor = tokens.new_ones(tokens.size(1), tokens.size(1))
        return torch.tril(tensor).byte()


class CopyTransformerSeq2SeqEncoder(TransformerSeq2SeqEncoder):
    def forward(self, tokens, seq_len):
        """

        :param tokens: batch x max_len
        :param seq_len: [batch]
        :return: bsz x max_len x d_model, (bsz x max_len(为0的地方为padding), bsz x max_len x embed_size)
        """
        embed = self.embed(tokens)  # batch, seq, dim
        x = embed * self.embed_scale
        batch_size, max_src_len, _ = x.size()
        device = x.device
        if self.pos_embed is not None:
            position = torch.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_mask = seq_len_to_mask(seq_len)
        encoder_mask = encoder_mask.to(device)

        for layer in self.layer_stacks:
            x = layer(x, encoder_mask)

        x = self.layer_norm(x)

        return x, (encoder_mask, embed, seq_len)


class CopyModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, src_embed, tgt_embed=None,
                    pos_embed='sin', max_position=1024, num_layers=6, d_model=512, n_head=8, dim_ff=2048,
                    dropout=0.1,
                    bind_decoder_input_output_embed=True):
        """
        初始化一个TransformerSeq2SeqModel

        :param nn.Module, StaticEmbedding, Tuple[int, int] src_embed: source的embedding
        :param nn.Module, StaticEmbedding, Tuple[int, int] tgt_embed: target的embedding，如果bind_encoder_decoder_embed为
            True，则不要输入该值
        :param str pos_embed: 支持sin, learned两种
        :param int max_position: 最大支持长度
        :param int num_layers: encoder和decoder的层数
        :param int d_model: encoder和decoder输入输出的大小
        :param int n_head: encoder和decoder的head的数量
        :param int dim_ff: encoder和decoder中FFN中间映射的维度
        :param float dropout: Attention和FFN dropout的大小
        :param bool bind_decoder_input_output_embed: decoder的输出embedding是否与其输入embedding是一样的权重
        :return: TransformerSeq2SeqModel
        """
        src_embed = get_embeddings(src_embed)

        assert tgt_embed is not None, "You need to pass `tgt_embed` when `bind_encoder_decoder_embed=False`"
        tgt_embed = get_embeddings(tgt_embed)

        if pos_embed == 'sin':
            encoder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, src_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
            deocder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, tgt_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
        elif pos_embed == 'learned':
            encoder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=0)
            deocder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=1)
        else:
            raise ValueError("pos_embed only supports sin or learned.")

        encoder = CopyTransformerSeq2SeqEncoder(embed=src_embed, pos_embed=encoder_pos_embed,
                                            num_layers=num_layers, d_model=d_model, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout)
        decoder = CopyTransformerSeq2SeqDecoder(embed=tgt_embed, pos_embed=deocder_pos_embed,
                                            d_model=d_model, num_layers=num_layers, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout,
                                            bind_decoder_input_output_embed=bind_decoder_input_output_embed)

        return cls(encoder, decoder)

    def prepare_state(self, src_tokens, src_seq_len=None):
        """
        调用encoder获取state，会把encoder的encoder_output, encoder_mask直接传入到decoder.init_state中初始化一个state

        :param src_tokens:
        :param src_seq_len:
        :return:
        """
        encoder_output, (encoder_mask, src_token_embeds, seq_len) = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask, src_token_embeds, src_seq_len=seq_len)
        return state





