import sys
sys.path.append('../')

from data.pipe import TopPipe
from fastNLP import Trainer
from model.model import CopyModel
from model.metrics import ExactMatch
from model.losses import Seq2SeqLoss
from fastNLP.embeddings import RobertaEmbedding
from fastNLP.models import SequenceGeneratorModel
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback, FitlogCallback
from fastNLP import SortedSampler
import fitlog
fitlog.debug()
fitlog.set_log_dir('logs')
fitlog.add_hyper_in_file(__file__)

#######hyper
lr = 1e-3
n_epochs = 100
batch_size = 32
n_heads = 6
d_model = 300
num_layers = 5
bind_decoder_input_output_embed = False
#######hyper


@cache_results('caches/data.pkl')
def get_data():
    data_bundle = TopPipe().process_from_file('../data/top-dataset-semantic-parsing')
    return data_bundle

data_bundle = get_data()
print(data_bundle)
tgt_vocab = data_bundle.get_vocab('tgt_tokens')

src_embed = RobertaEmbedding(data_bundle.get_vocab('src_tokens'), dropout=0.5, requires_grad=False)

model = CopyModel.build_model(src_embed=src_embed, tgt_embed=(len(data_bundle.get_vocab('tgt_tokens')), src_embed.embedding_dim),
                    pos_embed='sin', max_position=300, num_layers=num_layers, d_model=d_model, n_head=n_heads, dim_ff=1024,
                    dropout=0.1,
                    bind_decoder_input_output_embed=bind_decoder_input_output_embed)
model = SequenceGeneratorModel(model, bos_token_id=tgt_vocab.to_index('<SOS>'),
                               eos_token_id=tgt_vocab.to_index('<EOS>'), max_length=60, num_beams=4,
                               do_sample=False, temperature=1.0, top_k=50, top_p=1.0,
                               repetition_penalty=1, length_penalty=1.0, pad_token_id=0)
# import torch
# if torch.cuda.is_available():
#     model.cuda()

parameters = []
params = {'lr':lr}
params['params'] = [param for param in model.parameters() if param.requires_grad]
parameters.append(params)
params = {'lr':lr*0.01}
src_embed.requires_grad = True
params['params'] = [param for param in src_embed.parameters() if param.requires_grad]
parameters.append(params)

optimizer = optim.AdamW(parameters)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=1, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))

# sampler = None
sampler = BucketSampler(seq_len_field_name='src_seq_len')
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                    loss=Seq2SeqLoss(),
                    batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                    num_workers=2, n_epochs=n_epochs, print_every=1,
                    dev_data=data_bundle.get_dataset('dev'),
                    metrics=ExactMatch(tgt_vocab.to_index('<EOS>')), metric_key=None,
                    validate_every=-1, save_path=None, use_tqdm=True, device=0,
                    callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                    test_sampler=SortedSampler('src_seq_len'))

trainer.train(load_best_model=False)
