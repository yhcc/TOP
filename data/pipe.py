
from fastNLP.io import Loader, Pipe, DataBundle
from fastNLP import DataSet, Instance, Vocabulary
import os


class TopPipe(Pipe):
    def __init__(self):
        super().__init__()

    def process(self, data_bundle: DataBundle) -> DataBundle:
        data_bundle.apply_field(str.split, field_name='raw_words', new_field_name='src_tokens')
        data_bundle.apply_field(str.split, field_name='target', new_field_name='tgt_tokens')

        vocab = Vocabulary()
        vocab.from_dataset(data_bundle.get_dataset('train'), field_name='src_tokens',
                           no_create_entry_dataset=[data_bundle.get_dataset('test'), data_bundle.get_dataset('dev')])
        # target_vocab
        target_vocab = Vocabulary(unknown=None)
        target_vocab.add_word_lst(['[COPY]', '<SOS>', '<EOS>'])
        target_vocab.build_vocab()

        def build_vocab(ins):
            src_tokens = set(ins['src_tokens'])
            tgt_tokens = ins['tgt_tokens']
            for token in tgt_tokens:
                if token not in src_tokens:
                    target_vocab.add_word(token)
        data_bundle.apply(build_vocab, new_field_name=None)

        target_vocab.build_vocab()

        def index_target(ins):
            tgt_tokens = ins['tgt_tokens']
            src_tokens = ins['src_tokens']
            idx = 0
            target = [target_vocab.to_index('<SOS>')]
            for t in tgt_tokens:
                if idx>=len(src_tokens) or t != src_tokens[idx]:
                    target.append(target_vocab.to_index(t))
                else:
                    idx += 1
                    target.append(target_vocab.to_index('[COPY]'))
            assert idx==len(src_tokens)
            return target + [target_vocab.to_index('<EOS>')]
        data_bundle.apply(index_target, new_field_name='tgt_tokens')
        vocab.index_dataset(*data_bundle.datasets.values(), field_name='src_tokens')

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')

        data_bundle.set_vocab(target_vocab, field_name='tgt_tokens')
        data_bundle.set_vocab(vocab, field_name='src_tokens')

        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len')

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        data_bundle = TopLoader().load(paths)
        return self.process(data_bundle)


class TopLoader(Loader):
    def __init__(self):
        super().__init__()

    def load(self, paths) -> DataBundle:
        """

        raw_words            target
        'xxx xxx '      '[ xxx ]...'

        :param paths:
        :return:
        """
        data_bundle = DataBundle()
        for name in ['train', 'test', 'eval']:
            ds = DataSet()
            with open(os.path.join(paths, name + '.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    # 基于的是tokenized后的文字
                    parts = line.strip().split('\t')
                    text = parts[1]
                    top = parts[2]
                    ins = Instance(raw_words=text, target=top)
                    ds.append(ins)
            if name == 'eval':
                name = 'dev'
            data_bundle.set_dataset(name=name, dataset=ds)

        return data_bundle


if __name__ == '__main__':
    data_bundle = TopPipe().process_from_file('../../data/top-dataset-semantic-parsing')
    print(data_bundle)
    print(data_bundle.get_vocab('tgt_tokens').word2idx)
    exit()
    from collections import Counter
    length_counter = Counter()
    for name,ds in data_bundle.iter_datasets():
        length_counter.update(ds.get_field('tgt_seq_len').content)
    print(length_counter)







