
This is an attempt to reproduce the [Don't Parse, Generate! A Sequence to Sequence Architecture for Task-Oriented Semantic Parsing
](https://arxiv.org/abs/2001.11458)


You need to download data from http://fb.me/semanticparsingdialog and unzip
the data, your file structure should like the following  

```
- top-dataset-semantic-parsing
    - test.tsv
    - train.tsv
    - eval.tsv
    ...
- TOP
    train.py
    - data
        ...
    ...
```

To run the code, you need to install the lastest [fastNLP](https://github.com/fastnlp/fastNLP) by the following command
```bash
pip install git+https://github.com/fastnlp/fastNLP
```

Then you can run the code as
```bash
python train.py
```

The implementation is different from the paper in two folds:   
(1) We don't generate(copy) the token from the source sentence, since in 
this task, the copy occurs sequentially. Therefore, in our code, 
during generation, the model only needs to judge whether to copy 
at this position. An index indicator is kept to record which 
position in the source sentence has been copied (CopyTransformerState.copy_src_len 
is for this purpose).  
(2) Instead of keeping an embedding for the word copied from 
the source sentence, we use the embedding from the source sentence to represent
the copied tokens in the decoder sequence (the CopyTransformerSeq2SeqDecoder.get_real_embed() 
is used to assign corresponding embedding for different tokens.).

The performance in the TOP datasets are  
|                  | EM    | Acc   |
|------------------|-------|-------|
| Rongali(Roberta) | 86.67 | 98.13 |
| Ours(Roberta fix)| 78.61  | 94.01 |
| Ours(Roberta finetune)| 83.54  | 94.35 |
