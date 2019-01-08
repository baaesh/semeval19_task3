import pickle
from torchtext import data
import emoji


TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
filehandler = open('vocab.obj', 'rb')
TEXT.vocab = pickle.load(filehandler)

with open('vocab.txt', 'w', encoding='utf-8') as f:
    for i in range(len(TEXT.vocab)):
        str = ''.join(c for c in TEXT.vocab.itos[i] if c in emoji.UNICODE_EMOJI)
        if str != '':
            f.write(str + '\n')

