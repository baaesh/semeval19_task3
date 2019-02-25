import emoji

trainFilePath = 'data/raw/train.txt'
validFilepath = 'data/raw/dev.txt'

with open(trainFilePath, 'r', encoding='utf-8') as f:
    train = f.readlines()
with open(validFilepath, 'r', encoding='utf-8') as f:
    valid = f.readlines()

with open('data/train_emoji.txt', 'w', encoding='utf-8') as f:
    for line in train:
        line_list = list(line)
        for i in range(len(line_list)):
            if line_list[i] in emoji.UNICODE_EMOJI and line_list[i] == line_list[i+1]:
                line_list[i] = ''
        line_e = ''.join(line_list)
        f.write(line_e)

with open('data/dev_emoji.txt', 'w', encoding='utf-8') as f:
    for line in valid:
        line_list = list(line)
        for i in range(len(line_list)):
            if line_list[i] in emoji.UNICODE_EMOJI and line_list[i] == line_list[i+1]:
                line_list[i] = ''
        line_e = ''.join(line_list)
        f.write(line_e)
