import io
import glob
import numpy as np

file_path_list = glob.glob('./ensemble/*.txt')

file_lines = []
for file_path in file_path_list:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        file_lines.append(lines)

solutionPath = 'ensemble.txt'
with io.open(solutionPath, "w", encoding="utf8") as fout:
    fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')


    for i in range(len(file_lines[0])):
        if i == 0:
            continue

        pred_line = file_lines[0][i]
        others, happy, angry, sad = 0, 0, 0, 0
        for lines in file_lines:
            line = lines[i].strip().split('\t')
            label = line[4]
            if label == 'others':
                others += 1
            elif label == 'happy':
                happy += 1
            elif label == 'angry':
                angry += 1
            elif label == 'sad':
                sad += 1
            else:
                print('error line' + str(i))

            counts = [others, happy, angry, sad]
            #print(counts)
            idx = np.argmax(counts)
            pred_label = 'others'
            if idx == 1:
                pred_label = 'happy'
            elif idx == 2:
                pred_label = 'angry'
            elif idx == 3:
                pred_label = 'sad'

        fout.write('\t'.join(pred_line.strip().split('\t')[:4]) + '\t')
        fout.write(pred_label + '\n')