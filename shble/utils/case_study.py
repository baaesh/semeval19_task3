import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_path')
args = parser.parse_args()

with io.open(args.prediction_path, encoding='utf-8') as fpred:
    preds = fpred.readlines()

with io.open('data/dev.txt', encoding='utf-8') as fdev:
    truths = fdev.readlines()


diffs = []
for i in range(len(preds)):
    pred = preds[i]
    truth = truths[i]

    pred_split = pred.strip().split('\t')
    pred_label = pred_split[4]

    truth_split = truth.strip().split('\t')
    truth_label = truth_split[4]

    if pred_label != truth_label:
        example = '\t|'.join(truth_split[:4])
        dict = {'example': example, 'ground truth': truth_label, 'prediction': pred_label}
        diffs.append(dict)

with open('diff_' + args.prediction_path + '.txt', 'w', encoding='utf-8') as f:
    f.write('id\t|turn1\t|turn2\t|turn3\t|ground truth\t|prediction\n')
    for i in range(len(diffs)):
        f.write(diffs[i]['example'] + '\t|' + diffs[i]['ground truth'] + '\t|'
                + diffs[i]['prediction'] + '\n')
