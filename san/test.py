import numpy as np
import torch
from keras.utils import to_categorical
from statistics import getStatistics

def test(model, data, iterator, criterion, args):
    model.eval()
    acc, loss, size = 0, 0, 0

    if args.thresholding:
        others_idx = data.LABEL.vocab.stoi['others']
        happy_idx = data.LABEL.vocab.stoi['happy']
        sad_idx = data.LABEL.vocab.stoi['sad']
        angry_idx = data.LABEL.vocab.stoi['angry']
        train_dist = getStatistics(args.train_data_path, 'train')
        valid_dist = getStatistics(args.valid_data_path, 'valid')
        reverse_prior = [1.0] * args.class_size
        reverse_prior[others_idx] = valid_dist[0] / train_dist[0]
        reverse_prior[happy_idx] = valid_dist[1] / train_dist[1]
        reverse_prior[sad_idx] = valid_dist[2] / train_dist[2]
        reverse_prior[angry_idx] = train_dist[3] / train_dist[3]
        reverse_prior = torch.tensor(reverse_prior)

    preds = []
    labels = []
    for batch in iter(iterator):
        if args.char_emb:
            if args.fusion:
                char_c = torch.LongTensor(data.characterize(batch.context[0])).to(args.device)
                char_s = torch.LongTensor(data.characterize(batch.sent[0])).to(args.device)
                setattr(batch, 'char_c', char_c)
                setattr(batch, 'char_s', char_s)
            elif args.seperate:
                char_turn1 = torch.LongTensor(data.characterize(batch.turn1[0])).to(args.device)
                char_turn2 = torch.LongTensor(data.characterize(batch.turn2[0])).to(args.device)
                char_turn3 = torch.LongTensor(data.characterize(batch.turn3[0])).to(args.device)
                setattr(batch, 'char_turn1', char_turn1)
                setattr(batch, 'char_turn2', char_turn2)
                setattr(batch, 'char_turn3', char_turn3)
            else:
                char = torch.LongTensor(data.characterize(batch.text[0])).to(args.device)
                setattr(batch, 'char', char)

        pred = model(batch)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()

        pred = pred.detach()
        if args.thresholding:
            if reverse_prior.type() != pred.data.type():
                reverse_prior = reverse_prior.type_as(pred.data).to(pred.get_device())
            at = reverse_prior.repeat(pred.size()[0], 1)
            pred = torch.softmax(pred, dim=1) * at

        preds.append(pred.cpu().numpy())
        labels.append(batch.label.cpu().numpy())
        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float().cpu().item()
        size += len(pred)

    preds = np.concatenate(preds)
    labels = to_categorical(np.concatenate(labels), num_classes=args.class_size)

    acc_, prec, rec, f1 = getMetrics(preds, labels, data)
    acc /= size
    return loss, acc, f1


def getMetrics(predictions, ground, data):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    num_classes = len(data.LABEL.vocab)
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1), num_classes=num_classes)

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    #print("True Positives per class : ", truePositives)
    #print("False Positives per class : ", falsePositives)
    #print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, num_classes):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("    Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (data.LABEL.vocab.itos[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                                 macroPrecision + macroRecall) > 0 else 0
    #print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
    #macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    #print(
    #    "Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                                 microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("    Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


