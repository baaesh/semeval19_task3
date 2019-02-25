# Semi-Hierarchical Bi-LSTM Encoder (SHBLE) for SemEval-Task3 EmoContext

## Results
Dataset: [EmoContext](https://www.humanizing-ai.com/emocontext.html)

Single Models

| Model | Test Acc | std | Test micro F1 | std |
| ----- | ------------ | ----------- | ----------- | ----------- |
| Baseline (organizers) | - | - | .587 | - |
| Plain | .914 | .005 | .726 | .008 |
| Oversampling | .922 | .004 | .733 | .012 |
| Undersampling | .919 | .006 | .719 | .013 |
| Thresholding | .924 | .002 | .738 | .010 |
| Cost-Sensitive | .924 | .004 | .739 | .010 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.6
- Pytorch: 0.4.1

## Requirements
Please install the following library requirements first.

    nltk==3.3
    tensorboardX==1.2
    torch==0.4.1
    torchtext==0.2.3
