# Addressing Training-Test Class Distribution Mismatch in Conversational Classification for SemEval-2019 Task3 EmoContext
This is the implementation of Semi-Hierarchical Bi-LSTM Encoder (SHBLE), for [SemEval-2019](http://alt.qcri.org/semeval2019/) Task 3 - [EmoContext: Contextual Emotion Detection in Text](https://www.humanizing-ai.com/emocontext.html).

The description was included in this following paper. [link](https://arxiv.org/abs/1903.02163)

    @article{bae2019snu_ids,
        title={SNU\_IDS at SemEval-2019 Task 3: Addressing Training-Test Class Distribution Mismatch in Conversational Classification},
        author={Bae, Sanghwan and Choi, Jihun and Lee, Sang-goo},
        year={2019}
    }

## Results
Dataset: [EmoContext](https://www.humanizing-ai.com/emocontext.html)

Single Models

| Model | Test Acc | std | Test F1 | std |
| ----- | ------------: | -----------: | -----------: | -----------: |
| Baseline (organizers) | - | - | .587 | - |
| Plain | .914 | .005 | .726 | .008 |
| Oversampling | .922 | .004 | .733 | .012 |
| Undersampling | .919 | .006 | .719 | .013 |
| Thresholding | .924 | .002 | .738 | .010 |
| Cost-Sensitive | .924 | .004 | .739 | .010 |


Ensemble Models

| Model | Test Acc | Test F1 |
| ----- | ------------: | -----------: |
| Plain | .921 | .743 |
| Oversampling | .930 | .758 |
| Undersampling | .930 | .753 |
| Thresholding | .930 | .752 |
| Cost-Sensitive | .931 | .757 |
| Mixed (submitted) | .933 | .766 |

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
