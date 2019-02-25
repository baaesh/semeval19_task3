## Requirements

Please download following pre-trained embeddings:

- [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) at data/word2vec

- [emoji2vec](https://github.com/uclmr/emoji2vec/tree/master/pre-trained) at data/emoji2vec

- [datastories twitter.300d](https://mega.nz/#!u4hFAJpK!UeZ5ERYod-SwrekW-qsPSsl-GYwLFQkh06lPTR7K93I) at data/datastories

- elmo [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
and [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5) at data/elmo

## Training
You can train a model with default hyperparameters by just typing,
    
> python train.py
    
Also, there are other configures.

> python train.py --help

    usage: train.py [-h] [--train_data_path TRAIN_DATA_PATH]
                [--valid_data_path VALID_DATA_PATH]
                [--test_data_path TEST_DATA_PATH]
                [--elmo_option_path ELMO_OPTION_PATH]
                [--elmo_weight_path ELMO_WEIGHT_PATH]
                [--ss_vector_path SS_VECTOR_PATH] [--no_emoji_preprocess]
                [--word_dim WORD_DIM] [--d_e D_E] [--num_heads NUM_HEADS]
                [--d_ff D_FF] [--dist_mask] [--alpha ALPHA] [--seg_emb]
                [--seg_emb_share] [--pos_emb] [--elmo_num ELMO_NUM]
                [--no_elmo_feed_forward] [--elmo_dim ELMO_DIM] [--no_char_emb]
                [--char-dim CHAR_DIM] [--num-feature-maps NUM_FEATURE_MAPS]
                [--ss_emb] [--ss_emb_tune] [--fasttext] [--fasttext_tune]
                [--no_word2vec] [--word2vec_tune] [--no_datastories]
                [--no_lstm_bidirection] [--lstm_num_layers LSTM_NUM_LAYERS]
                [--lstm_hidden_dim LSTM_HIDDEN_DIM] [--simple_encoder]
                [--uni_encoder] [--fusion] [--no_share_encoder] [--separate]
                [--turn2] [--no_turn2] [--biattention] [--device DEVICE]
                [--tune_embeddings] [--dropout DROPOUT]
                [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                [--norm_limit NORM_LIMIT] [--lr_gamma LR_GAMMA]
                [--weight_decay WEIGHT_DECAY] [--max_epoch MAX_EPOCH]
                [--print_every PRINT_EVERY] [--validate_every VALIDATE_EVERY]
                [--oversampling] [--undersampling] [--wce_loss]
                [--thresholding] [--mfe_loss] [--mfe_alpha MFE_ALPHA]
                [--name_tag NAME_TAG]
