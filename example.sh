#!/bin/bash

# Directory that contains train/validation/test data etc.
DATA_HOME=sample_data/java/
# Directory in which the model will be saved.
MODEL_DIR=sample_data/java/model
mkdir $MODEL_DIR

# Filenames
TRAIN_FILE=java_training_slp_pre_enc_bpe_10000
VALIDATION_FILE=java_validation_slp_pre_enc_bpe_10000
TEST_FILE=java_test_slp_pre_enc_bpe_10000
TEST_PROJ_NAMES_FILE=testProjects
ID_MAP_FILE=sample_data/java/id_map_java_test_slp_pre_bpe_10000

# Maximum training epochs
EPOCHS=5
# Initial learning rate
LR=0.1 # This is the default value. You can skip it if you don't want to change it.
# Training batch size
BATCH_SIZE=32 # This is also the default.
# RNN unroll timesteps for gradient calculation.
STEPS=200 # This also the default.
# 1 - Dropout probability
KEEP_PROB=0.5 # This is also the default.
# RNN hidden state size
STATE_DIMS=512 # This is also the default.
# Checkpoint and validation loss calculation frequency.
CHECKPOINT_EVERY=5000 # This is also the default.


# Understanding boolean options.
# Most boolean options are set to False  by default.
# For using any boolean option set it to True.
# For instance for using a GRU instead of an LSTM add to your command the option --gru True.


# Train a small java model for 1 epoch.
python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS

# Because we are using the default values we could have shortened the command to:
# python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --word_level_perplexity True --cross_entropy True --max_epoch $EPOCHS


# Testing the model (Calculating test set entropy) 
python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True

# Test a dynamically adapted version of the model.
# Batch size must always be set to 1 for this scenario!
python code_nlm.py --dynamic_test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size 1 --word_level_perplexity True --cross_entropy True --test_proj_filename $TEST_PROJ_NAMES_FILE

# Code completion
python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE

# Dynamic Code Completion
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE

# Dynamic Code Completion and Measuring Identifier Performance
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE --identifier_map $ID_MAP_FILE

# Adding a Simple Identifier n-gram Cache
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE --identifier_map $ID_MAP_FILE --cache_ids True

