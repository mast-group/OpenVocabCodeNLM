#!/bin/bash
# Entrypoint for OpenVocabCodeNLM Experiment

# This file invokes the original python code of the openvocabcodenlm with the environment variables set in the docker container.
# Additionally, it does a switch-case which flags for training, validation and testing have been set.


# Run mkdir Model_dir, in case it does not exist yet (if you start with training)
mkdir $MODEL_DIR


# Training the model
if [ "$DO_TRAIN" = true ]; then
  if [ "$VERBOSE" = true ]; then
    python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS --verbose True
  else
    python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS
  fi
fi

# Testing the model (Calculating test set entropy)
if [ "$DO_TEST" = true ]; then
  if [ "$VERBOSE" = true ]; then
    python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --verbose True
  else
    python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True
  fi
fi

# Code completion
if [ "$DO_COMPLETION" = true ]; then
  if [ "$VERBOSE" = true ]; then
    python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --verbose True
  else
    python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE
  fi
fi

# Add this to keep the container open (e.g. for debugging or inspection)
#echo "Entrypoint finished - keeping container artifially open ..."
#tail -f /dev/null