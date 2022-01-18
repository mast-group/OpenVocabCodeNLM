# OpenVocabNLMs
Contains the code for our ICSE 2020 submission: open vocabulary language model for source code that uses the byte pair encoding algorithm to learn a segmentation of code tokens into subtokens. 

If you use our code/implementation, datasets or pre-trained models please cite our paper:
@inproceedings{Karampatsis2020ICSE,\
author = {Karampatsis, Rafael - Michael and Babii, Hlib and Robbes, Romain and Sutton, Charles and Janes, Andrea},\
title = {{Big Code != Big Vocabulary: Open-Vocabulary Models for Source code}},\
year = {2020},\
publisher = {ACM},\
url = {https://doi.org/10.1145/3377811.3380342}, \
doi = {10.1145/3377811.3380342},\
booktitle = {Proceedings of the 42nd International Conference on Software Engineering},\
pages = {},\
numpages = {11},\
location = {Seoul, South Korea},\
series = {ICSE ’20}\
}


# Code Structure
**non-ascii_sequences_to_unk.py** is a preprocessing script that can be used to remove non-ascii sequences from the data and replace them with a special symbol.

**create_subtoken_data.py** is also a preprocessing script that can be used to subtokenize data based on the heuristic of [Allamanis et al. (2015)](https://miltos.allamanis.com/publications/2015suggesting/).

**reader.py** contains utility functions for reading data and providing batches for training and testing of models.

**code_nlm.py** contains the implementation of our NLM for code and supports training, perplexity/cross-entropy calculation, code-completion simulation as well as dynamic versions of the test scenarios. The updated implementation has also some new features, previously not present in the code. That is measuring identifier specific performance for code completion. Another new feature implements a simple n-gram cache for identifiers that better simulates use of the model in an IDE where such information would be present. In order to use the identifier features a file containing identifier information must be provided through the options. 

# Installation

Python>2.7.6 or Python==3.6 is required!
Python>3.6 is not supported due to the tensorflow version not supporting it.

```shell script
git clone https://github.com/mast-group/OpenVocabCodeNLM
cd OpenVocabCodeNLM
pip install -r requirements.txt #python2
pip3 install -r requirements.txt #python3
```
The experiments in the paper were performed using Python 2.7.14 but we have currently not experienced any unresolved issue with Python 3. </br>
In case you encounter any issues please open a new issue entry.


# Usage Instructions
If you want to try the implementation unzip the directory containing the sample data.
The sample data contain the small training set, validation, and test set used in the paper with a BPE encdoding size of 10000.

## Option Constants
Let's first define constants for pointing to the data and network parameters. You'll need to modify these to point to your own data and satisfy the hyperparameters that you want to use.
```
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
EPOCHS=5 # Normally this would be larger. For instance 30-50
# Initial learning rate
LR=0.1 # This is the default value. You can skip it if you don't want to change it.
# Training batch size
BATCH_SIZE=32 # This is also the default.
# RNN unroll timesteps for gradient calculation.
STEPS=20 # 20-50 is a good range of values for dynamic experiments.
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
```


We next present the various scenarios supported by our implementation.

## Training
The training scenario creates a global model by training on the provided to it training data.
We will train a Java model with a BPE encoding of 10000 using the sample data.
In the following training example we set some of the hyperparameters (to their default values though).
Optionally, you can set all of them to your intented values.
Since the data is tokenized into subwords we need to let the script know so that it can calculate the metrics correctly.
For this reason we need to set the *word_level_perplexity* flag to **True**.
In order to also output validation cross-entropy instead of perplexity we set the *cross_entropy* option to **True**.

```
# Train a small java model for 1 epoch.
python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS

# Because we are using the default values we could shorten the above command to:
# python code_nlm.py --data_path $DATA_HOME --train_dir $MODEL_DIR --train_filename $TRAIN_FILE --validation_filename $VALIDATION_FILE --gru True --word_level_perplexity True --cross_entropy True --max_epoch $EPOCHS
```

## Test Scenarios
### Test Entropy Calculation
```
# Testing the model (Calculating test set entropy) 
python code_nlm.py --test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True
```

#### Dynamically Adapt the Model on Test Data
In order to dynamically adapt the model, the implementation needs to know when it is testing on a new project, so that it can revert the model back to the global one.
This is achieved via the *test_proj_filename* option.
```
# Batch size must always be set to 1 for this scenario! We are going through every file seperately.
# In an IDE this could instead be sped up through some engineering.
python code_nlm.py --dynamic_test True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size 1 --word_level_perplexity True --cross_entropy True --test_proj_filename $TEST_PROJ_NAMES_FILE --num_steps $STEPS
```

### Test Code Completion
In this scenario the *batch_size* option is used to set the beam size.
```
python code_nlm.py --completion True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE
```

#### Dynamic Code Completion on Test Data
Similarly to before we need to set the *test_proj_filename* option.
```
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE --num_steps $STEPS
```

#### Dynamic Code Completion on Test Data and Measuring Identifier Specific Performance
To run this experiment you need to provide a file containing a mapping that lets the implementation know for each subtoken whether it is part of an identifier or not.
This information would easily be present in an IDE.
The mapping is provided via the *identifier_map* option.
```
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE  --test_proj_filename $TEST_PROJ_NAMES_FILE --identifier_map $ID_MAP_FILE --num_steps $STEPS
```

#### Adding a Simple Identifier n-gram Cache
In an IDE setting we could improve the performance on identifiers by utilizing  a simple n-gram cache for identifiers that we have already encountered.
The *file_cache_weight* and  *cache_order* options can be used to control the cache's weight and the cache's order respectively.
By default we use a 6-gram with a weight of 0.2.
```
python code_nlm.py --completion True --dynamic True --data_path $DATA_HOME --train_dir $MODEL_DIR --test_filename $TEST_FILE --gru True --batch_size $BATCH_SIZE --test_proj_filename $TEST_PROJ_NAMES_FILE --identifier_map $ID_MAP_FILE --cache_ids True --num_steps $STEPS
```

### Predictability
Similar to testing but calculates the average entropy of the files instead of the per token one.



# Preprocessing

## BPE
The BPE implementation used can be found here: https://github.com/rsennrich/subword-nmt 

To apply byte pair encoding to word segmentation, invoke these commands:
```
subword-nmt learn-bpe -s {num_operations} < {train_file} > {codes_file}
subword-nmt apply-bpe -c {codes_file} < {test_file} > {out_file}
```
num_operations = The number of BPE ops e.g., 10000 <br/>
train_file = The file on which to learn the encoding <br/>
codes_file = The file in which to output the learned encoding <br/>
test_file = The file to segment with the learned encoding <br/>
out_file = The file in which to save the now segmented test_file <br/>

