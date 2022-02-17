FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3
# The Tag 22.01 stands for january 2022.
# The version 22.01 comes with tensorflow 2.6 and python 3.8

# Entry-Point Orchestration-Flags
ENV DO_TRAIN="true"
ENV DO_TEST="true"
ENV DO_COMPLETION="true"

ENV VERBOSE="false"

# Filenames
ENV TRAIN_FILE=java_training_slp_pre_enc_bpe_10000
ENV VALIDATION_FILE=java_validation_slp_pre_enc_bpe_10000
ENV TEST_FILE=java_test_slp_pre_enc_bpe_10000
ENV TEST_PROJ_NAMES_FILE=testProjects
ENV ID_MAP_FILE=/data/java/id_map_java_test_slp_pre_bpe_10000
# Directory that contains train/validation/test data etc.
ENV DATA_HOME=/data/java/
# Directory in which the model will be saved.
ENV MODEL_DIR=/models/java/model

# Maximum training epochs
ENV EPOCHS=2
# Initial learning rate
ENV LR=0.1
# Training batch size
ENV BATCH_SIZE=32
# RNN unroll timesteps for gradient calculation.
ENV STEPS=200
# 1 - Dropout probability
ENV KEEP_PROB=0.5
# RNN hidden state size
ENV STATE_DIMS=512
# Checkpoint and validation loss calculation frequency.
ENV CHECKPOINT_EVERY=5000


WORKDIR /openvocabcodenlm

COPY reduced_requirements.txt .
RUN pip install -r reduced_requirements.txt

COPY util util
COPY reader.py .
COPY code_nlm.py .
COPY create_subtoken_data.py .
COPY non-ascii_sequences_to_unk.py .

COPY entrypoint.sh .

ENTRYPOINT ["bash","entrypoint.sh"]