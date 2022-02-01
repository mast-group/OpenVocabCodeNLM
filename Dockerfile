FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3
# The Tag 22.01 stands for january 2022.
# The version 22.01 comes with tensorflow 2.6 and python 3.8

WORKDIR /openvocabcodenlm

COPY reduced_requirements.txt .
RUN pip install -r reduced_requirements.txt

COPY sample_data sample_data
COPY util util
COPY reader.py .
COPY code_nlm.py .
COPY create_subtoken_data.py .
COPY non-ascii_sequences_to_unk.py .

COPY example.sh .

#ENTRYPOINT ["tail","-f","/dev/null"]
ENTRYPOINT ["bash","example.sh"]