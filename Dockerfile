FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3
# The Tag 22.01 stands for january 2022.
# The version 22.01 comes with tensorflow 2.6 and python 3.8

WORKDIR /openvocabcodenlm

COPY reduced_requirements.txt .
RUN pip install -r reduced_requirements.txt


#ENTRYPOINT ["python","--version"]

ENTRYPOINT ["pip","list"]
#ENTRYPOINT ["echo","Hello Docker GPU!"]