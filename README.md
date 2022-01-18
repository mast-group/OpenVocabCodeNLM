# OpenVocabCodeNLM - Reproduction & Bump

For an Experiment I wanted to reproduce OpenVocabNLM, 
and while it worked fine with CPUs on my MacBook, 
it failed on my Cuda11 Windows machine and the GPUs. 

So this repository aims to bump the versions to be SOTA again, 
with modern python, tensorflow and hopefully dockered. 

Also see the [original Readme](./original_README.md).
See the [original repository](https://github.com/mast-group/OpenVocabCodeNLM)

## Environment

- Windows 10
- Cuda 11
- Python 3.9
- Tensorflow 2.7

## Changes

1. Ran [Tensorflow Migration Skript](https://blog.tensorflow.org/2019/02/upgrading-your-code-to-tensorflow-2-0.html)