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
- Cuda 11.6
- Python 3.9.9
- Tensorflow 2.6

I further had to manually (!) install a matching keras with pip:

``` 
pip install keras==2.6
```

This repository has **two** requirements.txt files - one is for windows, while the *reduced_requirements.txt* is for the docker-container.
The precise windows versions where not available for the docker-ubuntu. 

## Changes

1. Ran [Tensorflow Migration Skript](https://blog.tensorflow.org/2019/02/upgrading-your-code-to-tensorflow-2-0.html)
2. Adjusted the re-Shape for cost function, as a different format was required in tfa
3. Adjusted the reshape for cost function for completion and perplexity separately
4. Some prints (might be removed ...)
5. Adjusted the loss-functions default behavior to not average out over batch (done later manually)
6. Added DockerFile & Reduced Requirements