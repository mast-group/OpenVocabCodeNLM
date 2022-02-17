# OpenVocabCodeNLM - Reproduction & Bump

This is an reproduction and adjustment of OpenVocabNLM.
I ran into issues mostly with different versions and GPU-drivers, 
so this repository aims to bump the versions to be SOTA again, 
with modern python, tensorflow and docker. 

Also see the [original Readme](./original_README.md).
See the [original repository](https://github.com/mast-group/OpenVocabCodeNLM)


## Changes

1. Ran [Tensorflow Migration Skript](https://blog.tensorflow.org/2019/02/upgrading-your-code-to-tensorflow-2-0.html)
2. Adjusted the re-Shape for cost function, as a different format was required in tfa
3. Adjusted the reshape for cost function for completion and perplexity separately
4. Some prints (might be removed ...)
5. Adjusted the loss-functions default behavior to not average out over batch (done later manually)
6. Added DockerFile & Reduced Requirements
7. Changed some prints to be logging

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

**Optionally**: 
This repository provides a docker file starting from the official NVidia Tensorflow Image.
This should utilize a properly set up GPU on a linux machine - other OS are not supported! 
For any non-supported OS, or insufficiently set up Linux machines, it will default using CPUs. 
This is printed when the container starts.
The examples were created and run with Docker 20.10 and Docker-compose v2.2.3 .

The containers are **very memory hungry**. Limiting resources in the compose is highly advised. 

## Licence Warning

The original OpenVocabCodeNLM has Apache Licence (same as this fork).

But the used nvidia-container comes with an implicit licence agreement. Please study it carefully before using it.

## Troubleshooting 

### Warnings / Errors in Python

**Allocation exceeds free system memory**

```
[...] Allocation of 234393600 exceeds 10% of free system memory.
[...] Allocation of 234393600 exceeds 10% of free system memory.
[...] Allocation of 234393600 exceeds 10% of free system memory.
[...]
```
This error is likely related to the Batchsize. It can occur in or outside of docker. 
**Try reducing the batchsize**.
For older graphics cards try ~64, for non-gpu try batch sizes from 16 upward.

### Warnings in/with Docker

**FileNotFoundError**

```
openvocabcodenlm-experiment-1  | FileNotFoundError: [Errno 2] No such file or directory: '/data/java/java_test_slp_pre_enc_bpe_10000'
```

This is likely happening because there was a missmatch in mounting the volumes.
The way volume-paths are defined (e.g. ending with "/" *unrolls* the directory-elements into the mounted volume) in the docker-compose 
must match the behavior of the python script.
**Solution:** First, add the `tail -f /dev/null` to the end of the `entrypoint.sh`. 
Run the compose, and find your container with `docker ps`. Enter your docker container with `docker exec -it {ID} bash`.
Inspect the `/data` folder and see whether it matches your expectations, and adjust the values in the compose accordingly.

