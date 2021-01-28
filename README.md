# olminer

[![Releases](https://img.shields.io/github/downloads/overline-mining/olminer/total.svg)][Releases]

> Overline miner with CUDA (eventually OpenCL) and stratum support

**Olminer** is an olhash CPU and GPU mining worker: with olminer you can mine overline proof of distance compliant blockchain, so for now, only Overline. This is the actively maintained version of olminer. It originates from the [ethminer](https://github.com/ethereum-mining/ethminer) project and adapts its framework to Overline's proof of work system. See [FAQ](#faq) for more details. Many thanks to the ethminer devs since without their code this would not exist.

## Features

* CPU Mining
* Nvidia CUDA mining
* OpenCL mining (not yet there)
* realistic benchmarking against arbitrary blocks (to be done)
* stratum mining without proxy
* OpenCL devices picking (to do, but shouldn't be difficult)
* farm failover (getwork + stratum)


## Table of Contents

* [Install](#install)
* [Usage](#usage)
    * [Examples connecting to pools](#examples-connecting-to-pools)
* [Build](#build)
    * [Continuous Integration and development builds](#continuous-integration-and-development-builds)
    * [Building from source](#building-from-source)
* [Maintainers & Authors](#maintainers--authors)
* [Contribute](#contribute)
* [F.A.Q.](#faq)


## Install

1. clone this repository
2. make sure you have [docker](https://docs.docker.com/get-docker/) installed
3. build
    * if you are on linux follow the instructions in `docker.olminer/README`
    * if you are on windows that docker file is on the way :-)
4. copy the executable out of the docker container
    * `docker create -ti --name dummy olminer bash && sudo docker cp dummy:/home/bc/olminer /usr/local/bin ; docker rm -f dummy`
5. If you want to mine with a GPU make sure you have [CUDA 11](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#support-hardware) compatible drivers installed.

Note for 3060-3090 users: you need to change the build and base images in the Dockerfile to start with `nvidia/cuda:11.1.1` instead of `nvidia/cuda:11.0` and your drivers must be compatible with CUDA 11.1.

## Usage

The **olminer** is a command line program. This means you launch it either
from a Windows command prompt or Linux console, or create shortcuts to
predefined command lines using a Linux Bash script or Windows batch/cmd file.
For a full list of available command, please run:

```sh
olminer --help
```

### Examples connecting to pools

Check our [samples](docs/POOL_EXAMPLES_OL.md) to see how to connect to different pools.

## Build

### Continuous Integration and development builds

| CI            | OS            | Status  | Development builds |
| ------------- | ------------- | -----   | -----------------  |


To download the .exe on a build under `Job name` select the CUDA version you use, choose `Artifacts` then download the zip file.

### Building from source

See [docs/BUILD.md](docs/BUILD.md) for build/compilation details.

## Maintainers & Authors

[![Gitter](https://img.shields.io/gitter/room/overline-mining/olminer.svg)][Gitter]

The list of current and past maintainers, authors and contributors to the olminer project.
Ordered alphabetically. [Contributors statistics since 2015-08-20].

| Name                  | Contact                                                      |     |
| --------------------- | ------------------------------------------------------------ | --- |
| LG      | [@lgray](https://github.com/lgray)       |     |


## Contribute

[![Gitter](https://img.shields.io/gitter/room/overline-mining/olminer.svg)][Gitter]

To meet the community, ask general questions and chat about olminer join [the olminer channel on Gitter][Gitter].

All bug reports, pull requests and code reviews are very much welcome.


## License

Licensed under the [GNU General Public License, Version 3](LICENSE).


## F.A.Q

### Why is my hashrate with Nvidia cards on Windows 10 so low?

The new WDDM 2.x driver on Windows 10 uses a different way of addressing the GPU. This is good for a lot of things, but not for OL mining.

* For Kepler GPUs: I actually don't know. Please let me know what works best for good old Kepler.
* For Maxwell 1 GPUs: Unfortunately the issue is a bit more serious on the GTX750Ti, already causing suboptimal performance on Win7 and Linux. Apparently about 4MH/s can still be reached on Linux, which, depending on OL price, could still be profitable, considering the relatively low power draw.
* For Maxwell 2 GPUs: There is a way of mining OL at Win7/8/Linux speeds on Win10, by downgrading the GPU driver to a Win7 one (350.12 recommended) and using a build that was created using CUDA 6.5.
* For Pascal GPUs: You have to use the latest WDDM 2.1 compatible drivers in combination with Windows 10 Anniversary edition in order to get the full potential of your Pascal GPU.

### Why is a GTX 1080 slower than a GTX 1070?

Because of the GDDR5X memory, which can't be fully utilized for OL mining (yet).

### Are AMD cards also affected by slowdowns with increasing DAG size?

Only GCN 1.0 GPUs (78x0, 79x0, 270, 280), but in a different way. You'll see that on each new epoch (30K blocks), the hashrate will go down a little bit.

### Can I still mine OL with my 4GB GPU?

LOL, yes.

### What are the optimal launch parameters?

The default parameters are fine in most scenario's (CUDA). For OpenCL it varies a bit more. Just play around with the numbers and use powers of 2. GPU's like powers of 2.

### What does the `--cuda-parallel-hash` flag do?

[@davilizh](https://github.com/davilizh) made improvements to the CUDA kernel hashing process and added this flag to allow changing the number of tasks it runs in parallel. These improvements were optimised for GTX 1060 GPUs which saw a large increase in hashrate, GTX 1070 and GTX 1080/Ti GPUs saw some, but less, improvement. The default value is 4 (which does not need to be set with the flag) and in most cases this will provide the best performance.

### What is olminer's relationship with [Genoil's fork]?

[Genoil's fork] was the original source of this version, but as Genoil is no longer consistently maintaining that fork it became almost impossible for developers to get new code merged there. In the interests of progressing development without waiting for reviews this fork should be considered the active one and Genoil's as legacy code.

### Can I CPU Mine?

Yes --cpu.

### CUDA GPU order changes sometimes. What can I do?

There is an environment var `CUDA_DEVICE_ORDER` which tells the Nvidia CUDA driver how to enumerates the graphic cards.
The following values are valid:

* `FASTEST_FIRST` (Default) - causes CUDA to guess which device is fastest using a simple heuristic.
* `PCI_BUS_ID` - orders devices by PCI bus ID in ascending order.

To prevent some unwanted changes in the order of your CUDA devices you **might set the environment variable to `PCI_BUS_ID`**.
This can be done with one of the 2 ways:

* Linux:
    * Adapt the `/etc/environment` file and add a line `CUDA_DEVICE_ORDER=PCI_BUS_ID`
    * Adapt your start script launching olminer and add a line `export CUDA_DEVICE_ORDER=PCI_BUS_ID`

* Windows:
    * Adapt your environment using the control panel (just search `setting environment windows control panel` using your favorite search engine)
    * Adapt your start (.bat) file launching olminer and add a line `set CUDA_DEVICE_ORDER=PCI_BUS_ID` or `setx CUDA_DEVICE_ORDER PCI_BUS_ID`. For more info about `set` see [here](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/set_1), for more info about `setx` see [here](https://docs.microsoft.com/en-us/windows-server/administration/windows-commands/setx)

### Insufficient CUDA driver

```text
Error: Insufficient CUDA driver: 9010
```

You have to upgrade your Nvidia drivers. On Linux, install `nvidia-396` package or newer.


[Amazon S3 is needed]: https://docs.travis-ci.com/user/uploading-artifacts/
[Contributors statistics since 2015-08-20]: https://github.com/overline-mining/olminer/graphs/contributors?from=2015-08-20
[Gitter]: https://gitter.im/overline-mining/olminer
[Releases]: https://github.com/overline-mining/olminer
