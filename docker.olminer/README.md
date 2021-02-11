# Docker olminer

This will build a CPU miner binary optimized for the CPU found on the system.
You will need to have docker and git installed on your operating system and accessible by command line for this build process to work.

You will need to have windows 10 for the windows build process and executables to function.

### If you need help ask in https://t.me/overlineadvanced - do not email code authors directly.

## Build

```bash
git clone https://github.com/overline-mining/olminer.git
cd olminer
docker build -t olminer -f ./docker.olminer/<your OS>/Dockerfile .
```

### NOTE: In order to build olminer on windows you will:
  1. Need approximately 40GB of disk space for the build image. Microsoft loves cumbersome build tools.
  2. Need to be running docker desktop in [windows container mode](https://markheath.net/posts/2019/docker-desktop-windows-and-linux-1.png)

You can recover the used space afterwards by doing:
  ```bash
  docker image prune
  docker rmi mcr.microsoft.com/windows:2004 mcr.microsoft.com/windows/servercore:2004
  ```

## Run CPU miner inside Docker

To start the previously built miner, use ```docker run -d --restart=unless-stopped -e MINER_TYPE=cpu -e STRATUM_URL=stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@www.waietng-pool.us:3141 --name olminer-cpu olminer```

If don't want to all CPU threads available to the miner use something like ```docker run -d --cpus=4 --restart=unless-stopped -e MINER_TYPE=cpu -e STRATUM_URL=stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@www.waietng-pool.us:3141 --name olminer-cpu olminer```

To kill the miner type ```docker rm -f olminer-cpu```

## Run CUDA GPU miner natively (easier to setup harder to manage)

On either linux or windows you will need to have CUDA 11.1 compatible drivers installed for this to function.
On windows you will additionally need to install the [runtimes for MS Visual Studio 2019](https://aka.ms/vs/16/release/VC_redist.x64.exe).

You can run the miner outside of Docker. After successfully building the CPU miner, grab the miner binary using
- linux: ```docker create -ti --name dummy olminer bash && sudo docker cp dummy:/home/bc/olminer /usr/local/bin ; docker rm -f dummy```
- windows: ```docker create -ti --name dummy olminer cmd; docker cp dummy:C:/olminer/olminer.exe ./; docker rm -f dummy```
If the command completed without errors, the miner binary should be in /usr/local/bin if in linux and the current directory if in windows.

You then run the command like this:
```olminer --cuda -P stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@www.waietng-pool.us:3141```

## Run CUDA miner inside Docker (harder to setup easier to manage)

This will only run on linux with the [nvidia container runtimes](https://nvidia.github.io/nvidia-container-runtime/) installed.
Similarly for windows containers you will have provide the gpus to the docker containers. 
Neither of these are easy things to do (but it is significantly easier in linux).

To start the previously built miner, use ```docker run -d --gpus all --restart=unless-stopped -e MINER_TYPE=cuda -e STRATUM_URL=stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@www.waietng-pool.us:3141 --name olminer-cuda olminer```

If don't want to all GPUs available to the miner use something like ```docker run -d --gpus device=0,1 --restart=unless-stopped -e MINER_TYPE=cuda -e STRATUM_URL=stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@www.waietng-pool.us:3141 --name olminer-cuda olminer```

To kill the miner type ```docker rm -f olminer-cuda```
