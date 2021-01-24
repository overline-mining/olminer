# Docker olminer

This will build a CPU miner binary optimized for the CPU found on this system.

## Build

```docker build -t olminer-cpu -f ./docker.cpuminer/Dockerfile .```

## Run miner inside Docker

To start the previously built miner, use ```docker run -e STRATUM_URL=stratum://0x6a37b22b7fbbe8cc1f34c86e729905472e342e01@157.245.116.220:3141 -d -n olminer-cpu olminer-cpu```

## Run miner natively

If you prefer, you can run the miner outside of Docker. After successfully building the CPU miner, grab the miner binary using
```docker create -ti --name dummy olminer-cpu bash && sudo docker cp dummy:/home/bc/olminer /usr/local/bin ; docker rm -f dummy```

If the command completed without errors, the miner binary should be in /usr/local/bin.
