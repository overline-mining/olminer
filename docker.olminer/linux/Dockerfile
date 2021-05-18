FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends \ 
    build-essential \
    ca-certificates \
    cmake \
    git

RUN git clone https://github.com/overline-mining/olminer /usr/src/olminer

WORKDIR /usr/src/olminer
RUN git submodule update --init --recursive
RUN rm -rf ./build ; mkdir -p ./build && cd ./build && cmake -DOLHASHCUDA=ON -DCMAKE_BUILD_TYPE=Release -DHUNTER_JOBS_NUMBER=$(nproc) .. && make install

FROM nvidia/cuda:11.1.1-base-ubuntu20.04
RUN mkdir -p /home/bc
WORKDIR /home/bc
COPY --from=0 /usr/local/bin/olminer /home/bc
USER 1001
CMD ["sh", "-c", "/home/bc/olminer --${MINER_TYPE} -P ${STRATUM_URL}"]

