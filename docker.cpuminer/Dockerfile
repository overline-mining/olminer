# !!!!DO NOT PUSH/PUBLISH - MAY EXPOSE PRIVATE DOCKER CREDENTIALS!!!!
FROM debian:stable-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends \ 
    build-essential \
    ca-certificates \
    cmake \
    git

# TODO: pull from git repo once public
RUN mkdir -p /usr/src/olminer
COPY ./ /usr/src/olminer

WORKDIR /usr/src/olminer
RUN git submodule update --init --recursive
RUN rm -rf ./build ; mkdir -p ./build && cd ./build && cmake -DOLHASHCUDA=off -DCMAKE_BUILD_TYPE=Release -DHUNTER_JOBS_NUMBER=$(nproc) .. && make install

FROM debian:stable-slim
RUN mkdir -p /home/bc
WORKDIR /home/bc
COPY --from=0 /usr/local/bin/olminer /home/bc
USER 1001
CMD ["sh", "-c", "/home/bc/olminer --cpu -P ${STRATUM_URL}"]

