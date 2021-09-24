#!/usr/bin/env bash
cd `dirname $0`

. h-manifest.conf

while true; do
  for con in `netstat -anp | grep TIME_WAIT | grep $CUSTOM_API_PORT | awk '{print $5}'`; do
    killcx $con lo
  done
  netstat -anp | grep TIME_WAIT | grep $CUSTOM_API_PORT &&
    continue ||
    break
done

#wget https://PATHTOCONTROLFILE/control
#if [ $(cat control | grep -i "Version: $CUSTOM_VERSION" | wc -l) -eq 0 ];
#	then
#		mv $CUSTOM_CONFIG_FILENAME /hive/miners/custom/;
#		/hive/miners/custom/custom-get https://PATHTOFILE/FILE.tar.gz -f;
#		mv /hive/miners/custom/config.conf $CUSTOM_CONFIG_FILENAME;
#fi
#rm -f control

cd /hive/miners/custom/$CUSTOM_NAME
export GPU_FORCE_64BIT_PTR=1
export GPU_USE_SYNC_OBJECTS=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export GPU_MAX_HEAP_SIZE=100
ls
#./olminer --log-file=$CUSTOM_LOG_BASENAME.log --api-port=$CUSTOM_API_PORT $(< $CUSTOM_CONFIG_FILENAME) $@ 2>&1
olminer/olminer  --HWMON 2 --api-port=$CUSTOM_API_PORT $(< $CUSTOM_CONFIG_FILENAME) $@ 2>&1