#!/usr/bin/env bash

THIS_MINER_NAME="custom"
[ -t 1 ] && . colors

[[ `ps aux | grep "./xmrig-amd " | grep -v grep | wc -l` != 0 ]] &&
<-->echo -e "${RED}$CUSTOM_NAME miner is already running${NOCOLOR}" &&
<-->exit 1

[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}CUSTOM_TEMPLATE is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}CUSTOM_URL is empty${NOCOLOR}" && return 1
#[[ -z $CUSTOM_ALGO ]] && echo -e "${YELLOW}CUSTOM_ALGO is empty${NOCOLOR}" && return 1

#conf=" -o ${CUSTOM_URL} -u ${CUSTOM_TEMPLATE} ${CUSTOM_USER_CONFIG} -p ${CUSTOM_PASS}"
conf=" -P ${CUSTOM_URL}"

#replace tpl values in whole file
[[ ! -z $USER_CONFIG ]] && conf=$(sed "s/%USER_CONFIG%/$USER_CONFIG/g" <<< "$conf") #|| echo "${RED}USER_CONFIG not set${NOCOLOR}"
[[ ! -z $WORKER_NAME ]] && conf=$(sed "s/%WORKER_NAME%/$WORKER_NAME/g" <<< "$conf") #|| echo "${RED}WORKER_NAME not set${NOCOLOR}"

[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && return 1
echo "$conf" > $CUSTOM_CONFIG_FILENAME
