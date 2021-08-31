#!/usr/bin/env bash

DOCKER_NAME="my_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
      shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done


docker run -v $(pwd)/igibson.key:/opt/iGibson/igibson/data/igibson.key -v $(pwd)/ig_dataset:/opt/iGibson/igibson/data/ig_dataset -v $(pwd)/results:/results \
    --gpus=all \
    ${DOCKER_NAME} \
    /bin/bash -c \
    "export CONFIG_FILE=/opt/iGibson/igibson/examples/configs/behavior_onboard_sensing.yaml; export PHASE=minival; export OUTPUT_DIR=/results; bash submission.sh"
    
# make sure CONFIG_FILE, TASK and EPISODE_DIR are consistent
# you can use TASK environment variable to switch agent in agent.py
