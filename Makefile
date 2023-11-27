NAME=faster-lcm
TAG=0.1
PARENT_DIRECTORY = $(shell pwd)

.PHONY: build run exec

build:
	docker build -t ${NAME}:${TAG} -f Dockerfile .

run:
	@if [ `docker container ls -a --filter "name=${NAME}" | wc -l | sed "s/ //g"` -eq 2 ]; then \
		docker container stop ${NAME}; \
		docker container rm ${NAME}; \
		docker container run \
			-it --privileged \
			--gpus all \
			-d \
			-v ${PARENT_DIRECTORY}:/app \
			-v /var/run/docker.sock:/var/run/docker.sock \
			--name ${NAME} ${NAME}:${TAG} \
			/bin/bash; \
	else \
		docker container run \
			-it --privileged \
			--gpus all \
			-d \
			-v ${PARENT_DIRECTORY}:/app \
			-v /var/run/docker.sock:/var/run/docker.sock \
			--name ${NAME} ${NAME}:${TAG} \
			/bin/bash; \
	fi
