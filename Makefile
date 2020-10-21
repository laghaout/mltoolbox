TAG=0.0.0
AUTHOR=laghaout
REGISTRY=registry.learners
IMAGENAME=${AUTHOR}/learners
IMAGE=${REGISTRY}/${IMAGENAME}


tag:
	docker build -t ${IMAGE}:${TAG} .

latest:
	docker tag ${IMAGE}:${TAG} ${IMAGE}:latest 

build: tag latest

push-tag:
	docker push ${IMAGE}:${TAG}

push-latest:
	docker push ${IMAGE}:latest

push: push-tag push-latest

all: build push

run: tag
	docker run -it -v ${PWD}/${AUTHOR}:/learners/learners ${IMAGE}:${TAG} 

bash: tag
	docker run -it -v ${PWD}/${AUTHOR}:/learners/learners ${IMAGE}:${TAG} bash

test: tag
	docker run -v ${PWD}/tests/unit:/tests ${IMAGE}:${TAG} pytest /tests
