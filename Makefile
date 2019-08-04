TAG=0.2.6
REGISTRY=registry.adaptive.finance
IMAGENAME=adaptivefinance/aft-amine-package
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
	docker run -it -v ${PWD}/AFT:/workdir/AFT ${IMAGE}:${TAG} 

bash: tag
	docker run -it -v ${PWD}/AFT:/workdir/AFT ${IMAGE}:${TAG} bash

test: tag
	docker run -v ${PWD}/tests/unit:/tests ${IMAGE}:${TAG} pytest /tests
