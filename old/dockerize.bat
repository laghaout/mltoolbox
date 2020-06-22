# Stop and remove all containers
docker stop $(docker container ls -q -a)
docker rm $(docker container ls -q -a)

# Remove all dangling images
docker rmi $(docker images -f dangling=true -q)

# Generate the distribution
pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
#pip install dist/*

# Create a Docker image
docker build --tag=mltoolbox .

docker run -it mltoolbox python main.py $1
