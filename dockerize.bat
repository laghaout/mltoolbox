# Clean up artifacts from emacs.
rm *~
rm learners/*~

# Format the code as per PEP8.
autopep8 --in-place --aggressive --aggressive ./*.py
autopep8 --in-place --aggressive --aggressive ./learners/*.py

# Stop and remove all containers.
docker stop $(docker container ls -q -a)
docker rm $(docker container ls -q -a)

# Remove all dangling images.
docker rmi $(docker images -f dangling=true -q)

# Generate the distribution.
pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel

# Install the package locally. [OPTIONAL]
#pip install dist/*

# Create a Docker image.
docker build --tag=learners .

# Run the main script.
#docker run -it learners python main.py $1
