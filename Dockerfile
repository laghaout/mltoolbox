# Inherit from python 3.7 image
FROM python:3.7-slim

# Declare the workdir
WORKDIR /learners

# Update to the latest version of pip
RUN pip install --upgrade pip

# Install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt update
RUN apt -y install emacs-nox
RUN apt -y install less
RUN apt -y install tk
RUN apt -y install tree

# Install the package
COPY setup.py .
COPY learners/ learners
COPY data/ data
RUN pip install .
COPY main.py .

# Default command to run when running container
# We'll change this to actually run the analysis later
#CMD ["bash", ". <some launcher>.bat"]
