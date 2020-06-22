# Inherit from python 3.6 image
FROM python:3.6-slim

# Declare the workdir
WORKDIR /mltoolbox

# Update to the latest version of pip
RUN pip install --upgrade pip

# Install requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U py-cpuinfo
RUN apt update
RUN apt -y install emacs-nox
RUN apt -y install less
RUN apt -y install tk

# Install the package
COPY setup.py .
COPY mltoolbox/ mltoolbox
RUN pip install .
COPY main.py .

# Default command to run when running container
# We'll change this to actually run the analysis later
CMD ["python", "/mltoolbox/mltoolbox/main.py"]
