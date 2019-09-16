apt-get install -y libglib2.0-0
apt-get install -y libsm6
apt-get install -y libxrender1
apt-get install -y libxext-dev
apt-get install -y unzip
apt-get install -y nano
pip install --upgrade pip
pip install opencv-python

# install Detectron dependencies
pip install setuptools
pip install Cython
pip install matplotlib
pip install numpy
pip install pyyaml
pip install mock
pip install scipy
pip install six
pip install future
pip install protobuf

#Installing prebuilt DALI packages
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly/cuda/10.0 nvidia-dali-weekly
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/weekly/cuda/10.0 nvidia-dali-tf-plugin-weekly
#pip install -r requirements.txt