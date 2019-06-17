# install_cuda_docker.sh
# http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/
# Install CUDA
# wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
# 
cuda-repo=cuda-repo-ubuntu1604_10.1.168-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${cuda-repo}
sudo dpkg -i ${cuda-repo}
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt upgrade -y
sudo apt install cuda -y

#Install CuDNN
#https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/
#libcudnn7-dev_7.6.0.64-1+cuda9.2_amd64.deb
#wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.0.64-1+cuda9.2_amd64.deb
#libcudnn7-dev_7.6.0.64-1+cuda10.1_amd64.deb
#wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.0.64-1+cuda10.1_amd64.deb

# Install docker
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
sudo apt install docker-ce -y

# Install nvidia-docker2
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt install nvidia-docker2 -y
sudo pkill -SIGHUP dockerd
