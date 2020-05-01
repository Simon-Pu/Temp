# nvidia-docker installation for Debian-based distributions

# https://nvidia.github.io/nvidia-docker/
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -

# https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
sudo apt-get install nvidia-docker2
sudo pkill -SIGHUP dockerd

docker run --runtime=nvidia