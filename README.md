# How to check NV CUDA and cuDNN version
1. cat /usr/local/cuda/version.txt
		If the script above doesn¡¦t work, try this:
		nvcc  --version
	
2. cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
 		or
	 cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2

3. To check GPU Card info, deep learner might use this all the time.
	nvidia-smi

# Speed Up your PyTorch models
https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051

# Download a file or a folder easily.
curl gdrive.sh | bash -s https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing

# Temp
Temp 
