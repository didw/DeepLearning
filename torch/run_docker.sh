sudo docker run -ti -p 8888:8888 -v /home/didw/study/DeepLearning:/root/DeepLearning --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm --name cuitorch didw/cudaitorch:latest /bin/bash

