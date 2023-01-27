# EdgeVision

**EdgeVision is an open source software framework for collaborative edge-cloud video analytics.**

## Install
-**System Requirements**
* [ubuntu 18.04](http://releases.ubuntu.com/18.04/)
* [Python 3.6.9](https://www.python.org/downloads/release/python-369/)
* [Jetpack 4.5](https://developer.nvidia.com/jetpack-sdk-45-archive)
* [cuda 10.2](https://developer.nvidia.com/cuda-toolkit)
* [pytorch 1.9.0](https://pytorch.org/)

-**Edge Node** 

Please install the following libraries on each edge node.
1. Install the corresponding version of [torch, torchvision](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048), opencv-python.
2. Install the following libraries.
```bash
pip3 install munch
pip3 install grpcio
pip3 install grpcio-tools
pip3 install loguru
pip3 install mysql-connector-python
pip3 install mapcalc
pip3 install APScheduler
pip3 install imutils
```

-**Cloud**

Similar to the installation on the edge node, install the corresponding version of [torch, torchvision](https://pytorch.org/get-started/previous-versions/) and required libraries.

-**Database**

Please install and configure the MySQL database.
```bash
sudo apt install mysql-server
```
## Usage

To be able to start the service for video analytics, please configure it step by step.

-**Step 1:** Modify the configuration file (config/config.yaml) as needed.
1. Video Source

If the video source is a video file, please configure the path of the video file.
```
video_path: your video path
```

If the video source is a network camera, please configure the account, password, and IP address.
```
rtsp:
 label: True
 account: your account
 password: your password
 ip_address: your camera ip
 channel: 1
```
2. IP configuration

Please configure the IP address of the cloud server.
```
server_ip: 'server ip:50051'
```

Please configure the number and IP addresses of edge nodes.
```
edge_id: the edge node ID
edge_num: the number of edge nodes
edges: ['edge ip 1:50051', 'edge ip 2:50051', ...]
```
3. Offloading policy

Please configure offloading policy. 
```
policy: Edge-Cloud-Assited
```

For example:
-Edge-Cloud-Assisted: The inference for a video frame will be first conducted with the small DNN on the local edge, and the regions of the video frame that have low recognition confidence below a threshold will be offloaded to the cloud for inference with the large DNN model. The Edge node will not directly offload inference requests to the cloud. 
-Edge-Cloud-Threshold: When the length of the local inference queue on the edge node exceeds a specified threshold, the edge node will directly offload the video frame to the cloud.

## Contributing

PRs accepted.

## License

MIT © Richard McRichface
