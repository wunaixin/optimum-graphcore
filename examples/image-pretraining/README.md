How to Run mim with Graphcore IPU
1) download graphcore pytorch docker image for IPU
	docker pull graphcore/pytorch:3.1.0

2) start docker container with graphcore IPU device
	** 1: for IPU-POD
	docker run -it \
		 --network=host	\
		 --privileged 	\
		--ipc=host 	\
		-e IPUOF_VIPU_API_HOST=your_vipu_server_ip \
		-e IPUOF_VIPU_API_PARTITION_ID=partition_id \
		graphcore/pytorch:3.1.0 /bin/bash
	replace your_vipu_server_ip partition_id with your configuration
	you can get the your_vipu_server_ip and partition_id from command line
		cat /etc/vipu/vipu-cli.hcl
	  and   vipu-admin list partiton

	** 2: for C600 ( you need at least >=4 C600 device with IPU-Link connected)
	docker run -it \
		 --network=host	\
		 --privileged 	\
		--ipc=host 	\
		graphcore/pytorch:3.1.0 /bin/bash

3) Download optimum-graphcore code
	apt-get update && apt-get install vim git -y
	git clone http://github.com/wunaixin/optimum-graphcore

4) Run the example
	cd  optimum-graphcore/
	git checkout -b swin origin/swin
	pip install -e .
	cd examples/image-pretraining
	pip install -r requirements.txt
	bash run.sh
