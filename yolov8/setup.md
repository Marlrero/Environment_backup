# YOLOv8 setup
## driver, cuda remove
  ``` bash
  $ sudo apt --purge -y remove 'cuda*'
  $ sudo apt --purge -y remove 'nvidia*'
  $ sudo apt autoremove --purge cuda
  ```

## cudnn remove
  ``` bash
  $ cd /usr/local/
  $ sudo rm -rf cuda*
  ```
  - ~/.bashrc나 /etc/profile에 추가되어있는 CUDA 관련 설정도 제거
    ``` bash
    $ sudo vim ~/.bashrc
    $ sudo vim /etc/profile
    $ sudo source ~/.bashrc
    $ sudo source /etc/profile
    ```
    - vim 찾기 ESC -> /cuda
    ```
    export PATH=$PATH:/usr/local/cuda-11.0/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-{version}/lib64
    export CUDADIR=/usr/local/cuda-{version}
    ```

## Nvidia nouveau driver 블랙리스트 설정
  ``` bash
  $ sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
  $ sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
  $ cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
  ```
  - cat 결과
    ```
    blacklist nouveau
    options nouveau modeset=0
    ```
  ``` bash
  $ sudo init 6
  ```

## Kernel update
  ``` bash
  sudo update-initramfs -u
  ```

## Nvidia driver setup
  ``` bash
  $ sudo apt update
  $ ubuntu-drivers devices
  $ sudo apt install nvidia-driver-515
  $ sudo apt install dkms nvidia-modprobe
  $ sudo apt update
  $ sudo apt upgrade
  $ sudo init 6
  
  $ nvidia-smi
  ```

## CUDA down (https://developer.nvidia.com/cuda-toolkit-archive)
  ``` bash
  $ wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
  $ sudo sh cuda_11.7.1_515.65.01_linux.run
  ```
  - Continue -> accept -> CUDA만 X로 선택하고 Install (Driver는 설치 제외)

## CUDA 환경변수 추가
  ``` bash
  $ sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.7/bin'>> /etc/profile"
  $ sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64'>> /etc/profile"
  $ sudo sh -c "echo 'export CUDARDIR=/usr/local/cuda-11.7'>> /etc/profile"
  $ source /etc/profile
  
  $ nvcc -V
  ```

## cuDNN 설치(https://developer.nvidia.com/cudnn)
  - Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x
    ``` bash
    $ tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
    $ cd cudnn-linux-x86_64-8.6.0.163_cuda11-archive
    $ sudo cp include/cudnn* /usr/local/cuda/include
    $ sudo cp lib/libcudnn* /usr/local/cuda/lib64
    $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
    ```
  - Symbolic link setting
    ``` bash
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8   
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8   
    $ sudo ldconfig   
    $ ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn   
    ```
  -  결과
    ```
    libcudnn_ops_infer.so.8 -> libcudnn_ops_infer.so.8.5.0
    libcudnn_cnn_infer.so.8 -> libcudnn_cnn_infer.so.8.5.0
    libcudnn_cnn_train.so.8 -> libcudnn_cnn_train.so.8.5.0
    libcudnn_adv_infer.so.8 -> libcudnn_adv_infer.so.8.5.0
    libcudnn_adv_train.so.8 -> libcudnn_adv_train.so.8.5.0
    libcudnn.so.8 -> libcudnn.so.8.5.0
    libcudnn_ops_train.so.8 -> libcudnn_ops_train.so.8.5.0
    libcudnn_cnn_train.so.8 -> libcudnn_cnn_train.so.8.9.3
    libcudnn_adv_train.so.8 -> libcudnn_adv_train.so.8.9.3
    libcudnn_ops_infer.so.8 -> libcudnn_ops_infer.so.8.9.3
    libcudnn_cnn_infer.so.8 -> libcudnn_cnn_infer.so.8.9.3
    libcudnn_adv_infer.so.8 -> libcudnn_adv_infer.so.8.9.3
    libcudnn.so.8 -> libcudnn.so.8.9.3
    libcudnn_ops_train.so.8 -> libcudnn_ops_train.so.8.9.3
    ```

## pytorch 2.0.1 stable 설치 및 CUDA 연결 확인 (https://pytorch.org/get-started/locally/)
  ``` bash
  $ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  $ python
  ```
  ``` python
  import torch
  
  print(torch.cuda.is_available())
  # GPU 사용 가능 -> True, GPU 사용 불가 -> False

  exit()
  ```

## YOLO 설치 (v5, X, v8)
  ``` bash
  (conda_env) $ pip install ultralytics
  ```
