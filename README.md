# Vision Transformer
Pytorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

![fig1](./img/figure1.png)

Vision Transformer achieve State-of-the-Art in image recognition task with standard Transformer encoder and fixed-size patches. In order to perform classification, author use the standard approach of adding an extra learnable "classification token" to the sequence.

![fig2](./img/figure2.png)


## Usage
### 1. Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

### 2. Train Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `--gradient_accumulation_steps`.

Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
```



## Results
To verify that the converted model weight is correct, we simply compare it with the author's experimental results. We trained using mixed precision, and `--fp16_opt_level` was set to O2.

### imagenet-21k
* [**tensorboard**](https://tensorboard.dev/experiment/Oz9GmmQIQCOEr4xbdr8O3Q)

|    model     |  dataset  | resolution | acc(official) | acc(this repo) |  time   |
|:------------:|:---------:|:----------:|:-------------:|:--------------:|:-------:|
|   ViT-B_16   | CIFAR-10  |  224x224   |       -       |     0.9908     | 3h 13m  |
|   ViT-B_16   | CIFAR-10  |  384x384   |    0.9903     |     0.9906     | 12h 25m |
|   ViT_B_16   | CIFAR-100 |  224x224   |       -       |     0.923      |  3h 9m  |
|   ViT_B_16   | CIFAR-100 |  384x384   |    0.9264     |     0.9228     | 12h 31m |
| R50-ViT-B_16 | CIFAR-10  |  224x224   |       -       |     0.9892     | 4h 23m  |
| R50-ViT-B_16 | CIFAR-10  |  384x384   |     0.99      |     0.9904     | 15h 40m |
| R50-ViT-B_16 | CIFAR-100 |  224x224   |       -       |     0.9231     | 4h 18m  |
| R50-ViT-B_16 | CIFAR-100 |  384x384   |    0.9231     |     0.9197     | 15h 53m |
|   ViT_L_32   | CIFAR-10  |  224x224   |       -       |     0.9903     | 2h 11m  |
|   ViT_L_32   | CIFAR-100 |  224x224   |       -       |     0.9276     |  2h 9m  |
|   ViT_H_14   | CIFAR-100 |  224x224   |       -       |      WIP       |         |


### imagenet-21k + imagenet2012
* [**tensorboard**](https://tensorboard.dev/experiment/CXOzjFRqTM6aLCk0jNXgAw/#scalars)

|    model     |  dataset  | resolution |  acc   |
|:------------:|:---------:|:----------:|:------:|
| ViT-B_16-224 | CIFAR-10  |  224x224   |  0.99  |
| ViT_B_16-224 | CIFAR-100 |  224x224   | 0.9245 |
|   ViT-L_32   | CIFAR-10  |  224x224   | 0.9903 |
|   ViT-L_32   | CIFAR-100 |  224x224   | 0.9285 |


### shorter train
* In the experiment below, we used a resolution size (224x224).
* [**tensorboard**](https://tensorboard.dev/experiment/lpknnMpHRT2qpVrSZi10Ag/#scalars)

|  upstream   |  model   |  dataset  | total_steps /warmup_steps | acc(official) | acc(this repo) |
|:-----------:|:--------:|:---------:|:-------------------------:|:-------------:|:--------------:|
| imagenet21k | ViT-B_16 | CIFAR-10  |          500/100          |    0.9859     |     0.9859     |
| imagenet21k | ViT-B_16 | CIFAR-10  |         1000/100          |    0.9886     |     0.9878     |
| imagenet21k | ViT-B_16 | CIFAR-100 |          500/100          |    0.8917     |     0.9072     |
| imagenet21k | ViT-B_16 | CIFAR-100 |         1000/100          |    0.9115     |     0.9216     |


## Visualization
The ViT consists of a Standard Transformer Encoder, and the encoder consists of Self-Attention and MLP module.
The attention map for the input image can be visualized through the attention score of self-attention.

Visualization code can be found at [visualize_attention_map](./visualize_attention_map.ipynb).

![fig3](./img/figure3.png)


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

# Changes

vscode ➜ /workspaces/ViT-pytorch (dev_container) $ apt list --installed | grep -i 12.2

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

dbus-user-session/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
dbus/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libcudnn8-dev/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libcudnn8/unknown,now 8.9.7.29-1+cuda12.2 amd64 [installed]
libdbus-1-3/jammy-updates,jammy-security,now 1.12.20-2ubuntu4.1 amd64 [installed,automatic]
libnode72/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed,automatic]
libnvjpeg-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
libnvjpeg-dev-12-1/unknown,now 12.2.0.2-1 amd64 [installed,automatic]
nodejs-doc/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 all [installed,automatic]
nodejs/jammy-updates,jammy-security,now 12.22.9~dfsg-1ubuntu3.3 amd64 [installed]
vscode ➜ /workspaces/ViT-pytorch (dev_container) $ 

Running the updateContentCommand from devcontainer.json...

[35760 ms] Start: Run in container: /bin/sh -c bash .devcontainer/install-dev-tools.sh
Get:1 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]
0% [Waiting for headers] [1 InRelease 14.2 kB/110 kB 13%] [Connecting to developer.download.nvidia.com (152.199.20.126)] [Connecting to pac                                                                                                                                           Get:2 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]
0% [2 InRelease 14.2 kB/270 kB 5%] [1 InRelease 51.8 kB/110 kB 47%] [Connected to developer.download.nvidia.com (152.199.20.126)] [Connecte                                                                                                                                           Get:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease [1581 B]   
Get:4 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [44.0 kB]                         
Get:5 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1047 kB]                                       
Get:6 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages [1323 kB]                                   
0% [2 InRelease 75.0 kB/270 kB 28%] [5 Packages 63.5 kB/1047 kB 6%] [6 Packages 32.8 kB/1323 kB 2%] [Connected to packagecloud.io (54.215.1                                                                                                                                           Get:7 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1582 kB]                                            
0% [2 InRelease 159 kB/270 kB 59%] [7 Packages 20.5 kB/1582 kB 1%] [6 Packages 115 kB/1323 kB 9%] [Connected to packagecloud.io (54.215.1680% [5 Packages store 0 B] [2 InRelease 159 kB/270 kB 59%] [7 Packages 20.5 kB/1582 kB 1%] [6 Packages 115 kB/1323 kB 9%] [Connected to pack                                                                                                                                           Get:8 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [1332 kB]                                                 
0% [7 Packages store 0 B] [2 InRelease 211 kB/270 kB 78%] [8 Packages 8477 B/1332 kB 1%] [6 Packages 541 kB/1323 kB 41%] [Waiting for heade                                                                                                                                           Get:9 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [119 kB]                                        
Get:11 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [109 kB]             
Get:12 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]                   
Get:13 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB]
Get:10 https://packagecloud.io/github/git-lfs/ubuntu jammy InRelease [28.0 kB]  
Get:15 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]  
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [1611 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [1606 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1307 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [49.8 kB]
Get:21 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [28.1 kB]
Get:22 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [50.4 kB]
Get:14 https://packagecloud.io/github/git-lfs/ubuntu jammy/main amd64 Packages [1654 B]
Fetched 30.3 MB in 3s (11.2 MB/s)                           
Reading package lists... Done
.devcontainer/install-dev-tools.sh: line 8: sud: command not found
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Calculating upgrade... Done
The following packages will be upgraded:
  binutils binutils-common binutils-x86-64-linux-gnu cuda-keyring distro-info-data libbinutils libctf-nobfd0 libctf0 libcudnn8 libcudnn8-dev libssh-4
  openssh-client vim-common vim-tiny xxd
15 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
Need to get 890 MB of archives.
After this operation, 73.3 MB disk space will be freed.
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 distro-info-data all 0.52ubuntu0.6 [5160 B]
Get:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  cuda-keyring 1.1-1 [4328 B]
Get:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libcudnn8-dev 8.9.7.29-1+cuda12.2 [440 MB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-tiny amd64 2:8.2.3995-1ubuntu2.15 [710 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 xxd amd64 2:8.2.3995-1ubuntu2.15 [55.2 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-common all 2:8.2.3995-1ubuntu2.15 [81.5 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 openssh-client amd64 1:8.9p1-3ubuntu0.6 [906 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf0 amd64 2.38-4ubuntu2.4 [103 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf-nobfd0 amd64 2.38-4ubuntu2.4 [108 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-x86-64-linux-gnu amd64 2.38-4ubuntu2.4 [2327 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libbinutils amd64 2.38-4ubuntu2.4 [662 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils amd64 2.38-4ubuntu2.4 [3194 B]
Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-common amd64 2.38-4ubuntu2.4 [222 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libssh-4 amd64 0.9.6-2ubuntu0.22.04.2 [186 kB]
57% [3 libcudnn8-dev 431 MB/440 MB 98%]                                                                                                   759% [Waiting for headers]                                                                                                                 7                                                                                                                                           Get:15 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  libcudnn8 8.9.7.29-1+cuda12.2 [444 MB]
59% [15 libcudnn8 53.5 kB/444 MB 0%]                                                                                                      762% [15 libcudnn8 39.5 MB/444 MB 9%]                                                                                                      766% [15 libcudnn8 76.6 MB/444 MB 17%]                                                                                                     769% [15 libcudnn8 115 MB/444 MB 26%]                                                                                                      772% [15 libcudnn8 152 MB/444 MB 34%]                                                                                                      776% [15 libcudnn8 188 MB/444 MB 42%]                                                                                                      779% [15 libcudnn8 228 MB/444 MB 51%]                                                                                                      783% [15 libcudnn8 267 MB/444 MB 60%]                                                                                                      786% [15 libcudnn8 302 MB/444 MB 68%]                                                                                                      789% [15 libcudnn8 341 MB/444 MB 77%]                                                                                                      792% [15 libcudnn8 374 MB/444 MB 84%]                                                                                                      795% [15 libcudnn8 403 MB/444 MB 91%]                                                                                                      799% [15 libcudnn8 443 MB/444 MB 100%]                                                                                                     7100% [Working]                                                                                                                            7                                                                                                                                           Fetched 890 MB in 13s (69.3 MB/s)
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-distro-info-data_0.52ubuntu0.6_all.deb ...
Unpacking distro-info-data (0.52ubuntu0.6) over (0.52ubuntu0.5) ...
Preparing to unpack .../01-vim-tiny_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking vim-tiny (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../02-xxd_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking xxd (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../03-vim-common_2%3a8.2.3995-1ubuntu2.15_all.deb ...
Unpacking vim-common (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../04-openssh-client_1%3a8.9p1-3ubuntu0.6_amd64.deb ...
Unpacking openssh-client (1:8.9p1-3ubuntu0.6) over (1:8.9p1-3ubuntu0.4) ...
Preparing to unpack .../05-libctf0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../06-libctf-nobfd0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf-nobfd0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../07-binutils-x86-64-linux-gnu_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../08-libbinutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libbinutils:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../09-binutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../10-binutils-common_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-common:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../11-cuda-keyring_1.1-1_all.deb ...
Unpacking cuda-keyring (1.1-1) over (1.0-1) ...
Preparing to unpack .../12-libcudnn8-dev_8.9.7.29-1+cuda12.2_amd64.deb ...
update-alternatives: removing manually selected alternative - switching libcudnn to auto mode
Unpacking libcudnn8-dev (8.9.7.29-1+cuda12.2) over (8.9.0.131-1+cuda12.1) ...
Preparing to unpack .../13-libcudnn8_8.9.7.29-1+cuda12.2_amd64.deb ...
Unpacking libcudnn8 (8.9.7.29-1+cuda12.2) over (8.9.0.131-1+cuda12.1) ...
Preparing to unpack .../14-libssh-4_0.9.6-2ubuntu0.22.04.2_amd64.deb ...
Unpacking libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) over (0.9.6-2ubuntu0.22.04.1) ...
Setting up distro-info-data (0.52ubuntu0.6) ...
Setting up openssh-client (1:8.9p1-3ubuntu0.6) ...
Setting up libcudnn8 (8.9.7.29-1+cuda12.2) ...
Setting up binutils-common:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf-nobfd0:amd64 (2.38-4ubuntu2.4) ...
Setting up xxd (2:8.2.3995-1ubuntu2.15) ...
Setting up vim-common (2:8.2.3995-1ubuntu2.15) ...
Setting up cuda-keyring (1.1-1) ...
Setting up libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) ...
Setting up libcudnn8-dev (8.9.7.29-1+cuda12.2) ...
update-alternatives: using /usr/include/x86_64-linux-gnu/cudnn_v8.h to provide /usr/include/cudnn.h (libcudnn) in auto mode
Setting up libbinutils:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf0:amd64 (2.38-4ubuntu2.4) ...
Setting up vim-tiny (2:8.2.3995-1ubuntu2.15) ...
Setting up binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) ...
Setting up binutils (2.38-4ubuntu2.4) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
pkg-config is already the newest version (0.29.2-1ubuntu3).
pkg-config set to manually installed.
python3-dev is already the newest version (3.10.6-1~22.04).
The following additional packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2 libmount-dev
  libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev libxcb-shm0-dev
Suggested packages:
  libcairo2-doc libgirepository1.0-dev libglib2.0-doc libgdk-pixbuf2.0-bin | libgdk-pixbuf2.0-dev libxml2-utils libice-doc libsm-doc
The following NEW packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libcairo2-dev libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2
  libmount-dev libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev
  libxcb-shm0-dev
0 upgraded, 20 newly installed, 0 to remove and 0 not upgraded.
Need to get 4790 kB of archives.
After this operation, 23.4 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-gobject2 amd64 1.16.0-5ubuntu2 [19.4 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 liblzo2-2 amd64 2.10-2build3 [53.7 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-script-interpreter2 amd64 1.16.0-5ubuntu2 [62.0 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy/main amd64 libice-dev amd64 2:1.0.10-1build2 [51.4 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsm-dev amd64 2:1.2.3-1build2 [18.1 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpixman-1-dev amd64 0.40.0-1ubuntu0.22.04.1 [280 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-render0-dev amd64 1.14-3ubuntu3 [19.6 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-shm0-dev amd64 1.14-3ubuntu3 [6848 B]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-bin amd64 2.72.4-0ubuntu2.2 [80.9 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev-bin amd64 2.72.4-0ubuntu2.2 [117 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy/main amd64 libblkid-dev amd64 2.37.2-4ubuntu3 [185 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsepol-dev amd64 3.3-1build1 [378 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/main amd64 libselinux1-dev amd64 3.3-1build2 [158 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmount-dev amd64 2.37.2-4ubuntu3 [14.5 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre16-3 amd64 2:8.39-13ubuntu0.22.04.1 [164 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre32-3 amd64 2:8.39-13ubuntu0.22.04.1 [155 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcrecpp0v5 amd64 2:8.39-13ubuntu0.22.04.1 [16.5 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre3-dev amd64 2:8.39-13ubuntu0.22.04.1 [579 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev amd64 2.72.4-0ubuntu2.2 [1739 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo2-dev amd64 1.16.0-5ubuntu2 [692 kB]
Fetched 4790 kB in 1s (3384 kB/s)      
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Selecting previously unselected package libcairo-gobject2:amd64.
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-libcairo-gobject2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package liblzo2-2:amd64.
Preparing to unpack .../01-liblzo2-2_2.10-2build3_amd64.deb ...
Unpacking liblzo2-2:amd64 (2.10-2build3) ...
Selecting previously unselected package libcairo-script-interpreter2:amd64.
Preparing to unpack .../02-libcairo-script-interpreter2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package libice-dev:amd64.
Preparing to unpack .../03-libice-dev_2%3a1.0.10-1build2_amd64.deb ...
Unpacking libice-dev:amd64 (2:1.0.10-1build2) ...
Selecting previously unselected package libsm-dev:amd64.
Preparing to unpack .../04-libsm-dev_2%3a1.2.3-1build2_amd64.deb ...
Unpacking libsm-dev:amd64 (2:1.2.3-1build2) ...
Selecting previously unselected package libpixman-1-dev:amd64.
Preparing to unpack .../05-libpixman-1-dev_0.40.0-1ubuntu0.22.04.1_amd64.deb ...
Unpacking libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Selecting previously unselected package libxcb-render0-dev:amd64.
Preparing to unpack .../06-libxcb-render0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libxcb-shm0-dev:amd64.
Preparing to unpack .../07-libxcb-shm0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libglib2.0-bin.
Preparing to unpack .../08-libglib2.0-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libglib2.0-dev-bin.
Preparing to unpack .../09-libglib2.0-dev-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libblkid-dev:amd64.
Preparing to unpack .../10-libblkid-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libsepol-dev:amd64.
Preparing to unpack .../11-libsepol-dev_3.3-1build1_amd64.deb ...
Unpacking libsepol-dev:amd64 (3.3-1build1) ...
Selecting previously unselected package libselinux1-dev:amd64.
Preparing to unpack .../12-libselinux1-dev_3.3-1build2_amd64.deb ...
Unpacking libselinux1-dev:amd64 (3.3-1build2) ...
Selecting previously unselected package libmount-dev:amd64.
Preparing to unpack .../13-libmount-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libpcre16-3:amd64.
Preparing to unpack .../14-libpcre16-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre32-3:amd64.
Preparing to unpack .../15-libpcre32-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcrecpp0v5:amd64.
Preparing to unpack .../16-libpcrecpp0v5_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre3-dev:amd64.
Preparing to unpack .../17-libpcre3-dev_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libglib2.0-dev:amd64.
Preparing to unpack .../18-libglib2.0-dev_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libcairo2-dev:amd64.
Preparing to unpack .../19-libcairo2-dev_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
Setting up libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Setting up libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Setting up libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libice-dev:amd64 (2:1.0.10-1build2) ...
Setting up libsm-dev:amd64 (2:1.2.3-1build2) ...
Setting up libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Setting up liblzo2-2:amd64 (2.10-2build3) ...
Setting up libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libsepol-dev:amd64 (3.3-1build1) ...
Setting up libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Setting up libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Setting up libselinux1-dev:amd64 (3.3-1build2) ...
Setting up libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libglib2.0-0:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Setting up libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  javascript-common libc-ares2 libjs-highlight.js libnode72 libuv1 nodejs-doc
Suggested packages:
  apache2 | lighttpd | httpd npm
The following NEW packages will be installed:
  javascript-common libc-ares2 libjs-highlight.js libnode72 libuv1 nodejs nodejs-doc
0 upgraded, 7 newly installed, 0 to remove and 0 not upgraded.
Need to get 13.8 MB of archives.
After this operation, 54.2 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libuv1 amd64 1.43.0-1 [93.1 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 javascript-common all 11+nmu1 [5936 B]
Get:3 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libjs-highlight.js all 9.18.5+dfsg1-1 [367 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc-ares2 amd64 1.18.1-1ubuntu0.22.04.2 [45.0 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 libnode72 amd64 12.22.9~dfsg-1ubuntu3.3 [10.8 MB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 nodejs-doc all 12.22.9~dfsg-1ubuntu3.3 [2410 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 nodejs amd64 12.22.9~dfsg-1ubuntu3.3 [122 kB]
Fetched 13.8 MB in 1s (21.7 MB/s)
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.3.2)
Defaulting to user installation because normal site-packages is not writeable
Collecting torch (from -r requirements.txt (line 1))
  Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl.metadata (25 kB)
Collecting numpy (from -r requirements.txt (line 2))
  Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 2.5 MB/s eta 0:00:00
Collecting tqdm (from -r requirements.txt (line 3))
  Downloading tqdm-4.66.1-py3-none-any.whl.metadata (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.6/57.6 kB 6.8 MB/s eta 0:00:00
Collecting tensorboard (from -r requirements.txt (line 4))
  Downloading tensorboard-2.15.1-py3-none-any.whl.metadata (1.7 kB)
Collecting ml-collections (from -r requirements.txt (line 5))
  Downloading ml_collections-0.1.1.tar.gz (77 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 6.6 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting filelock (from torch->-r requirements.txt (line 1))
  Downloading filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
Collecting typing-extensions (from torch->-r requirements.txt (line 1))
  Downloading typing_extensions-4.9.0-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy (from torch->-r requirements.txt (line 1))
  Downloading sympy-1.12-py3-none-any.whl (5.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.7/5.7 MB 20.5 MB/s eta 0:00:00
Collecting networkx (from torch->-r requirements.txt (line 1))
  Downloading networkx-3.2.1-py3-none-any.whl.metadata (5.2 kB)
Collecting jinja2 (from torch->-r requirements.txt (line 1))
  Downloading Jinja2-3.1.2-py3-none-any.whl (133 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 15.1 MB/s eta 0:00:00
Collecting fsspec (from torch->-r requirements.txt (line 1))
  Downloading fsspec-2023.12.2-py3-none-any.whl.metadata (6.8 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 96.6 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 8.2 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 145.3 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.1.3.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 19.1 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.0.2.54 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 49.5 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.2.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 71.2 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 50.3 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 35.3 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.18.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.8/209.8 MB 30.5 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 10.4 MB/s eta 0:00:00
Collecting triton==2.1.0 (from torch->-r requirements.txt (line 1))
  Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.3 kB)
Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch->-r requirements.txt (line 1))
  Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting absl-py>=0.4 (from tensorboard->-r requirements.txt (line 4))
  Downloading absl_py-2.0.0-py3-none-any.whl.metadata (2.3 kB)
Collecting grpcio>=1.48.2 (from tensorboard->-r requirements.txt (line 4))
  Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting google-auth<3,>=1.6.3 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth-2.26.1-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<2,>=0.5 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting markdown>=2.6.8 (from tensorboard->-r requirements.txt (line 4))
  Downloading Markdown-3.5.1-py3-none-any.whl.metadata (7.1 kB)
Collecting protobuf<4.24,>=3.19.6 (from tensorboard->-r requirements.txt (line 4))
  Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl.metadata (540 bytes)
Collecting requests<3,>=2.21.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools>=41.0.0 in /usr/lib/python3/dist-packages (from tensorboard->-r requirements.txt (line 4)) (59.6.0)
Collecting six>1.9 (from tensorboard->-r requirements.txt (line 4))
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard->-r requirements.txt (line 4))
  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
Collecting PyYAML (from ml-collections->-r requirements.txt (line 5))
  Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
Collecting contextlib2 (from ml-collections->-r requirements.txt (line 5))
  Downloading contextlib2-21.6.0-py2.py3-none-any.whl (13 kB)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading cachetools-5.3.2-py3-none-any.whl.metadata (5.2 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 20.0 MB/s eta 0:00:00
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading rsa-4.9-py3-none-any.whl (34 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading idna-3.6-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading urllib3-2.1.0-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading certifi-2023.11.17-py3-none-any.whl.metadata (2.2 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard->-r requirements.txt (line 4))
  Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
Collecting mpmath>=0.19 (from sympy->torch->-r requirements.txt (line 1))
  Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 45.8 MB/s eta 0:00:00
Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1-0.5.1-py2.py3-none-any.whl.metadata (8.6 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151.7/151.7 kB 17.5 MB/s eta 0:00:00
Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl (670.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 670.2/670.2 MB 11.8 MB/s eta 0:00:00
Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 10.8 MB/s eta 0:00:00
Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.2/89.2 MB 51.0 MB/s eta 0:00:00
Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 128.5 MB/s eta 0:00:00
Downloading tqdm-4.66.1-py3-none-any.whl (78 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.3/78.3 kB 9.6 MB/s eta 0:00:00
Downloading tensorboard-2.15.1-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 43.8 MB/s eta 0:00:00
Downloading absl_py-2.0.0-py3-none-any.whl (130 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 18.4 MB/s eta 0:00:00
Downloading google_auth-2.26.1-py2.py3-none-any.whl (186 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 186.4/186.4 kB 18.5 MB/s eta 0:00:00
Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl (24 kB)
Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 139.3 MB/s eta 0:00:00
Downloading Markdown-3.5.1-py3-none-any.whl (102 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 102.2/102.2 kB 9.2 MB/s eta 0:00:00
Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl (304 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 304.5/304.5 kB 30.6 MB/s eta 0:00:00
Downloading requests-2.31.0-py3-none-any.whl (62 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 10.9 MB/s eta 0:00:00
Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 155.1 MB/s eta 0:00:00
Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 21.5 MB/s eta 0:00:00
Downloading filelock-3.13.1-py3-none-any.whl (11 kB)
Downloading fsspec-2023.12.2-py3-none-any.whl (168 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 169.0/169.0 kB 18.4 MB/s eta 0:00:00
Downloading networkx-3.2.1-py3-none-any.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 89.0 MB/s eta 0:00:00
Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 705.5/705.5 kB 56.3 MB/s eta 0:00:00
Downloading typing_extensions-4.9.0-py3-none-any.whl (32 kB)
Downloading cachetools-5.3.2-py3-none-any.whl (9.3 kB)
Downloading certifi-2023.11.17-py3-none-any.whl (162 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.5/162.5 kB 18.4 MB/s eta 0:00:00
Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 15.0 MB/s eta 0:00:00
Downloading idna-3.6-py3-none-any.whl (61 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 kB 6.4 MB/s eta 0:00:00
Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Downloading urllib3-2.1.0-py3-none-any.whl (104 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.6/104.6 kB 13.1 MB/s eta 0:00:00
Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl (20.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.5/20.5 MB 118.6 MB/s eta 0:00:00
Downloading pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 10.6 MB/s eta 0:00:00
Building wheels for collected packages: ml-collections
  Building wheel for ml-collections (setup.py) ... done
  Created wheel for ml-collections: filename=ml_collections-0.1.1-py3-none-any.whl size=94522 sha256=46a9973cdc674b7db6f6a765cee47fadd50bbe79ff1975633638cc94e42a4b88
  Stored in directory: /home/vscode/.cache/pip/wheels/7b/89/c9/a9b87790789e94aadcfc393c283e3ecd5ab916aed0a31be8fe
Successfully built ml-collections
Installing collected packages: mpmath, urllib3, typing-extensions, tqdm, tensorboard-data-server, sympy, six, PyYAML, pyasn1, protobuf, oauthlib, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, markdown, idna, grpcio, fsspec, filelock, contextlib2, charset-normalizer, certifi, cachetools, absl-py, werkzeug, triton, rsa, requests, pyasn1-modules, nvidia-cusparse-cu12, nvidia-cudnn-cu12, ml-collections, jinja2, requests-oauthlib, nvidia-cusolver-cu12, google-auth, torch, google-auth-oauthlib, tensorboard
  WARNING: The script tqdm is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script isympy is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script f2py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown_py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script normalizer is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts pyrsa-decrypt, pyrsa-encrypt, pyrsa-keygen, pyrsa-priv2pub, pyrsa-sign and pyrsa-verify are installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts convert-caffe2-to-onnx, convert-onnx-to-caffe2 and torchrun are installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script google-oauthlib-tool is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script tensorboard is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed MarkupSafe-2.1.3 PyYAML-6.0.1 absl-py-2.0.0 cachetools-5.3.2 certifi-2023.11.17 charset-normalizer-3.3.2 contextlib2-21.6.0 filelock-3.13.1 fsspec-2023.12.2 google-auth-2.26.1 google-auth-oauthlib-1.2.0 grpcio-1.60.0 idna-3.6 jinja2-3.1.2 markdown-3.5.1 ml-collections-0.1.1 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.3 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.3.101 nvidia-nvtx-cu12-12.1.105 oauthlib-3.2.2 protobuf-4.23.4 pyasn1-0.5.1 pyasn1-modules-0.3.0 requests-2.31.0 requests-oauthlib-1.3.1 rsa-4.9 six-1.16.0 sympy-1.12 tensorboard-2.15.1 tensorboard-data-server-0.7.2 torch-2.1.2 tqdm-4.66.1 triton-2.1.0 typing-extensions-4.9.0 urllib3-2.1.0 werkzeug-3.0.1
Running the postCreateCommand from Feature 'ghcr.io/devcontainers/features/git-lfs:1'...

[183136 ms] Start: Run in container: /bin/sh -c /usr/local/share/pull-git-lfs-artifacts.sh
Fetching git lfs artifacts...
Running the postCreateCommand from devcontainer.json...

[183263 ms] Start: Run in container: nvidia-smi
Fri Jan  5 19:06:21 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On  | 00000000:00:06.0 Off |                  N/A |
| 30%   29C    P8              24W / 350W |      3MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Done. Press any key to close the terminal.



Running the updateContentCommand from devcontainer.json...

[31789 ms] Start: Run in container: /bin/sh -c bash .devcontainer/install-dev-tools.sh
Get:1 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]
0% [Connecting to archive.ubuntu.com (91.189.91.82)] [1 InRelease 14.2 kB/110 kB 13%] [Connected to developer.download.nvidia.com (152.199.                                                                                                                                           Get:2 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]                              
Get:3 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease [1581 B]   
Get:4 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [44.0 kB]                        
Get:5 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1047 kB]                                       
Get:6 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages [1323 kB]                                   
0% [2 InRelease 51.8 kB/270 kB 19%] [5 Packages 105 kB/1047 kB 10%] [6 Packages 0 B/1323 kB 0%] [Connected to packagecloud.io (54.215.168.3                                                                                                                                           Get:7 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1582 kB]                                             
0% [2 InRelease 121 kB/270 kB 45%] [7 Packages 8440 B/1582 kB 1%] [6 Packages 295 kB/1323 kB 22%] [Connected to packagecloud.io (54.215.1680% [5 Packages store 0 B] [2 InRelease 121 kB/270 kB 45%] [7 Packages 9888 B/1582 kB 1%] [6 Packages 295 kB/1323 kB 22%] [Connected to pack                                                                                                                                           0% [2 InRelease 159 kB/270 kB 59%] [7 Packages 1009 kB/1582 kB 64%] [6 Packages 606 kB/1323 kB 46%] [Connected to packagecloud.io (54.215.1                                                                                                                                           Get:8 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [1332 kB]                                                   
0% [2 InRelease 173 kB/270 kB 64%] [8 Packages 9057 B/1332 kB 1%] [6 Packages 606 kB/1323 kB 46%] [Connected to packagecloud.io (54.215.1680% [7 Packages store 0 B] [2 InRelease 173 kB/270 kB 64%] [8 Packages 36.6 kB/1332 kB 3%] [6 Packages 606 kB/1323 kB 46%] [Connected to pac                                                                                                                                           Get:9 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [119 kB]                                         
Get:11 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [109 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB]
Get:10 https://packagecloud.io/github/git-lfs/ubuntu jammy InRelease [28.0 kB]  
Get:15 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [49.8 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [1611 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [1606 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1307 kB]
Get:21 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [28.1 kB]
Get:22 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [50.4 kB]
Get:16 https://packagecloud.io/github/git-lfs/ubuntu jammy/main amd64 Packages [1654 B]
Fetched 30.3 MB in 3s (11.2 MB/s)                           
Reading package lists... Done
cuda-toolkit set on hold.
libcudnn8-dev set on hold.
libcudnn8 set on hold.
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Calculating upgrade... Done
The following packages have been kept back:
  libcudnn8 libcudnn8-dev
The following packages will be upgraded:
  binutils binutils-common binutils-x86-64-linux-gnu cuda-keyring distro-info-data libbinutils libctf-nobfd0 libctf0 libssh-4 openssh-client
  vim-common vim-tiny xxd
13 upgraded, 0 newly installed, 0 to remove and 2 not upgraded.
Need to get 5372 kB of archives.
After this operation, 5120 B of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 distro-info-data all 0.52ubuntu0.6 [5160 B]
Get:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  cuda-keyring 1.1-1 [4328 B]
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-tiny amd64 2:8.2.3995-1ubuntu2.15 [710 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 xxd amd64 2:8.2.3995-1ubuntu2.15 [55.2 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-common all 2:8.2.3995-1ubuntu2.15 [81.5 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 openssh-client amd64 1:8.9p1-3ubuntu0.6 [906 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf0 amd64 2.38-4ubuntu2.4 [103 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf-nobfd0 amd64 2.38-4ubuntu2.4 [108 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-x86-64-linux-gnu amd64 2.38-4ubuntu2.4 [2327 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libbinutils amd64 2.38-4ubuntu2.4 [662 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils amd64 2.38-4ubuntu2.4 [3194 B]
Get:12 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-common amd64 2.38-4ubuntu2.4 [222 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libssh-4 amd64 0.9.6-2ubuntu0.22.04.2 [186 kB]
Fetched 5372 kB in 1s (4731 kB/s)
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-distro-info-data_0.52ubuntu0.6_all.deb ...
Unpacking distro-info-data (0.52ubuntu0.6) over (0.52ubuntu0.5) ...
Preparing to unpack .../01-vim-tiny_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking vim-tiny (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../02-xxd_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking xxd (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../03-vim-common_2%3a8.2.3995-1ubuntu2.15_all.deb ...
Unpacking vim-common (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../04-openssh-client_1%3a8.9p1-3ubuntu0.6_amd64.deb ...
Unpacking openssh-client (1:8.9p1-3ubuntu0.6) over (1:8.9p1-3ubuntu0.4) ...
Preparing to unpack .../05-libctf0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../06-libctf-nobfd0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf-nobfd0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../07-binutils-x86-64-linux-gnu_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../08-libbinutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libbinutils:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../09-binutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../10-binutils-common_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-common:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../11-cuda-keyring_1.1-1_all.deb ...
Unpacking cuda-keyring (1.1-1) over (1.0-1) ...
Preparing to unpack .../12-libssh-4_0.9.6-2ubuntu0.22.04.2_amd64.deb ...
Unpacking libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) over (0.9.6-2ubuntu0.22.04.1) ...
Setting up distro-info-data (0.52ubuntu0.6) ...
Setting up openssh-client (1:8.9p1-3ubuntu0.6) ...
Setting up binutils-common:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf-nobfd0:amd64 (2.38-4ubuntu2.4) ...
Setting up xxd (2:8.2.3995-1ubuntu2.15) ...
Setting up vim-common (2:8.2.3995-1ubuntu2.15) ...
Setting up cuda-keyring (1.1-1) ...
Setting up libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) ...
Setting up libbinutils:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf0:amd64 (2.38-4ubuntu2.4) ...
Setting up vim-tiny (2:8.2.3995-1ubuntu2.15) ...
Setting up binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) ...
Setting up binutils (2.38-4ubuntu2.4) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
pkg-config is already the newest version (0.29.2-1ubuntu3).
pkg-config set to manually installed.
python3-dev is already the newest version (3.10.6-1~22.04).
The following additional packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2 libmount-dev
  libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev libxcb-shm0-dev
Suggested packages:
  libcairo2-doc libgirepository1.0-dev libglib2.0-doc libgdk-pixbuf2.0-bin | libgdk-pixbuf2.0-dev libxml2-utils libice-doc libsm-doc
The following NEW packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libcairo2-dev libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2
  libmount-dev libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev
  libxcb-shm0-dev
0 upgraded, 20 newly installed, 0 to remove and 2 not upgraded.
Need to get 4790 kB of archives.
After this operation, 23.4 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-gobject2 amd64 1.16.0-5ubuntu2 [19.4 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 liblzo2-2 amd64 2.10-2build3 [53.7 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-script-interpreter2 amd64 1.16.0-5ubuntu2 [62.0 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy/main amd64 libice-dev amd64 2:1.0.10-1build2 [51.4 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsm-dev amd64 2:1.2.3-1build2 [18.1 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpixman-1-dev amd64 0.40.0-1ubuntu0.22.04.1 [280 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-render0-dev amd64 1.14-3ubuntu3 [19.6 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-shm0-dev amd64 1.14-3ubuntu3 [6848 B]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-bin amd64 2.72.4-0ubuntu2.2 [80.9 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev-bin amd64 2.72.4-0ubuntu2.2 [117 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy/main amd64 libblkid-dev amd64 2.37.2-4ubuntu3 [185 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsepol-dev amd64 3.3-1build1 [378 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/main amd64 libselinux1-dev amd64 3.3-1build2 [158 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmount-dev amd64 2.37.2-4ubuntu3 [14.5 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre16-3 amd64 2:8.39-13ubuntu0.22.04.1 [164 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre32-3 amd64 2:8.39-13ubuntu0.22.04.1 [155 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcrecpp0v5 amd64 2:8.39-13ubuntu0.22.04.1 [16.5 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre3-dev amd64 2:8.39-13ubuntu0.22.04.1 [579 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev amd64 2.72.4-0ubuntu2.2 [1739 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo2-dev amd64 1.16.0-5ubuntu2 [692 kB]
Fetched 4790 kB in 1s (5531 kB/s)     
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Selecting previously unselected package libcairo-gobject2:amd64.
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-libcairo-gobject2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package liblzo2-2:amd64.
Preparing to unpack .../01-liblzo2-2_2.10-2build3_amd64.deb ...
Unpacking liblzo2-2:amd64 (2.10-2build3) ...
Selecting previously unselected package libcairo-script-interpreter2:amd64.
Preparing to unpack .../02-libcairo-script-interpreter2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package libice-dev:amd64.
Preparing to unpack .../03-libice-dev_2%3a1.0.10-1build2_amd64.deb ...
Unpacking libice-dev:amd64 (2:1.0.10-1build2) ...
Selecting previously unselected package libsm-dev:amd64.
Preparing to unpack .../04-libsm-dev_2%3a1.2.3-1build2_amd64.deb ...
Unpacking libsm-dev:amd64 (2:1.2.3-1build2) ...
Selecting previously unselected package libpixman-1-dev:amd64.
Preparing to unpack .../05-libpixman-1-dev_0.40.0-1ubuntu0.22.04.1_amd64.deb ...
Unpacking libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Selecting previously unselected package libxcb-render0-dev:amd64.
Preparing to unpack .../06-libxcb-render0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libxcb-shm0-dev:amd64.
Preparing to unpack .../07-libxcb-shm0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libglib2.0-bin.
Preparing to unpack .../08-libglib2.0-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libglib2.0-dev-bin.
Preparing to unpack .../09-libglib2.0-dev-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libblkid-dev:amd64.
Preparing to unpack .../10-libblkid-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libsepol-dev:amd64.
Preparing to unpack .../11-libsepol-dev_3.3-1build1_amd64.deb ...
Unpacking libsepol-dev:amd64 (3.3-1build1) ...
Selecting previously unselected package libselinux1-dev:amd64.
Preparing to unpack .../12-libselinux1-dev_3.3-1build2_amd64.deb ...
Unpacking libselinux1-dev:amd64 (3.3-1build2) ...
Selecting previously unselected package libmount-dev:amd64.
Preparing to unpack .../13-libmount-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libpcre16-3:amd64.
Preparing to unpack .../14-libpcre16-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre32-3:amd64.
Preparing to unpack .../15-libpcre32-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcrecpp0v5:amd64.
Preparing to unpack .../16-libpcrecpp0v5_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre3-dev:amd64.
Preparing to unpack .../17-libpcre3-dev_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libglib2.0-dev:amd64.
Preparing to unpack .../18-libglib2.0-dev_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libcairo2-dev:amd64.
Preparing to unpack .../19-libcairo2-dev_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
Setting up libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Setting up libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Setting up libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libice-dev:amd64 (2:1.0.10-1build2) ...
Setting up libsm-dev:amd64 (2:1.2.3-1build2) ...
Setting up libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Setting up liblzo2-2:amd64 (2.10-2build3) ...
Setting up libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libsepol-dev:amd64 (3.3-1build1) ...
Setting up libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Setting up libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Setting up libselinux1-dev:amd64 (3.3-1build2) ...
Setting up libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libglib2.0-0:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Setting up libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  javascript-common libc-ares2 libjs-highlight.js libnode72 libuv1 nodejs-doc
Suggested packages:
  apache2 | lighttpd | httpd npm
The following NEW packages will be installed:
  javascript-common libc-ares2 libjs-highlight.js libnode72 libuv1 nodejs nodejs-doc
0 upgraded, 7 newly installed, 0 to remove and 2 not upgraded.
Need to get 13.8 MB of archives.
After this operation, 54.2 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libuv1 amd64 1.43.0-1 [93.1 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 javascript-common all 11+nmu1 [5936 B]
Get:3 http://archive.ubuntu.com/ubuntu jammy/universe amd64 libjs-highlight.js all 9.18.5+dfsg1-1 [367 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libc-ares2 amd64 1.18.1-1ubuntu0.22.04.2 [45.0 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 libnode72 amd64 12.22.9~dfsg-1ubuntu3.3 [10.8 MB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 nodejs-doc all 12.22.9~dfsg-1ubuntu3.3 [2410 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 nodejs amd64 12.22.9~dfsg-1ubuntu3.3 [122 kB]
Fetched 13.8 MB in 1s (17.2 MB/s)
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.3.2)
Defaulting to user installation because normal site-packages is not writeable
Collecting torch (from -r requirements.txt (line 1))
  Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl.metadata (25 kB)
Collecting numpy (from -r requirements.txt (line 2))
  Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 2.6 MB/s eta 0:00:00
Collecting tqdm (from -r requirements.txt (line 3))
  Downloading tqdm-4.66.1-py3-none-any.whl.metadata (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.6/57.6 kB 4.3 MB/s eta 0:00:00
Collecting tensorboard (from -r requirements.txt (line 4))
  Downloading tensorboard-2.15.1-py3-none-any.whl.metadata (1.7 kB)
Collecting ml-collections (from -r requirements.txt (line 5))
  Downloading ml_collections-0.1.1.tar.gz (77 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 5.7 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting filelock (from torch->-r requirements.txt (line 1))
  Downloading filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
Collecting typing-extensions (from torch->-r requirements.txt (line 1))
  Downloading typing_extensions-4.9.0-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy (from torch->-r requirements.txt (line 1))
  Downloading sympy-1.12-py3-none-any.whl (5.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.7/5.7 MB 20.1 MB/s eta 0:00:00
Collecting networkx (from torch->-r requirements.txt (line 1))
  Downloading networkx-3.2.1-py3-none-any.whl.metadata (5.2 kB)
Collecting jinja2 (from torch->-r requirements.txt (line 1))
  Downloading Jinja2-3.1.2-py3-none-any.whl (133 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 19.2 MB/s eta 0:00:00
Collecting fsspec (from torch->-r requirements.txt (line 1))
  Downloading fsspec-2023.12.2-py3-none-any.whl.metadata (6.8 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 93.2 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 72.6 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 97.5 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.1.3.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 17.3 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.0.2.54 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 36.2 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.2.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 31.3 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 50.2 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 35.3 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.18.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.8/209.8 MB 35.2 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 14.3 MB/s eta 0:00:00
Collecting triton==2.1.0 (from torch->-r requirements.txt (line 1))
  Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.3 kB)
Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch->-r requirements.txt (line 1))
  Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting absl-py>=0.4 (from tensorboard->-r requirements.txt (line 4))
  Downloading absl_py-2.0.0-py3-none-any.whl.metadata (2.3 kB)
Collecting grpcio>=1.48.2 (from tensorboard->-r requirements.txt (line 4))
  Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting google-auth<3,>=1.6.3 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth-2.26.1-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<2,>=0.5 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting markdown>=2.6.8 (from tensorboard->-r requirements.txt (line 4))
  Downloading Markdown-3.5.1-py3-none-any.whl.metadata (7.1 kB)
Collecting protobuf<4.24,>=3.19.6 (from tensorboard->-r requirements.txt (line 4))
  Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl.metadata (540 bytes)
Collecting requests<3,>=2.21.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools>=41.0.0 in /usr/lib/python3/dist-packages (from tensorboard->-r requirements.txt (line 4)) (59.6.0)
Collecting six>1.9 (from tensorboard->-r requirements.txt (line 4))
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard->-r requirements.txt (line 4))
  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
Collecting PyYAML (from ml-collections->-r requirements.txt (line 5))
  Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
Collecting contextlib2 (from ml-collections->-r requirements.txt (line 5))
  Downloading contextlib2-21.6.0-py2.py3-none-any.whl (13 kB)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading cachetools-5.3.2-py3-none-any.whl.metadata (5.2 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 23.5 MB/s eta 0:00:00
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading rsa-4.9-py3-none-any.whl (34 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading idna-3.6-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading urllib3-2.1.0-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading certifi-2023.11.17-py3-none-any.whl.metadata (2.2 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard->-r requirements.txt (line 4))
  Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
Collecting mpmath>=0.19 (from sympy->torch->-r requirements.txt (line 1))
  Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 50.7 MB/s eta 0:00:00
Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1-0.5.1-py2.py3-none-any.whl.metadata (8.6 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151.7/151.7 kB 20.1 MB/s eta 0:00:00
Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl (670.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 670.2/670.2 MB 12.2 MB/s eta 0:00:00
Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 10.0 MB/s eta 0:00:00
Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.2/89.2 MB 57.0 MB/s eta 0:00:00
Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 120.1 MB/s eta 0:00:00
Downloading tqdm-4.66.1-py3-none-any.whl (78 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.3/78.3 kB 8.2 MB/s eta 0:00:00
Downloading tensorboard-2.15.1-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 139.5 MB/s eta 0:00:00
Downloading absl_py-2.0.0-py3-none-any.whl (130 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 18.3 MB/s eta 0:00:00
Downloading google_auth-2.26.1-py2.py3-none-any.whl (186 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 186.4/186.4 kB 20.6 MB/s eta 0:00:00
Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl (24 kB)
Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 161.2 MB/s eta 0:00:00
Downloading Markdown-3.5.1-py3-none-any.whl (102 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 102.2/102.2 kB 14.7 MB/s eta 0:00:00
Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl (304 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 304.5/304.5 kB 28.9 MB/s eta 0:00:00
Downloading requests-2.31.0-py3-none-any.whl (62 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 8.6 MB/s eta 0:00:00
Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 157.4 MB/s eta 0:00:00
Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 28.7 MB/s eta 0:00:00
Downloading filelock-3.13.1-py3-none-any.whl (11 kB)
Downloading fsspec-2023.12.2-py3-none-any.whl (168 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 169.0/169.0 kB 22.9 MB/s eta 0:00:00
Downloading networkx-3.2.1-py3-none-any.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 90.4 MB/s eta 0:00:00
Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 705.5/705.5 kB 61.3 MB/s eta 0:00:00
Downloading typing_extensions-4.9.0-py3-none-any.whl (32 kB)
Downloading cachetools-5.3.2-py3-none-any.whl (9.3 kB)
Downloading certifi-2023.11.17-py3-none-any.whl (162 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.5/162.5 kB 18.0 MB/s eta 0:00:00
Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 15.4 MB/s eta 0:00:00
Downloading idna-3.6-py3-none-any.whl (61 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 kB 8.8 MB/s eta 0:00:00
Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Downloading urllib3-2.1.0-py3-none-any.whl (104 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.6/104.6 kB 14.0 MB/s eta 0:00:00
Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl (20.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.5/20.5 MB 129.7 MB/s eta 0:00:00
Downloading pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 13.5 MB/s eta 0:00:00
Building wheels for collected packages: ml-collections
  Building wheel for ml-collections (setup.py) ... done
  Created wheel for ml-collections: filename=ml_collections-0.1.1-py3-none-any.whl size=94522 sha256=bfc71cbe1c75f1a437b10cd7c91266f2aaaf0a90b0d82ed13846aa89784b4cca
  Stored in directory: /home/vscode/.cache/pip/wheels/7b/89/c9/a9b87790789e94aadcfc393c283e3ecd5ab916aed0a31be8fe
Successfully built ml-collections
Installing collected packages: mpmath, urllib3, typing-extensions, tqdm, tensorboard-data-server, sympy, six, PyYAML, pyasn1, protobuf, oauthlib, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, markdown, idna, grpcio, fsspec, filelock, contextlib2, charset-normalizer, certifi, cachetools, absl-py, werkzeug, triton, rsa, requests, pyasn1-modules, nvidia-cusparse-cu12, nvidia-cudnn-cu12, ml-collections, jinja2, requests-oauthlib, nvidia-cusolver-cu12, google-auth, torch, google-auth-oauthlib, tensorboard
  WARNING: The script tqdm is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script isympy is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script f2py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown_py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script normalizer is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts pyrsa-decrypt, pyrsa-encrypt, pyrsa-keygen, pyrsa-priv2pub, pyrsa-sign and pyrsa-verify are installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts convert-caffe2-to-onnx, convert-onnx-to-caffe2 and torchrun are installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script google-oauthlib-tool is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script tensorboard is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed MarkupSafe-2.1.3 PyYAML-6.0.1 absl-py-2.0.0 cachetools-5.3.2 certifi-2023.11.17 charset-normalizer-3.3.2 contextlib2-21.6.0 filelock-3.13.1 fsspec-2023.12.2 google-auth-2.26.1 google-auth-oauthlib-1.2.0 grpcio-1.60.0 idna-3.6 jinja2-3.1.2 markdown-3.5.1 ml-collections-0.1.1 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.3 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.3.101 nvidia-nvtx-cu12-12.1.105 oauthlib-3.2.2 protobuf-4.23.4 pyasn1-0.5.1 pyasn1-modules-0.3.0 requests-2.31.0 requests-oauthlib-1.3.1 rsa-4.9 six-1.16.0 sympy-1.12 tensorboard-2.15.1 tensorboard-data-server-0.7.2 torch-2.1.2 tqdm-4.66.1 triton-2.1.0 typing-extensions-4.9.0 urllib3-2.1.0 werkzeug-3.0.1
Running the postCreateCommand from Feature 'ghcr.io/devcontainers/features/git-lfs:1'...

[107427 ms] Start: Run in container: /bin/sh -c /usr/local/share/pull-git-lfs-artifacts.sh
Fetching git lfs artifacts...
Running the postCreateCommand from devcontainer.json...

[107541 ms] Start: Run in container: nvidia-smi
Fri Jan  5 19:13:07 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On  | 00000000:00:06.0 Off |                  N/A |
| 30%   28C    P8              25W / 350W |      3MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Done. Press any key to close the terminal.


vscode ➜ /workspaces/ViT-pytorch (dev_container) $ python has_cuda.py 
has_cuda = True
device = cuda
n_gpu = 1


Running the updateContentCommand from devcontainer.json...

[31818 ms] Start: Run in container: /bin/sh -c bash .devcontainer/install-dev-tools.sh
Get:1 http://archive.ubuntu.com/ubuntu jammy InRelease [270 kB]
Get:2 http://security.ubuntu.com/ubuntu jammy-security InRelease [110 kB]                                            
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [119 kB]                                                        
Get:4 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [109 kB]                                                               
Get:5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease [1581 B]                                
Get:6 http://archive.ubuntu.com/ubuntu jammy/multiverse amd64 Packages [266 kB]                                    
Get:7 http://archive.ubuntu.com/ubuntu jammy/universe amd64 Packages [17.5 MB]                                  
Get:8 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  Packages [1323 kB]              
Get:9 http://archive.ubuntu.com/ubuntu jammy/restricted amd64 Packages [164 kB]                    
Get:10 http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1792 kB]                                                       
Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1307 kB]                                              
Get:12 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [1047 kB]                                            
0% [7 Packages store 0 B] [11 Packages 1107 kB/1307 kB 85%] [12 Packages 2686 B/1047 kB 0%] [8 Packages 262 kB/1323 kB 20%] [Waiting for he                                                                                                                                           Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [1606 kB]                                                  
0% [7 Packages store 0 B] [13 Packages 1385 B/1606 kB 0%] [12 Packages 7030 B/1047 kB 1%] [8 Packages 262 kB/1323 kB 20%] [Waiting for head                                                                                                                                           Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [49.8 kB]                                             
0% [7 Packages store 0 B] [14 Packages 1211 B/49.8 kB 2%] [12 Packages 41.8 kB/1047 kB 4%] [8 Packages 557 kB/1323 kB 42%] [Waiting for hea                                                                                                                                           Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [1611 kB]                                             
0% [7 Packages store 0 B] [15 Packages 340 B/1611 kB 0%] [12 Packages 41.8 kB/1047 kB 4%] [8 Packages 557 kB/1323 kB 42%] [Waiting for head                                                                                                                                           0% [7 Packages store 0 B] [Connecting to archive.ubuntu.com (185.125.190.39)] [12 Packages 124 kB/1047 kB 12%] [8 Packages 1130 kB/1323 kB                                                                                                                                            Get:17 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [28.1 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [50.4 kB]                                                     
Get:19 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [44.0 kB]                             
Get:20 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [1582 kB]
Get:21 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [1332 kB]
Get:16 https://packagecloud.io/github/git-lfs/ubuntu jammy InRelease [28.0 kB]   
Get:22 https://packagecloud.io/github/git-lfs/ubuntu jammy/main amd64 Packages [1654 B]
Fetched 30.3 MB in 3s (11.3 MB/s)    
Reading package lists... Done
cuda-toolkit set on hold.
libcudnn8-dev set on hold.
libcudnn8 set on hold.
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Calculating upgrade... Done
The following packages have been kept back:
  libcudnn8 libcudnn8-dev
The following packages will be upgraded:
  binutils binutils-common binutils-x86-64-linux-gnu cuda-keyring distro-info-data libbinutils libctf-nobfd0 libctf0 libssh-4 openssh-client
  vim-common vim-tiny xxd
13 upgraded, 0 newly installed, 0 to remove and 2 not upgraded.
Need to get 5372 kB of archives.
After this operation, 5120 B of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 distro-info-data all 0.52ubuntu0.6 [5160 B]
Get:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  cuda-keyring 1.1-1 [4328 B]
Get:3 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-tiny amd64 2:8.2.3995-1ubuntu2.15 [710 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 xxd amd64 2:8.2.3995-1ubuntu2.15 [55.2 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 vim-common all 2:8.2.3995-1ubuntu2.15 [81.5 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 openssh-client amd64 1:8.9p1-3ubuntu0.6 [906 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf0 amd64 2.38-4ubuntu2.4 [103 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libctf-nobfd0 amd64 2.38-4ubuntu2.4 [108 kB]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-x86-64-linux-gnu amd64 2.38-4ubuntu2.4 [2327 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libbinutils amd64 2.38-4ubuntu2.4 [662 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils amd64 2.38-4ubuntu2.4 [3194 B]
Get:12 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 binutils-common amd64 2.38-4ubuntu2.4 [222 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libssh-4 amd64 0.9.6-2ubuntu0.22.04.2 [186 kB]
Fetched 5372 kB in 1s (4743 kB/s)   
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-distro-info-data_0.52ubuntu0.6_all.deb ...
Unpacking distro-info-data (0.52ubuntu0.6) over (0.52ubuntu0.5) ...
Preparing to unpack .../01-vim-tiny_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking vim-tiny (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../02-xxd_2%3a8.2.3995-1ubuntu2.15_amd64.deb ...
Unpacking xxd (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../03-vim-common_2%3a8.2.3995-1ubuntu2.15_all.deb ...
Unpacking vim-common (2:8.2.3995-1ubuntu2.15) over (2:8.2.3995-1ubuntu2.13) ...
Preparing to unpack .../04-openssh-client_1%3a8.9p1-3ubuntu0.6_amd64.deb ...
Unpacking openssh-client (1:8.9p1-3ubuntu0.6) over (1:8.9p1-3ubuntu0.4) ...
Preparing to unpack .../05-libctf0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../06-libctf-nobfd0_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libctf-nobfd0:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../07-binutils-x86-64-linux-gnu_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../08-libbinutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking libbinutils:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../09-binutils_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../10-binutils-common_2.38-4ubuntu2.4_amd64.deb ...
Unpacking binutils-common:amd64 (2.38-4ubuntu2.4) over (2.38-4ubuntu2.3) ...
Preparing to unpack .../11-cuda-keyring_1.1-1_all.deb ...
Unpacking cuda-keyring (1.1-1) over (1.0-1) ...
Preparing to unpack .../12-libssh-4_0.9.6-2ubuntu0.22.04.2_amd64.deb ...
Unpacking libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) over (0.9.6-2ubuntu0.22.04.1) ...
Setting up distro-info-data (0.52ubuntu0.6) ...
Setting up openssh-client (1:8.9p1-3ubuntu0.6) ...
Setting up binutils-common:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf-nobfd0:amd64 (2.38-4ubuntu2.4) ...
Setting up xxd (2:8.2.3995-1ubuntu2.15) ...
Setting up vim-common (2:8.2.3995-1ubuntu2.15) ...
Setting up cuda-keyring (1.1-1) ...
Setting up libssh-4:amd64 (0.9.6-2ubuntu0.22.04.2) ...
Setting up libbinutils:amd64 (2.38-4ubuntu2.4) ...
Setting up libctf0:amd64 (2.38-4ubuntu2.4) ...
Setting up vim-tiny (2:8.2.3995-1ubuntu2.15) ...
Setting up binutils-x86-64-linux-gnu (2.38-4ubuntu2.4) ...
Setting up binutils (2.38-4ubuntu2.4) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
pkg-config is already the newest version (0.29.2-1ubuntu3).
pkg-config set to manually installed.
python3-dev is already the newest version (3.10.6-1~22.04).
The following additional packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2 libmount-dev
  libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev libxcb-shm0-dev
Suggested packages:
  libcairo2-doc libgirepository1.0-dev libglib2.0-doc libgdk-pixbuf2.0-bin | libgdk-pixbuf2.0-dev libxml2-utils libice-doc libsm-doc
The following NEW packages will be installed:
  libblkid-dev libcairo-gobject2 libcairo-script-interpreter2 libcairo2-dev libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libice-dev liblzo2-2
  libmount-dev libpcre16-3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libpixman-1-dev libselinux1-dev libsepol-dev libsm-dev libxcb-render0-dev
  libxcb-shm0-dev
0 upgraded, 20 newly installed, 0 to remove and 2 not upgraded.
Need to get 4790 kB of archives.
After this operation, 23.4 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-gobject2 amd64 1.16.0-5ubuntu2 [19.4 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 liblzo2-2 amd64 2.10-2build3 [53.7 kB]
Get:3 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo-script-interpreter2 amd64 1.16.0-5ubuntu2 [62.0 kB]
Get:4 http://archive.ubuntu.com/ubuntu jammy/main amd64 libice-dev amd64 2:1.0.10-1build2 [51.4 kB]
Get:5 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsm-dev amd64 2:1.2.3-1build2 [18.1 kB]
Get:6 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpixman-1-dev amd64 0.40.0-1ubuntu0.22.04.1 [280 kB]
Get:7 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-render0-dev amd64 1.14-3ubuntu3 [19.6 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy/main amd64 libxcb-shm0-dev amd64 1.14-3ubuntu3 [6848 B]
Get:9 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-bin amd64 2.72.4-0ubuntu2.2 [80.9 kB]
Get:10 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev-bin amd64 2.72.4-0ubuntu2.2 [117 kB]
Get:11 http://archive.ubuntu.com/ubuntu jammy/main amd64 libblkid-dev amd64 2.37.2-4ubuntu3 [185 kB]
Get:12 http://archive.ubuntu.com/ubuntu jammy/main amd64 libsepol-dev amd64 3.3-1build1 [378 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy/main amd64 libselinux1-dev amd64 3.3-1build2 [158 kB]
Get:14 http://archive.ubuntu.com/ubuntu jammy/main amd64 libmount-dev amd64 2.37.2-4ubuntu3 [14.5 kB]
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre16-3 amd64 2:8.39-13ubuntu0.22.04.1 [164 kB]
Get:16 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre32-3 amd64 2:8.39-13ubuntu0.22.04.1 [155 kB]
Get:17 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcrecpp0v5 amd64 2:8.39-13ubuntu0.22.04.1 [16.5 kB]
Get:18 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libpcre3-dev amd64 2:8.39-13ubuntu0.22.04.1 [579 kB]
Get:19 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 libglib2.0-dev amd64 2.72.4-0ubuntu2.2 [1739 kB]
Get:20 http://archive.ubuntu.com/ubuntu jammy/main amd64 libcairo2-dev amd64 1.16.0-5ubuntu2 [692 kB]
Fetched 4790 kB in 1s (5509 kB/s)      
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Selecting previously unselected package libcairo-gobject2:amd64.
(Reading database ... 35779 files and directories currently installed.)
Preparing to unpack .../00-libcairo-gobject2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package liblzo2-2:amd64.
Preparing to unpack .../01-liblzo2-2_2.10-2build3_amd64.deb ...
Unpacking liblzo2-2:amd64 (2.10-2build3) ...
Selecting previously unselected package libcairo-script-interpreter2:amd64.
Preparing to unpack .../02-libcairo-script-interpreter2_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Selecting previously unselected package libice-dev:amd64.
Preparing to unpack .../03-libice-dev_2%3a1.0.10-1build2_amd64.deb ...
Unpacking libice-dev:amd64 (2:1.0.10-1build2) ...
Selecting previously unselected package libsm-dev:amd64.
Preparing to unpack .../04-libsm-dev_2%3a1.2.3-1build2_amd64.deb ...
Unpacking libsm-dev:amd64 (2:1.2.3-1build2) ...
Selecting previously unselected package libpixman-1-dev:amd64.
Preparing to unpack .../05-libpixman-1-dev_0.40.0-1ubuntu0.22.04.1_amd64.deb ...
Unpacking libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Selecting previously unselected package libxcb-render0-dev:amd64.
Preparing to unpack .../06-libxcb-render0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libxcb-shm0-dev:amd64.
Preparing to unpack .../07-libxcb-shm0-dev_1.14-3ubuntu3_amd64.deb ...
Unpacking libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Selecting previously unselected package libglib2.0-bin.
Preparing to unpack .../08-libglib2.0-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libglib2.0-dev-bin.
Preparing to unpack .../09-libglib2.0-dev-bin_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libblkid-dev:amd64.
Preparing to unpack .../10-libblkid-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libsepol-dev:amd64.
Preparing to unpack .../11-libsepol-dev_3.3-1build1_amd64.deb ...
Unpacking libsepol-dev:amd64 (3.3-1build1) ...
Selecting previously unselected package libselinux1-dev:amd64.
Preparing to unpack .../12-libselinux1-dev_3.3-1build2_amd64.deb ...
Unpacking libselinux1-dev:amd64 (3.3-1build2) ...
Selecting previously unselected package libmount-dev:amd64.
Preparing to unpack .../13-libmount-dev_2.37.2-4ubuntu3_amd64.deb ...
Unpacking libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Selecting previously unselected package libpcre16-3:amd64.
Preparing to unpack .../14-libpcre16-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre32-3:amd64.
Preparing to unpack .../15-libpcre32-3_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcrecpp0v5:amd64.
Preparing to unpack .../16-libpcrecpp0v5_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libpcre3-dev:amd64.
Preparing to unpack .../17-libpcre3-dev_2%3a8.39-13ubuntu0.22.04.1_amd64.deb ...
Unpacking libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Selecting previously unselected package libglib2.0-dev:amd64.
Preparing to unpack .../18-libglib2.0-dev_2.72.4-0ubuntu2.2_amd64.deb ...
Unpacking libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Selecting previously unselected package libcairo2-dev:amd64.
Preparing to unpack .../19-libcairo2-dev_1.16.0-5ubuntu2_amd64.deb ...
Unpacking libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
Setting up libpcrecpp0v5:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libglib2.0-dev-bin (2.72.4-0ubuntu2.2) ...
Setting up libblkid-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libpixman-1-dev:amd64 (0.40.0-1ubuntu0.22.04.1) ...
Setting up libpcre16-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libice-dev:amd64 (2:1.0.10-1build2) ...
Setting up libsm-dev:amd64 (2:1.2.3-1build2) ...
Setting up libglib2.0-bin (2.72.4-0ubuntu2.2) ...
Setting up liblzo2-2:amd64 (2.10-2build3) ...
Setting up libxcb-shm0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libpcre32-3:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libsepol-dev:amd64 (3.3-1build1) ...
Setting up libxcb-render0-dev:amd64 (1.14-3ubuntu3) ...
Setting up libcairo-gobject2:amd64 (1.16.0-5ubuntu2) ...
Setting up libcairo-script-interpreter2:amd64 (1.16.0-5ubuntu2) ...
Setting up libselinux1-dev:amd64 (3.3-1build2) ...
Setting up libpcre3-dev:amd64 (2:8.39-13ubuntu0.22.04.1) ...
Setting up libmount-dev:amd64 (2.37.2-4ubuntu3) ...
Setting up libglib2.0-dev:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libglib2.0-0:amd64 (2.72.4-0ubuntu2.2) ...
Processing triggers for libc-bin (2.35-0ubuntu3.5) ...
Processing triggers for man-db (2.10.2-1) ...
Setting up libcairo2-dev:amd64 (1.16.0-5ubuntu2) ...
2024-01-05 19:27:32 - Installing pre-requisites
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Hit:2 http://security.ubuntu.com/ubuntu jammy-security InRelease                                                                      
Hit:3 http://archive.ubuntu.com/ubuntu jammy-updates InRelease                                                                        
Hit:4 http://archive.ubuntu.com/ubuntu jammy-backports InRelease                                                                      
Hit:5 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease   
Hit:6 https://packagecloud.io/github/git-lfs/ubuntu jammy InRelease     
Reading package lists... Done
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
ca-certificates is already the newest version (20230311ubuntu0.22.04.1).
curl is already the newest version (7.81.0-1ubuntu1.15).
gnupg is already the newest version (2.2.27-3ubuntu2.1).
apt-transport-https is already the newest version (2.4.11).
0 upgraded, 0 newly installed, 0 to remove and 2 not upgraded.
gpg: WARNING: unsafe ownership on homedir '/home/vscode/.gnupg'
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Get:2 https://deb.nodesource.com/node_21.x nodistro InRelease [12.1 kB]                                            
0% [Waiting for headers] [Waiting for headers] [Waiting for headers] [Connecting to packagecloud.io (54.215.95.52)] [2 InRelease 12.1 kB/12                                                                                                                                           Hit:3 http://archive.ubuntu.com/ubuntu jammy-updates InRelease                                                     
Hit:4 http://archive.ubuntu.com/ubuntu jammy-backports InRelease                                                   
Hit:5 http://security.ubuntu.com/ubuntu jammy-security InRelease                             
Hit:6 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease  
Get:7 https://deb.nodesource.com/node_21.x nodistro/main amd64 Packages [1827 B]
Hit:8 https://packagecloud.io/github/git-lfs/ubuntu jammy InRelease             
Fetched 14.0 kB in 1s (10.7 kB/s)
Reading package lists... Done
2024-01-05 19:27:37 - Repository configured successfully. To install Node.js, run: apt-get install nodejs -y
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following NEW packages will be installed:
  nodejs
0 upgraded, 1 newly installed, 0 to remove and 2 not upgraded.
Need to get 32.4 MB of archives.
After this operation, 201 MB of additional disk space will be used.
Get:1 https://deb.nodesource.com/node_21.x nodistro/main amd64 nodejs amd64 21.5.0-1nodesource1 [32.4 MB]
Fetched 32.4 MB in 1s (51.8 MB/s)
debconf: unable to initialize frontend: Dialog
debconf: (Dialog frontend requires a screen at least 13 lines tall and 31 columns wide.)
debconf: falling back to frontend: Readline
Selecting previously unselected package nodejs.
(Reading database ... 36691 files and directories currently installed.)
Preparing to unpack .../nodejs_21.5.0-1nodesource1_amd64.deb ...
Unpacking nodejs (21.5.0-1nodesource1) ...
Setting up nodejs (21.5.0-1nodesource1) ...
Processing triggers for man-db (2.10.2-1) ...
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.3.2)
Defaulting to user installation because normal site-packages is not writeable
Collecting torch (from -r requirements.txt (line 1))
  Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl.metadata (25 kB)
Collecting numpy (from -r requirements.txt (line 2))
  Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.2/61.2 kB 2.7 MB/s eta 0:00:00
Collecting tqdm (from -r requirements.txt (line 3))
  Downloading tqdm-4.66.1-py3-none-any.whl.metadata (57 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.6/57.6 kB 4.1 MB/s eta 0:00:00
Collecting tensorboard (from -r requirements.txt (line 4))
  Downloading tensorboard-2.15.1-py3-none-any.whl.metadata (1.7 kB)
Collecting ml-collections (from -r requirements.txt (line 5))
  Downloading ml_collections-0.1.1.tar.gz (77 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 5.4 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting filelock (from torch->-r requirements.txt (line 1))
  Downloading filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
Collecting typing-extensions (from torch->-r requirements.txt (line 1))
  Downloading typing_extensions-4.9.0-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy (from torch->-r requirements.txt (line 1))
  Downloading sympy-1.12-py3-none-any.whl (5.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.7/5.7 MB 20.1 MB/s eta 0:00:00
Collecting networkx (from torch->-r requirements.txt (line 1))
  Downloading networkx-3.2.1-py3-none-any.whl.metadata (5.2 kB)
Collecting jinja2 (from torch->-r requirements.txt (line 1))
  Downloading Jinja2-3.1.2-py3-none-any.whl (133 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.1/133.1 kB 12.9 MB/s eta 0:00:00
Collecting fsspec (from torch->-r requirements.txt (line 1))
  Downloading fsspec-2023.12.2-py3-none-any.whl.metadata (6.8 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 90.2 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 41.0 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 132.7 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.1.3.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 18.9 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.0.2.54 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 47.6 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.2.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 77.6 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 52.3 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 35.4 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.18.1 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.8/209.8 MB 35.4 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.1.105 (from torch->-r requirements.txt (line 1))
  Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 12.5 MB/s eta 0:00:00
Collecting triton==2.1.0 (from torch->-r requirements.txt (line 1))
  Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.3 kB)
Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch->-r requirements.txt (line 1))
  Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting absl-py>=0.4 (from tensorboard->-r requirements.txt (line 4))
  Downloading absl_py-2.0.0-py3-none-any.whl.metadata (2.3 kB)
Collecting grpcio>=1.48.2 (from tensorboard->-r requirements.txt (line 4))
  Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting google-auth<3,>=1.6.3 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth-2.26.1-py2.py3-none-any.whl.metadata (4.7 kB)
Collecting google-auth-oauthlib<2,>=0.5 (from tensorboard->-r requirements.txt (line 4))
  Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl.metadata (2.7 kB)
Collecting markdown>=2.6.8 (from tensorboard->-r requirements.txt (line 4))
  Downloading Markdown-3.5.1-py3-none-any.whl.metadata (7.1 kB)
Collecting protobuf<4.24,>=3.19.6 (from tensorboard->-r requirements.txt (line 4))
  Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl.metadata (540 bytes)
Collecting requests<3,>=2.21.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: setuptools>=41.0.0 in /usr/lib/python3/dist-packages (from tensorboard->-r requirements.txt (line 4)) (59.6.0)
Collecting six>1.9 (from tensorboard->-r requirements.txt (line 4))
  Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard->-r requirements.txt (line 4))
  Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
Collecting werkzeug>=1.0.1 (from tensorboard->-r requirements.txt (line 4))
  Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
Collecting PyYAML (from ml-collections->-r requirements.txt (line 5))
  Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
Collecting contextlib2 (from ml-collections->-r requirements.txt (line 5))
  Downloading contextlib2-21.6.0-py2.py3-none-any.whl (13 kB)
Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading cachetools-5.3.2-py3-none-any.whl.metadata (5.2 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 18.1 MB/s eta 0:00:00
Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading rsa-4.9-py3-none-any.whl (34 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
Collecting charset-normalizer<4,>=2 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading idna-3.6-py3-none-any.whl.metadata (9.9 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading urllib3-2.1.0-py3-none-any.whl.metadata (6.4 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorboard->-r requirements.txt (line 4))
  Downloading certifi-2023.11.17-py3-none-any.whl.metadata (2.2 kB)
Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard->-r requirements.txt (line 4))
  Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
Collecting mpmath>=0.19 (from sympy->torch->-r requirements.txt (line 1))
  Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 45.6 MB/s eta 0:00:00
Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->-r requirements.txt (line 4))
  Downloading pyasn1-0.5.1-py2.py3-none-any.whl.metadata (8.6 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard->-r requirements.txt (line 4))
  Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151.7/151.7 kB 17.3 MB/s eta 0:00:00
Downloading torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl (670.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 670.2/670.2 MB 11.6 MB/s eta 0:00:00
Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 10.9 MB/s eta 0:00:00
Downloading triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 89.2/89.2 MB 60.1 MB/s eta 0:00:00
Downloading numpy-1.26.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 117.8 MB/s eta 0:00:00
Downloading tqdm-4.66.1-py3-none-any.whl (78 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.3/78.3 kB 9.2 MB/s eta 0:00:00
Downloading tensorboard-2.15.1-py3-none-any.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 142.8 MB/s eta 0:00:00
Downloading absl_py-2.0.0-py3-none-any.whl (130 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 19.1 MB/s eta 0:00:00
Downloading google_auth-2.26.1-py2.py3-none-any.whl (186 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 186.4/186.4 kB 21.2 MB/s eta 0:00:00
Downloading google_auth_oauthlib-1.2.0-py2.py3-none-any.whl (24 kB)
Downloading grpcio-1.60.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.4/5.4 MB 132.5 MB/s eta 0:00:00
Downloading Markdown-3.5.1-py3-none-any.whl (102 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 102.2/102.2 kB 13.8 MB/s eta 0:00:00
Downloading protobuf-4.23.4-cp37-abi3-manylinux2014_x86_64.whl (304 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 304.5/304.5 kB 28.7 MB/s eta 0:00:00
Downloading requests-2.31.0-py3-none-any.whl (62 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 8.9 MB/s eta 0:00:00
Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 166.7 MB/s eta 0:00:00
Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 29.6 MB/s eta 0:00:00
Downloading filelock-3.13.1-py3-none-any.whl (11 kB)
Downloading fsspec-2023.12.2-py3-none-any.whl (168 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 169.0/169.0 kB 23.4 MB/s eta 0:00:00
Downloading networkx-3.2.1-py3-none-any.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 76.2 MB/s eta 0:00:00
Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 705.5/705.5 kB 55.8 MB/s eta 0:00:00
Downloading typing_extensions-4.9.0-py3-none-any.whl (32 kB)
Downloading cachetools-5.3.2-py3-none-any.whl (9.3 kB)
Downloading certifi-2023.11.17-py3-none-any.whl (162 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.5/162.5 kB 17.8 MB/s eta 0:00:00
Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 15.9 MB/s eta 0:00:00
Downloading idna-3.6-py3-none-any.whl (61 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.6/61.6 kB 8.6 MB/s eta 0:00:00
Downloading MarkupSafe-2.1.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Downloading urllib3-2.1.0-py3-none-any.whl (104 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.6/104.6 kB 11.7 MB/s eta 0:00:00
Downloading nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl (20.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.5/20.5 MB 128.6 MB/s eta 0:00:00
Downloading pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 8.7 MB/s eta 0:00:00
Building wheels for collected packages: ml-collections
  Building wheel for ml-collections (setup.py) ... done
  Created wheel for ml-collections: filename=ml_collections-0.1.1-py3-none-any.whl size=94522 sha256=b72ce51eb58f1faa2a90b7e075818f56ceab99c6c4c24c12edb5dcabb4fb3e44
  Stored in directory: /home/vscode/.cache/pip/wheels/7b/89/c9/a9b87790789e94aadcfc393c283e3ecd5ab916aed0a31be8fe
Successfully built ml-collections
Installing collected packages: mpmath, urllib3, typing-extensions, tqdm, tensorboard-data-server, sympy, six, PyYAML, pyasn1, protobuf, oauthlib, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, markdown, idna, grpcio, fsspec, filelock, contextlib2, charset-normalizer, certifi, cachetools, absl-py, werkzeug, triton, rsa, requests, pyasn1-modules, nvidia-cusparse-cu12, nvidia-cudnn-cu12, ml-collections, jinja2, requests-oauthlib, nvidia-cusolver-cu12, google-auth, torch, google-auth-oauthlib, tensorboard
  WARNING: The script tqdm is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script isympy is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script f2py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script markdown_py is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script normalizer is installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts pyrsa-decrypt, pyrsa-encrypt, pyrsa-keygen, pyrsa-priv2pub, pyrsa-sign and pyrsa-verify are installed in '/home/vscode/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
