#!/bin/bash

# update system
sudo apt-get update
# https://askubuntu.com/questions/99774/exclude-packages-from-apt-get-upgrade
# https://help.ubuntu.com/community/PinningHowto
# https://linuxopsys.com/topics/exclude-specific-package-apt-upgrade
sudo apt-mark hold cuda-toolkit libcudnn8-dev libcudnn8
sudo apt-get upgrade -y

# Para executar o GUI
# https://stackoverflow.com/questions/15884075/tkinter-in-a-virtualenv
# sudo apt-get install python3-tk
# Required for pycairo (https://cairographics.org/)
# pkg-config --print-errors --exists cairo >= 1.15.10
# https://pycairo.readthedocs.io/en/latest/getting_started.html
# https://askubuntu.com/questions/1377608/what-is-the-package-for-pycairo-in-ubuntu
# sudo apt-get install libcairo2
# sudo apt-get install libcairo2-dev
sudo apt-get install -y libcairo2-dev pkg-config python3-dev

# https://nodejs.org/en/download/package-manager#debian-and-ubuntu-based-linux-distributions
# https://github.com/nodesource/distributions
# sudo apt install -y nodejs
curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash - &&\
sudo apt-get install -y nodejs

# Why version 12, when TF requires Cuda 11.8? 
# Could not load library libcublasLt.so.12. Error: libcublasLt.so.12: cannot open shared object file: No such file or directory
# Aborted (core dumped)
# sudo apt-get install -y libcublas-12-0

# install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
