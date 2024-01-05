#!/bin/bash

# update system
sudo apt-get update
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

sudo apt install -y nodejs

# Why version 12, when TF requires Cuda 11.8? 
# Could not load library libcublasLt.so.12. Error: libcublasLt.so.12: cannot open shared object file: No such file or directory
# Aborted (core dumped)
# sudo apt-get install -y libcublas-12-0

# install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
