#!/bin/bash

# mkdir deps &> /dev/null
# cd deps
#
# #Add necessary extra repos
# version=$(lsb_release -a 2>&1)
# if [[ $version == *"14.04"* ]] ; then
#     wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
#     sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
#     rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
#     sudo apt-get update
#     sudo apt-get install cuda-7-5
# elif [[ $version == *"15.04"* ]] ; then
#     wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
#     sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
#     rm cuda-repo-ubuntu1504_7.5-18_amd64.deb
#     sudo apt-get update
#     sudo apt-get install cuda-7-5
# elif [[ $version == *"16.04"* ]] ; then
#     wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
#     sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
#     rm cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
#     sudo add-apt-repository ppa:openjdk-r/ppa
#     sudo apt-get update
#     sudo apt-get install cuda-8-0
# else
#     echo "Don't use this on anything except 14.04, 15.04, or 16.04"
#     exit
# fi
#
# sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev libglew-dev libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev

#Installing Pangolin
# git clone https://github.com/stevenlovegrove/Pangolin.git
# cd Pangolin
# mkdir build
# cd build
# cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
# make -j8
# cd ../..

#Actually build ElasticFusion
cd ./Core
mkdir build
cd build
cmake ../src
make -j
cd ../../GUI
mkdir build
cd build
cmake ../src
make -j
