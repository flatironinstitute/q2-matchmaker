# This is an installation script to automatically download and compile cmdstan
# wget https://github.com/stan-dev/cmdstan/releases/download/v2.26.0/cmdstan-2.26.0.tar.gz
# tar -zxvf cmdstan-2.26.0.tar.gz
# cd cmdstan-2.26.0 && make build -j4
# see https://github.com/stan-dev/cmdstanpy/issues/374
export CXX=g++
export TBB_CXX_TYPE=gcc
install_cmdstan
