# download esmfold params
sudo apt-get install aria2 -qq

CURRENTPATH=`pwd`
CONDAFOLDDIR="${CURRENTPATH}/esmfold"

mkdir -p ${CONDAFOLDDIR}
#!/usr/bin/bash

wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDAFOLDDIR}/conda
rm Miniconda3-latest-Linux-x86_64.sh

export PATH="${CONDAFOLDDIR}/conda/condabin:${PATH}"
eval "$(conda shell.bash hook)"
conda create -f environmental.yml -y
conda activate $CONDAFOLDDIR/esmfold-conda
conda update -n base conda -y


pip install torch torchvision torchaudio

pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

