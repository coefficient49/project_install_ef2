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
conda env create -f ./project_install_ef2/environment.yml
#conda activate esmfold
conda activate esmfold-conda
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install tqdm dnaio "jax[cpu]" matplotlib
git clone -b beta https://github.com/sokrypton/esm.git
cd ./esm

pip install .


cd ..

aria2c -x 16 https://colabfold.steineggerlab.workers.dev/esm/esmfold.model

sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


