# download esmfold params
sudo apt-get install aria2 -qq

CURRENTPATH=`pwd`
CONDAFOLDDIR="${CURRENTPATH}/esmfold"

mkdir -p ${CONDAFOLDDIR}
#!/usr/bin/bash
cd ${CONDAFOLDDIR}
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDAFOLDDIR}/conda
rm Miniconda3-latest-Linux-x86_64.sh
. "${CONDAFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${CONDAFOLDDIR}/conda/condabin:${PATH}"
conda create -p $CONDAFOLDDIR/colabfold-conda python=3.10 -y
conda activate $CONDAFOLDDIR/colabfold-conda
conda update -n base conda -y

aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/esmfold.model

# install libs
pip install  omegaconf pytorch_lightning biopython ml_collections einops py3Dmol
pip install  git+https://github.com/NVIDIA/dllogger.git"

# install openfold
commit = "6908936b68ae89f67755240e2f588c09ec31d4c8"
pip install  git+https://github.com/aqlaboratory/openfold.git@$commit


pip install  git+https://github.com/sokrypton/esm.git@beta

# wait for Params to finish downloading...

aria2c -x 16 https://files.ipd.uw.edu/pub/esmfold/esmfold.model

sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

