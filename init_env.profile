### for init ESM
CURRENTPATH=`pwd`
CONDAFOLDDIR="${CURRENTPATH}/esmfold"
export PATH="${CONDAFOLDDIR}/conda/condabin:${PATH}"
eval "$(conda shell.bash hook)"
conda create -f environment.yml
#conda activate esmfold
