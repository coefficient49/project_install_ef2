### for init ESM
CURRENTPATH=`pwd`
CONDAFOLDDIR="${CURRENTPATH}/esmfold"
export PATH="${CONDAFOLDDIR}/conda/condabin:${PATH}"
export PATH="${CONDAFOLDDIR}/colabfold-conda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate $CONDAFOLDDIR/colabfold-conda
