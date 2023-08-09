### for init ESM
CURRENTPATH=`pwd`
CONDAFOLDDIR="${CURRENTPATH}/esmfold"
export PATH="${CONDAFOLDDIR}/conda/condabin:${PATH}"
export PATH="${CONDAFOLDDIR}/colabfold-conda/bin:$PATH"
conda activate $CONDAFOLDDIR/colabfold-conda
