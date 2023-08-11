#!/usr/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
for ff in ./fastas/*.fasta
do
    python ~/project_install_ef2/ESM_complex_prediction.py  -f $ff    
done