
for ff in ./fastas/*.fasta
do
    python ~/project_install_ef2/ESM_complex_prediction.py  -f $ff    
done