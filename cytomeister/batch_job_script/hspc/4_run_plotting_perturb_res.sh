#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 1 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11 # working directory
#BSUB -o TRACE-reproducibility/logs/plt_perturb_res_B2M_%J.out # output file
#BSUB -e TRACE-reproducibility/logs/plt_perturb_res_B2M_%J.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J plt_perturb_res_B2M # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

echo "--- Start plotting"

python3 $cwd/TRACE-reproducibility/HSPC/3.1_multiple_perturbation.py \
--perturbed_gene 'B2M' \
--p_perturbation '/lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/res/hspc/perturbation/20250708-13:57_minference_adata_gB2M_ssrc_tmask.h5ad' \
--logfc_threshold 0.25 \
--perturb_genes_file 'T_perturb/res/hspc/intermediate_rank_genes_groups.csv'

echo "--- Finished plotting"
