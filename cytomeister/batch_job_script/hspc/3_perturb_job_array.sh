#!/bin/bash
#BSUB -q gpu-normal # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1:gmodel=NVIDIAA100_SXM4_80GB" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -J hspc_perturb_cluster_[1-81]%4 # job array with 35 jobs, max 4 running at the same time
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing team361
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/ # working directory
#BSUB -o T_perturb/cytomeister/logs/perturb_cluster_mTORC_%J_%I.out # output file
#BSUB -e T_perturb/cytomeister/logs/perturb_cluster_mTORC_%J_%I.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

GENES=(
  SERPINH1
  HSPD1
  SCD
  HSPA4
  CDC25A
  HSPE1
  STIP1
  GGA2
  CACYBP
  HK2
  PSMB5
  CCT6A
  ELOVL6
  HMGCS1
  MTHFD2
  ACSL3
  ADIPOR2
  NUP205
  MCM4
  IMMT
  PDK1
  BCAT1
  HMGCR
  RRM2
  PITPNB
  P4HA1
  DHCR24
  FADS2
  UCHL5
  SQLE
  SLC2A3
  GMPS
  POLR3G
  PSMD12
  NFYC
  SC5D
  PNO1
  FADS1
  ACTR2
  AURKA
  GSR
  TPI1
  EEF1E1
  ACTR3
  RRP9
  LDLR
  GTF2H1
  CTSC
  FDXR
  IDH1
  DHFR
  PLK1
  PSMC4
  CYP51A1
  TUBG1
  PSMA3
  TXNRD1
  BUB1
  GOT1
  SLC7A5
  SLC1A5
  TBK1
  PSME3
  ABCF2
  ATP2A2
  ACACA
  CANX
  AK4
  IDI1
  LDHA
  SLC2A1
  TFRC
  PSMD14
  PPP1R15A
  DDX39A
  PSPH
  PSMC6
  STARD4
  EIF2S2
  CCNF
)



INDEX=$((LSB_JOBINDEX - 1))
PERTURBED_GENE=${GENES[$INDEX]}

echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene to be perturbed: $PERTURBED_GENE"

CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_inference_perturbation_jobarray.yaml"
TMP_CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_generate_perturbation_${PERTURBED_GENE}_${LSB_JOBID}.yaml"

sed "s/{{PERTURBED_GENE}}/${PERTURBED_GENE}/g" $CONFIG_PATH > $TMP_CONFIG_PATH

if [[ $INDEX -ge ${#GENES[@]} ]]; then
  echo "Error: LSB_JOBINDEX ($LSB_JOBINDEX) out of range for GENES array" >&2
  exit 1
fi

# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/Perturb/val.py \
--config $TMP_CONFIG_PATH 

# Clean up temporary config file
trap "rm -f $TMP_CONFIG_PATH" EXIT
