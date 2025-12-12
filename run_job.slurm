#!/bin/bash
#SBATCH --job-name=job_wgpu

#SBATCH --account=ds_ga_1011_001-2025fa

#SBATCH --partition=c12m85-a100-1

#SBATCH --open-mode=append

#SBATCH --output=./%j_%x.out

#SBATCH --error=./%j_%x.err

#SBATCH --export=ALL

#SBATCH --time=00:10:00

#SBATCH --gres=gpu:1

#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay /scratch/hy2902/overlay-25GB-500K.ext3:rw /scratch/hy2902/ubuntu-20.04.3.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh

conda activate nlp_env
  cd /scratch/hy2902/CVHw3/moved-object-detection-detr
  python train_detr_moved_objects.py \
  --metadata-csv cv_data_hw2/metadata/pairs.csv \
  --regime full \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --train-only-moved \
  --target-long-side 800 \
  --run-name fullrun2
  python train_detr_moved_objects.py \
  --metadata-csv cv_data_hw2/metadata/pairs.csv \
  --regime backbone \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --train-only-moved \
  --target-long-side 800 \
  --run-name backbone_finetune
  python train_detr_moved_objects.py \
  --metadata-csv cv_data_hw2/metadata/pairs.csv \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --train-only-moved \
  --target-long-side 800 \
  --run-name head_finetune
  python train_detr_moved_objects.py \
  --metadata-csv cv_data_hw2/metadata/pairs.csv \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --train-only-moved \
  --target-long-side 800 \
  --run-name transformer_finetune
"
