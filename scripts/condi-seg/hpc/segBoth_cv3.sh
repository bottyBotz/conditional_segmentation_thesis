#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=300:0:0
#$ -j y
#$ -N segBothCV3
#$ -cwd
hostname
date
source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.0.source
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name segModeBothCV3 \
--data_path ./Data/fullResCropIntensityClip_resampled \
--batch_size 8 \
--input_mode both \
--inc 2 \
--outc 2 \
--cv 3 \
--input_shape 64 101 91 \
--lr 1e-5 \
--affine_scale 0.15 \
--save_frequency 100 \
--num_epochs 50000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16 \
--two_stage_sampling 0 \
                   
                   