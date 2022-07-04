#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=300:0:0
#$ -j y
#$ -N segCTCV0
#$ -cwd
hostname
date
source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.0.source
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name segModeCTCV0 \
--data_path ../../../raid/candi/daniel/Data/others/deepRegData/fullResCropIntensityClip_resampled \
--batch_size 16 \
--input_mode ct \
--gpu 1 \
--inc 1 \
--outc 2 \
--cv 0 \
--input_shape 64 101 91 \
--lr 1e-5 \
--affine_scale 0.15 \
--save_frequency 100 \
--num_epochs 500 \
--w_dce 1.0 \
--using_HPC 0 \
--nc_initial 16 \
--two_stage_sampling 0 \
                   
                   