#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=96:0:0
#$ -j y
#$ -N segMCTCV0
#$ -cwd
hostname
date
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name segModeBothCV2 \
--data_path ../../../raid/candi/daniel/Data/others/deepRegData/fullResCropIntensityClip_resampled \
--batch_size 16 \
--input_mode both \
--inc 2 \
--outc 2 \
--gpu 1 \
--cv 2 \
--input_shape 64 101 91 \
--lr 1e-5 \
--affine_scale 0.15 \
--save_frequency 100 \
--num_epochs 10000 \
--w_dce 1.0 \
--using_HPC 0 \
--nc_initial 16 \
--two_stage_sampling 0 \
                   
                   