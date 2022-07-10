#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=75:0:0
#$ -j y
#$ -N CS_cv5_nc16
#$ -cwd
hostname
date
source /share/apps/source_files/python/python-3.9.5.source
source /share/apps/source_files/cuda/cuda-11.0.source
python3 -u train.py \
--project ConditionalSeg \
--exp_name CondisegCBCT_cv5_nc16 \
--data_path ./Data/fullResCropIntensityClip_resampled \
--batch_size 8 \
--cv 5 \
--input_shape 64 101 91 \
--lr e-5 \
--two_stage_sampling 0 \
--affine_scale 0.15 \
--save_frequency 1000 \
--num_epochs 50000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16
                   