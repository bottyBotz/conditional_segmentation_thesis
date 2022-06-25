#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=100:0:0
#$ -j y
#$ -N CS_TEST_cv1_nc16
#$ -cwd
hostname
date
python3 -u train.py \
--project ConditionalSegTest \
--exp_name CondisegCBCT_cv0_Test_hpc_nc16 \
--data_path ../../../Data/fullResCropIntensityClip_resampled \
--batch_size 16 \
--cv 0 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 500 \
--num_epochs 5000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16
                   