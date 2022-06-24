#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=300:0:0
#$ -j y
#$ -N CS_cv1_nc16
#$ -cwd
hostname
date
python3 -u train.py \
--project ConditionalSeg \
--exp_name CondisegCBCT_cv2_nc16 \
--data_path ../../../raid/candi/daniel/Data/others/deepRegData/fullResCropIntensityClip_resampled \
--batch_size 16 \
--cv 2 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 500 \
--num_epochs 10000 \
--w_dce 1.0 \
--using_HPC 0 \
--nc_initial 16
                   