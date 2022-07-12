import os, time
from config.global_train_config import config
import torch
import torch.optim as optim
import numpy as np
import random



from src.model.archs.condiSeg import condiSeg


model = condiSeg(config) 


#model.get_input()
optimizer = optim.Adam(model.net.parameters(), lr=model.config.lr, weight_decay=1e-6)
for epoch in range(1, model.config.num_epochs + 1):
    for model.step, input_dict in enumerate(model.train_loader):
        #print(f"input_dict: {input_dict}")
        # for key in input_dict.keys():
        #     print(f"key: {key}")
        # for key, value in input_dict.items():
        #     print(f"{key}: {value}")
        fx_img, mv_img = input_dict['fx_img'].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()  # label
        print(f"fx_img shape: {fx_img.shape}")
        print(f"mv_img shape: {mv_img.shape}")
        print(f"fx_seg shape: {fx_seg.shape}")
        print(f"mv_seg shape: {mv_seg.shape}")
        print('mv_seg.shape[0]: ', mv_seg.shape[0])
        print('mv_seg.shape[0] - 1: ', mv_seg.shape[0] - 1)
        random_label_index = random.randint(0, mv_seg.shape[0]-1) #pick a random example in this batch
        moving_label, fixed_label = mv_seg[random_label_index], fx_seg[random_label_index] #
        print("random_label_index", random_label_index)
        print(f"moving_label shape: {moving_label.shape}")
        print(f"fixed_label shape: {fixed_label.shape}")
        print(f"mv_seg shape: {mv_seg.shape}")
        print(f"mv_seg[random_label_index]: {mv_seg[random_label_index].shape}")
        print(f"torch.FloatTensor(moving_image[None, ...]: {torch.FloatTensor(mv_img[None, ...])})")
        break
        model.net.zero_grad()
        output = model.net(input_dict)
        loss = model.loss(output, input_dict)
        loss.backward()
        optimizer.step()
        if model.step % model.config.print_freq == 0:
            print(f'Epoch: {epoch}, Step: {model.step}, Loss: {loss.item():.4f}')
    break

#python3 scratchpad.py --project ConditionalSeg --exp_name test_exp  --model CondiSegUNet --cv 0 --patient_cohort intra --crop_on_seg_aug 0 --use_pseudo_label 0 --w_dce 1.0 --w_bce 1 --input_shape 64 101 91 --data_path ./Data/fullResCropIntensityClip_resampled --batch_size 8 --using_HPC 0 --two_stage_sampling 0