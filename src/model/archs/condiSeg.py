from src.model.networks.local import CondiSegUNet
from src.model import loss
import src.model.functions as smfunctions
from src.model.archs.baseArch import BaseArch
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats
import cv2, random


class condiSeg(BaseArch):
    def __init__(self, config):
        super(condiSeg, self).__init__(config) #https://www.pythonforbeginners.com/super/working-python-super-function
        self.config = config
        self.net = self.net_parsing()
        self.set_dataloader()
        self.best_metric = 0
        self.writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard_logs'))
        print(f"Writing Tensorboard to {os.path.join(os.getcwd(), 'tensorboard_logs')}")

    def net_parsing(self):
        model = self.config.model
        if model == 'CondiSegUNet':
            net = CondiSegUNet(self.config)
        else:
            raise NotImplementedError
        return net.cuda()

    def set_dataloader(self):
        self.train_set = dataloaders.CBCTData(config=self.config, phase='train')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.config.batch_size,  
            num_workers=4,
            shuffle=True, 
            drop_last=True)
        print('>>> Train set ready.')  
        self.val_set = dataloaders.CBCTData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.CBCTData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False)
        print('>>> Holdout set ready.')

    def get_input(self, input_dict, aug=True):
        """Get input to the CBCT segmentation network

        Args:
            input_dict (ndarray): Fixed and Moving image and label tensors
            aug (bool, optional): Whether to perform data augmentation (generation of perturbed 3D Grid). Defaults to True.

        Raises:
            NotImplementedError: If the input mode is not valid (cbct/ct/oneof/both)

        Returns:
            tensor : Correct type of input tensor combining fixed and moving image and label tensors
        """      
        fx_img, mv_img = input_dict['fx_img'].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
        fx_seg, mv_seg = input_dict['fx_seg'].cuda(), input_dict['mv_seg'].cuda()  # label
        
        # print(f"fx_img shape: {fx_img.shape}")
        # print(f"mv_img shape: {mv_img.shape}")
        # print(f"fx_seg shape: {fx_seg.shape}")
        # print(f"mv_seg shape: {mv_seg.shape}")

        if self.config.two_stage_sampling == 0:
            fx_seg, mv_seg = fx_seg[:, 0, ...], mv_seg[:, 0, ...] #This removes the label dimension from the segmentation

        # print(f"fx_img shape: {fx_img.shape}")
        # print(f"mv_img shape: {mv_img.shape}")
        # print(f"fx_seg shape: {fx_seg.shape}")
        # print(f"mv_seg shape: {mv_seg.shape}")

        #If affine scale set in config, perform affine scaling
        if (self.config.affine_scale != 0.0) and aug:
            mv_affine_grid = smfunctions.rand_affine_grid(
                mv_img, 
                scale=self.config.affine_scale, 
                random_seed=self.config.affine_seed
                )
            fx_affine_grid = smfunctions.rand_affine_grid(
                fx_img, 
                scale=self.config.affine_scale,
                random_seed=self.config.affine_seed
                )
            mv_img = torch.nn.functional.grid_sample(mv_img, mv_affine_grid, mode='bilinear', align_corners=True)
            mv_seg = torch.nn.functional.grid_sample(mv_seg, mv_affine_grid, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, fx_affine_grid, mode='bilinear', align_corners=True)
            fx_seg = torch.nn.functional.grid_sample(fx_seg, fx_affine_grid, mode='bilinear', align_corners=True)
        else:
            pass

        # print("done with aug...")
        # print(f"fx_img shape: {fx_img.shape}")
        # print(f"mv_img shape: {mv_img.shape}")
        # print(f"fx_seg shape: {fx_seg.shape}")
        # print(f"mv_seg shape: {mv_seg.shape}")
        # if self.config.two_stage_sampling == 1:
        #     ret
        if self.config.input_mode == 'condiRev':
            #If training, input is reversed condi-seg
            if self.phase == 'train':
                return mv_img, mv_seg, fx_img, fx_seg
            #At test time, input is normal condiseg
            else:
                return fx_img, fx_seg, mv_img, mv_seg
        return fx_img, fx_seg, mv_img, mv_seg

    def gen_pseudo_data(self):
        
        pseudo_data = []
        for i in range(self.config.batch_size):
            lx, ly, lz = self.config.input_shape
            cx = random.sample(list(range(lx)), 1)[0] 
            cy = random.sample(list(range(ly)), 1)[0]
            cz = random.sample(list(range(lz)), 1)[0]
            rad = 16

            Lx, Rx = max(0, cx-rad), min(lx, cx+rad)  # Left & Right x
            Ly, Ry = max(0, cy-rad), min(ly, cy+rad)  # Left & Right y
            Lz, Rz = max(0, cz-rad), min(lz, cz+rad)  # Left & Right z
            
            seg_arr = torch.zeros(self.config.input_shape)
            seg_arr = seg_arr[None, ...]  # add channel dim
            seg_arr[:, Lx:Rx, Ly:Ry, Lz:Rz] = 1.0
            pseudo_data.append(seg_arr)

        pseudo_data = torch.stack(pseudo_data, dim=0)
        return pseudo_data.cuda()
    
    @torch.no_grad()
    def forward_pseudo_data(self, pseudo_input):
        self.net.eval()
        pseudo_out = self.net(torch.cat(pseudo_input, dim=1))
        pseudo_out = (pseudo_out >= 0.5) * 1.0
        return pseudo_out

    def train(self):
        """
        Training and validation loop for the model. Writes to tensorboard and logs directory for the current folds training and validation. 
        Performs inference on the holdout set after training.
        """        
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, weight_decay=1e-6)
        log_dir = f"./logs/{self.config.project}"
        log_save_path = os.path.join(log_dir, 'running_logs')
        os.makedirs(log_save_path, exist_ok=True)
        with open(os.path.join(log_save_path, f'{self.config.exp_name}_train_log_{self.config.cv}.txt'), 'w') as f:
            f.write(f"project,exp_name,fold,train_val_test,epoch,value,value_type\n")
            for self.epoch in range(1, self.config.num_epochs + 1):
                self.net.train()

                print('-' * 10, f'Train epoch_{self.epoch}', '-' * 10)
                for self.step, input_dict in enumerate(self.train_loader):
                    fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict) 

                    optimizer.zero_grad() # clear gradients
                    # print(f"fx_img shape: {fx_img.shape}")
                    # print(f"mv_img shape: {mv_img.shape}")
                    # print(f"fx_seg shape: {fx_seg.shape}")
                    # print(f"mv_seg shape: {mv_seg.shape}")
                    # print(f"torch.cat([fx_img, mv_img, mv_seg], dim=1).shape: {torch.cat([fx_img, mv_img, mv_seg], dim=1).shape}")

                    pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg], dim=1)) # forward pass

                    global_loss = self.loss(pred_seg, fx_seg) # compute loss

                    global_loss.backward() # backward pass
                    optimizer.step() #Gradient Descent

                    if self.config.use_pseudo_label:
                        print("in pseudo training....")
                        # generate pseudo data
                        pseudo_input = self.gen_pseudo_data()
                        pseudo_out = self.forward_pseudo_data([fx_img, mv_img, pseudo_input])
                        pseudo_label = pseudo_out.detach()

                        # use pseudo data to train another round
                        self.net.train()
                        optimizer.zero_grad()
                        pred_seg = self.net(torch.cat([fx_img, mv_img, pseudo_input], dim=1))
                        global_loss = self.loss(pred_seg, pseudo_label)
                        global_loss.backward()
                        optimizer.step()

                self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/Loss/train", global_loss, self.epoch) #Write Loss for Epoch to Tensorboard
                f.write(f"{self.config.project},{self.config.exp_name},{self.config.cv},'train',{self.epoch},{global_loss},'loss'\n") # Write Loss for Epoch to Log File
                
                #Save the model at periodic frequencies
                if self.epoch % self.config.save_frequency == 0:
                    self.save()
                print('-' * 10, 'validation', '-' * 10) 
                
                #Run the validation step
                self.validation(f) 

        self.writer.add_graph(self.net, torch.cat([fx_img, mv_img, mv_seg], dim=1)) #Save Network Graph to Tensorboard

        #Run Holdout Test Set for this fold.
        self.inference()

    def loss(self, pred_seg, fx_seg):
        L_All = 0
        Info = f'step {self.step}'
        
        #Weighted DICE
        if self.config.w_dce != 0: 
            L_dice = loss.single_scale_dice(fx_seg, pred_seg) * self.config.w_dce
            L_All += L_dice
            Info += f', Loss_dice: {L_dice:.3f}'
        
        #Binary Cross Entropy
        if self.config.w_bce != 0:
            L_BCE = loss.wBCE(pred_seg, fx_seg, weights=self.config.class_weights)
            L_All += L_BCE
            Info += f', Loss_wBCE: {L_BCE:.3f}'

        Info += f', Loss_All: {L_All:.3f}'

        print(Info)
        return L_All

    @torch.no_grad() #No Gradient Computation for validation step
    def validation(self, f = None):
        self.net.eval() # Set model to validation/evaluation Mode
        # visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-in-val')
        # os.makedirs(visualization_path, exist_ok=True)

        res = []
        #Iterate through the validation dataloader
        for idx, input_dict in enumerate(self.val_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            # fx Size([1, 1, 2, 152, 269, 121]) mv Size([1, 1, 2, 152, 269, 121])

            # self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            # self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            for label_idx in range(fx_seg.shape[2]):
                pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg[:, :, label_idx, ...]], dim=1))
                binary_dice = loss.binary_dice(pred_seg, fx_seg[:, :, label_idx, ...])
                subject = input_dict['subject']
                self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/binary_dice/validation/subject_{subject}/label_idx_{label_idx}", binary_dice, self.epoch)

                print(f'subject:{subject}', f'label_idx:{label_idx}', f'DICE:{binary_dice:.3f}')
                res.append(binary_dice)

                # self.save_img(fx_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-fx_img_{label_idx}.nii'))
                # self.save_img(mv_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-mv_img_{label_idx}.nii'))
                # self.save_img(pred_seg[0], os.path.join(visualization_path, f'{idx+1}-pred_img_{label_idx}.nii'))

        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)

        self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/Dice_Mean/validation", mean, self.epoch) #Write Dice Mean for Epoch to Tensorboard
        self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/Dice_Std/validation", std, self.epoch) #Write Dice Std for Epoch to Tensorboard
        f.write(f"{self.config.project},{self.config.exp_name},{self.config.cv},'val',{self.epoch},{mean},'dice_mean'\n") # Write Dice Mean for Epoch to Log File
        f.write(f"{self.config.project},{self.config.exp_name},{self.config.cv},'val',{self.epoch},{std},'dice_std'\n") # Write Dice Std for Epoch to Log File

        #Save the best model as it's performance on the validation set
        if mean > self.best_metric:
            self.best_metric = mean
            print('better model found.')
            self.save(type='best')
        print('Dice:', mean, std)

    @torch.no_grad() #No Gradient Computation for inference step
    def inference(self):
        self.net.eval() #set model to test/eval mode
        visualization_path = os.path.join(self.log_dir, f'{self.config.exp_name}-vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)

        results = {
            'dice': [],
            'dice-wo-reg': [],
            }

        #Iterate through the test data loader
        for idx, input_dict in enumerate(self.test_loader):
            fx_img, fx_seg, mv_img, mv_seg = self.get_input(input_dict, aug=False)
            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))

            label_num = fx_seg.shape[2]
            for label_idx in range(fx_seg.shape[2]):
                pred_seg = self.net(torch.cat([fx_img, mv_img, mv_seg[:, :, label_idx, ...]], dim=1)) #Forward Pass

                aft_dice = loss.binary_dice(pred_seg, fx_seg[:, :, label_idx, ...]).cpu().numpy()
                bef_dice = loss.binary_dice(fx_seg[:, :, label_idx, ...], mv_seg[:, :, label_idx, ...]).cpu().numpy()

                subject = input_dict['subject']
                results['dice'].append(aft_dice)
                results['dice-wo-reg'].append(bef_dice)
                #self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/aft_dice/inference/subject_{subject}/label_idx_{label_idx}", results['dice'][-1], self.epoch) #Write Dice for subject and organ to Tensorboard
                #self.writer.add_scalar(f"{self.config.project}/{self.config.exp_name}/dice-wo-reg/inference/subject_{subject}/label_idx_{label_idx}", results['dice-wo-reg'][-1], self.epoch) #Write Dice for subject and organ to Tensorboard

                print(f'subject:{subject}', f'label_idx:{label_idx}', f'BEF-DICE:{bef_dice:.3f}', f'AFT-DICE:{aft_dice:.3f}')

                self.save_img(fx_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-fx_img_{label_idx}.nii'))
                self.save_img(mv_seg[:, :, label_idx, ...], os.path.join(visualization_path, f'{idx+1}-mv_img_{label_idx}.nii'))
                self.save_img(pred_seg[0], os.path.join(visualization_path, f'{idx+1}-pred_img_{label_idx}.nii'))

                anatomical_list = ['bladder', 'rectum']
                self.vis_with_contour(
                    fx_img=fx_img[0, 0].cpu().numpy(), 
                    fx_seg=fx_seg[:, :, label_idx, ...][0, 0].cpu().numpy(), 
                    mv_img=mv_img[0, 0].cpu().numpy(), 
                    mv_seg=mv_seg[:, :, label_idx, ...][0, 0].cpu().numpy(), 
                    pred_seg=pred_seg[0, 0].cpu().numpy(), 
                    save_folder=os.path.join(visualization_path, 'vis_png', subject[0]),
                    color=(255, 0, 0), 
                    prefix=f'DSC_{anatomical_list[label_idx]}_bef_{bef_dice:.3f}_after_{aft_dice:.3f}',
                    suffix=''
                    )

            print('-' * 20)

        for k, v in results.items():
            print(f'overall {k}, {np.mean(v):.3f}, {np.std(v):.3f}')
            if 'dice' in k or 'cd' in k:
                for idx in range(label_num):
                    tmp = v[idx::label_num]
                    print(f'results of {k} on label {idx}:, {np.mean(tmp):.3f} +- {np.std(tmp):.3f}')        

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)


    @staticmethod
    def vis_with_contour(fx_img, fx_seg, mv_img, mv_seg, pred_seg, save_folder, color=(255, 255, 0), prefix='', suffix=''):
        """fx/mv_img/seg -> 3d volume"""
        def normalize0255(arr):
            return (arr - arr.min())*255.0 / (arr.max() - arr.min())

        def add_contours(t2, label, color):
            if len(t2.shape) != 3:
                _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
            else:
                _t2 = t2
            
            _t2 = normalize0255(_t2).astype('uint8')
            _label = label.astype('uint8')
            blank = np.zeros(_t2.shape)
            contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
            tmp = _t2.copy()  # ?????
            cv2.drawContours(tmp, contours, -1, color, 1)
            return tmp

        img_set = np.concatenate([mv_img, fx_img, fx_img], axis=0)
        img_set = normalize0255(img_set)
        seg_set = np.concatenate([mv_seg, fx_seg, pred_seg], axis=0)
        
        for z in range(fx_img.shape[-1]):
            img_slice = img_set[..., z]
            seg_slice = seg_set[..., z]
            contoured_slice = add_contours(img_slice, seg_slice, color=color)
            os.makedirs(save_folder, exist_ok=True)

            dst_img = np.transpose(contoured_slice, (1,0,2))[::-1, ...]
            # print(np.array(dst_img.shape[:2])*3)
            cv2.imwrite(
                os.path.join(save_folder, f"{prefix}_{z}_{suffix}.png"), 
                # dst_img
                cv2.resize(dst_img, (dst_img.shape[1]*3, dst_img.shape[0]*3))
                )