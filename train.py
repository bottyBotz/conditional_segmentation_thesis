import os, time
from config.global_train_config import config
import torch
import numpy as np

## For reproducible results    
def seed_all(s):
    np.random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s) 
    torch.manual_seed(s)
    print(f'Seeds set to {s}!')

if not config.using_HPC:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.gpu}"

if __name__ == "__main__":
    # if args.model == "origin":
    #     model = archs.mpMRIRegUnsupervise(args)
    # elif args.model == "weakly":
    #     from src.model.archs.mpMRIRegWeakSupervise import weakSuperVisionMpMRIReg
    #     model = weakSuperVisionMpMRIReg(args)
    # elif args.model == "joint3":
    #     model = archs.joint3(args)

    seed_all(42) #Set random seeds for reproducibility

    if config.project == 'Longitudinal':
        from src.model.archs.longitudinal import LongiReg
        model = LongiReg(config)
    elif config.project == "Icn":
        from src.model.archs.icReg import icReg
    elif config.project == "ConditionalSeg":
        from src.model.archs.condiSeg import condiSeg
        model = condiSeg(config)
    elif config.project == "ConditionalSegReversed":
        from src.model.archs.condiSegReversed import condiSegReversed
        model = condiSegReversed(config)
    elif config.project == "WeakSup":
        from src.model.archs.weakSup import weakSup
        model = weakSup(config)
    elif config.project == "CBCTUnetSeg":
        from src.model.archs.cbctSeg import cbctSeg
        model = cbctSeg(config)
    elif config.project == "mpmrireg":
        from src.model.archs.mpmrireg import mpmrireg
        model = mpmrireg(config)
    else:
        raise NotImplementedError

    if config.continue_epoch != '-1':
        model.load_epoch(config.continue_epoch)

    startTime = time.time()
    model.train()
    print(f'Training {config.project} model took {time.time() - startTime} seconds')

    #Tensorboard
    model.writer.flush() #Write to Disk
    model.writer.close() #Close
    print('Optimization done.')
