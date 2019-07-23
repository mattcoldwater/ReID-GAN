import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data
from torch.backends import cudnn
from utils import make_folder

import glob

from torch import multiprocessing as mp

def main(config):
    # For fast training
    cudnn.benchmark = True
    
    print('number class:', config.n_class)

    # Data loader 
    data_loader = Data(config) # Data_Loader(config.train, config.dataset, config.image_path, config.imsize, config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    print('config data_loader and build logs folder')

    if config.train:
        trainer = Trainer(data_loader.train_loader, config)
        trainer.train()
    else:
        tester = Tester(data_loader.test_loader, config)
        tester.test()

if __name__ == '__main__':
    mp.set_start_method('spawn', True) 
    config = get_parameters()
    print(config)
    main(config)