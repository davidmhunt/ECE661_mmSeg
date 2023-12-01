import sys
mmseg_path = "../../submodules/mmsegmentation/"
sys.path.append(mmseg_path)

#check mmseg installation
import os
import torch, torchvision
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmseg
from mmseg.utils import register_all_modules
from mmseg.apis import inference_model, init_model

from mmengine import Config
from mmengine.runner import Runner

#check versions of mmseg, mmcv, and mmsegmentation
print('Pytorch Version', torch.__version__)
print('CUDA available',torch.cuda.is_available())

print('MMCV Version', mmcv.__version__)
print('CUDA Version', get_compiling_cuda_version())
print('Compiler Version', get_compiler_version())

print('mmsegmentation Version', mmseg.__version__)

#create a dictionary of configurations and their respective checkpoints
configs = [
    {
        "config_name":"unet-s5-d16_deeplabv3_4xb4-40k_DME-64x64.py",
        "checkpoint":"deeplabv3_unet_s5-d16_64x64_40k_drive_20201226_094047-0671ff20.pth"
    },
    {
        "config_name":"unet-s5-d16_deeplabv3_4xb4-ce-1.0-dice-3.0-40k_DME-64x64.py",
        "checkpoint":"deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201825-6bf0efd7.pth"
    },
    {
        "config_name": "unet-s5-d16_fcn_4xb4-40k_DME-64x64.py",
        "checkpoint": "fcn_unet_s5-d16_64x64_40k_drive_20201223_191051-5daf6d3b.pth"
    },
    {
        "config_name": "unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_DME-64x64.py",
        "checkpoint":"fcn_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201820-785de5c2.pth"
    },
    {
        "config_name":"unet-s5-d16_pspnet_4xb4-40k_DME-64x64.py",
        "checkpoint":"pspnet_unet_s5-d16_64x64_40k_drive_20201227_181818-aac73387.pth"
    },
    {
        "config_name":"unet-s5-d16_pspnet_4xb4-ce-1.0-dice-3.0-40k_DME-64x64.py",
        "checkpoint":"pspnet_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201821-22b3e3ba.pth"
    }
]

#initialize the configuration file
config_idx = 3
config_folder = "ece661"
config_name = configs[config_idx]["config_name"]
cfg_path = os.path.join(mmseg_path,"configs",config_folder,config_name)
cfg = Config.fromfile(cfg_path)

#set the working directory
cfg.work_dir = os.path.join(mmseg_path,'work_dirs',config_name)

#set the checkpoint to load from
checkpoint_name = configs[config_idx]["checkpoint"]
checkpoint_path = os.path.join(mmseg_path,'checkpoints',checkpoint_name)
cfg.load_from = checkpoint_path

#increase the batch size for the training
#NOTE: Config 0 has issue when we change the batch size for some reason
cfg.train_dataloader.batch_size = 16

#set the number of training iterations
cfg.train_cfg.max_iters = 8000

#initialize the runner
runner = Runner.from_cfg(cfg)

runner.train()

