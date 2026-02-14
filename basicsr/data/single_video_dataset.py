import os
import cv2
import glob
import random
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, single_random_crop, paired_random_crop, triplet_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.options import yaml_load


# @DATASET_REGISTRY.register()
class REDSSingleVideoDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSSingleVideoDataset, self).__init__()
        self.opt = opt

        self.gt_root = Path(opt['dataroot_gt'])

        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.gt_root]
            self.io_backend_opt['client_keys'] = ['gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 100 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 100 - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts = single_random_crop(img_gts, gt_size, img_gt_path)

        # augmentation - flip, rotate
        img_gts = torch.stack(img2tensor(img_gts, bgr2rgb=True, float32=True), dim=0)

        # img_gts: (t, c, h, w)
        # key: str
        return {'gts': img_gts, 'keys': key}

    def __len__(self):
        return len(self.keys)


# @DATASET_REGISTRY.register()
class REDSAutoencoderDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_lt (str): Data root path for lt.
        dataroot_ps:(str): Data root path for phasor.
        dataroot_fl (str): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(REDSAutoencoderDataset, self).__init__()
        self.opt = opt

        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq'])
        self.lt_root = Path(opt['dataroot_lt'])
        self.ps_root = Path(opt['dataroot_ps'])
        self.fl_root = Path(opt['dataroot_fl'])

        self.num_frame = opt['num_frame']
        self.load_fix_indices_only = opt['load_fix_indices_only']
        self.seq_num = opt['seq_num']
        self.frame_num = opt['frame_num']
        self.keys = []
        # for j in range(self.seq_num):
        #     self.keys.extend([f'{j:04d}/{i:03d}' for i in range(self.frame_num)])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        elif opt['val_partition'] == 'HATIR':
            if opt['test_mode']:
                for j in range(self.seq_num):
                    self.keys.extend([f'test/{j:04d}/{i:03d}' for i in range(self.frame_num)])
            else:
                for j in range(self.seq_num):
                    self.keys.extend([f'train/{j:04d}/{i:03d}' for i in range(self.frame_num)])
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4', 'HATIR'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0].split('_')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0].split('_')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.sp_root, self.lt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'ps', 'lt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.opt['gt_size']),
            torchvision.transforms.CenterCrop(self.opt['gt_size']),
        ])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > 30 - self.num_frame * interval:
            start_frame_idx = random.randint(0, 30 - self.num_frame * interval)
        if self.load_fix_indices_only:
            start_frame_idx = start_frame_idx - start_frame_idx % self.num_frame
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring frames
        img_lqs = []
        img_gts = []
        img_pss = []
        img_lts = []
        img_fls_fwd = []
        img_fls_bwd = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:03d}'
                img_gt_path = f'{clip_name}/{neighbor:03d}'
                img_ps_path = f'{clip_name}/{neighbor:03d}'
                img_lt_path = f'{clip_name}/{neighbor:03d}'
                img_fls_fwd = f'{clip_name}/{neighbor:03d}'
                img_fls_bwd = f'{clip_name}/{neighbor:03d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:03d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:03d}.png'
                img_ps_path = self.ps_root / clip_name / f'{neighbor:03d}.npy'
                img_lt_path = self.lt_root / clip_name / f'{neighbor:03d}.npy'
                

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            # get phasor
            img_ps = np.load(str(img_ps_path))
            img_ps = torch.from_numpy(img_ps).float()
            img_pss.append(img_ps)

            # get latent
            img_lt = np.load(str(img_lt_path))
            img_lt = torch.from_numpy(img_lt).float()
            img_lts.append(img_lt)
            
            if neighbor != neighbor_list[0]:
                img_fl_fwd_path = self.fl_root / clip_name / f'{neighbor:03d}_fwd.npy'
                img_fl_fwd = np.load(str(img_fl_fwd_path))
                img_fl_fwd = torch.from_numpy(img_fl_fwd).float()
                img_fls_fwd.append(img_fl_fwd)
                
            if neighbor != neighbor_list[-1]:
                img_fl_bwd_path = self.fl_root / clip_name / f'{neighbor:03d}_bwd.npy'
                img_fl_bwd = np.load(str(img_fl_bwd_path))
                img_fl_bwd = torch.from_numpy(img_fl_bwd).float()
                img_fls_bwd.append(img_fl_bwd)
        

        # randomly crop
        # img_gts, img_lqs, img_sps = triplet_random_crop(img_gts, img_lqs, img_sps, gt_size, scale, img_gt_path)

        img_gts = torch.stack(img2tensor(img_gts, bgr2rgb=True, float32=True), dim=0)
        img_lqs = torch.stack(img2tensor(img_lqs, bgr2rgb=True, float32=True), dim=0)
        img_pss = torch.stack(img_pss, dim=0)
        img_lts = torch.stack(img_lts, dim=0)
        img_fls_fwd = torch.stack(img_fls_fwd, dim=0)
        img_fls_bwd = torch.stack(img_fls_bwd, dim=0)
        

        img_lqs = F.interpolate(
            img_lqs,
            size=(img_gts.size(-2),img_gts.size(-1)),
            mode='bicubic'
        )
        
        img_gts = self.transform(img_gts)
        img_lqs = self.transform(img_lqs)
    

        img_fls = (img_fls_fwd, img_fls_bwd)
        

        return {'lqs': img_lqs, 'gts': img_gts, 'pss': img_pss, 'lts': img_lts, 'fls' : img_fls, 'keys': key}

    def __len__(self):
        return len(self.keys)
