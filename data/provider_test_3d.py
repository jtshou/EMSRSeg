from emsrseg.temporal import context_sections
import os
import cv2
import h5py
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .utils.seg_util import mknhood3d, genSegMalis
from .utils.aff_util import seg_to_affgraph
from .data.data_segmentation import seg_widen_border, weight_binary_ratio
from .data.data_affinity import seg_to_aff
from .utils.affinity_ours import gen_affs_mutex_3d
import torch


class Provider_valid(Dataset):
    def __init__(self, cfg, scale,test_num,num_z=9, test=False, stage=None):
        # basic settings
        self.cfg = cfg
        self.if_dilate = cfg['if_dilate']  # True
        self.shift_channels = cfg['shift_channels']  # None
        self.output_nc = cfg['output_nc']
        self.separate_weight = cfg['separate_weight']  # False
        self.if_sparse = cfg['if_sparse']  # False
        self.num_z = num_z
        self.test = test
        self.stage = stage
        self.scale = scale
        valid_dataset_name = cfg['dataset_name']

        # basic settings
        # the input size of network
        #self.crop_size = [9, 160, 160]

        # training dataset files (h5), may contain many datasets
        if cfg['dataset_name'] == 'cremiA':
            self.train_datasets = ['sample_A_20160501_inputs.h5']
            self.train_labels = ['sample_A_20160501_labels.h5']
        elif cfg['dataset_name'] == 'cremiB':
            self.train_datasets = ['sample_B_20160501_inputs.h5']
            self.train_labels = ['sample_B_20160501_labels.h5']
        elif cfg['dataset_name'] == 'cremiC':
            self.train_datasets = ['sample_C_20160501_inputs.h5']
            self.train_labels = ['sample_C_20160501_labels.h5']

        elif cfg['dataset_name'] == 'cremi-all':
            self.train_datasets = ['cremiA_inputs.h5', 'cremiB_inputs.h5', 'cremiC_inputs.h5']
            self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']

        elif cfg['dataset_name'] == 'ac3':
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']

        elif cfg['dataset_name'] == 'ac4':
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']


        elif cfg['dataset_name'] == 'wafer':
            self.train_datasets = [['wafer4_inputs.h5', 'wafer36_2_inputs.h5'][test_num]]
            self.train_labels = [['wafer4_labels.h5', 'wafer36_2_labels.h5'][test_num]]
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name_gt = cfg['data_folder_gt']
        self.folder_name_lr = cfg['data_folder_lr']
        assert len(self.train_datasets) == len(self.train_labels)

        self.test_split = cfg["test_split"]

        # load dataset
        self.dataset = []
        self.dataset_hr = []
        self.labels = []
        self.labels_origin = []

        for k in range(len(self.train_datasets)):
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name_lr, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()

            if valid_dataset_name == 'ac3':
                data = data[:self.test_split]
            else:
                data = data[-self.test_split:]
            self.dataset.append(data)

            f_raw = h5py.File(os.path.join(self.folder_name_gt, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()

            if valid_dataset_name == 'ac3':
                data = data[:self.test_split]
            else:
                data = data[-self.test_split:]
            self.dataset_hr.append(data.astype(np.float32) / 255.0)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name_gt, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()

            if valid_dataset_name == 'ac3':
                label = label[:self.test_split]
            else:
                label = label[-self.test_split:]
            self.labels_origin.append(label.copy())
            if self.if_dilate:
                label = seg_widen_border(label, tsz_h=1)
            self.labels.append(label)
        self.origin_data_shape = list(self.dataset[0].shape)

        # generate gt affinity
        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k]
            self.gt_affs.append(seg_to_aff(temp).astype(np.float32))

        if 'fib' in valid_dataset_name:
            padding_xy = 20
            num_xy = 6
        else:  
            # padding_xy = 48
            # num_xy = 13
            if self.scale == 8:
                #padding_xy = 8
                padding_xy = 0
                num_xy = 1
            if self.scale == 16:
                padding_xy = 0
                num_xy = 1
            if self.scale == 4:
                padding_xy = 0
                num_xy = 1
            if self.scale == 1:
                padding_xy = 64
                num_xy = 1

        if self.dataset[0].shape[0] == 200:
            self.stride = [10, 80, 80]
            self.valid_padding = [4, padding_xy, padding_xy]
            self.num_zyx = [20, num_xy, num_xy]
        elif self.dataset[0].shape[0] == 125:
            #self.stride = [5, 80, 80]
            #self.valid_padding = [4, padding_xy, padding_xy]
            #self.num_zyx = [26, num_xy, num_xy]
            self.stride = [5, 80, 80]
            self.valid_padding = [1, padding_xy, padding_xy]
            self.num_zyx = [25, num_xy, num_xy]
        elif self.dataset[0].shape[0] == 100:
            self.stride = [10, 80, 80]
            self.valid_padding = [4, padding_xy, padding_xy]
            self.num_zyx = [10, num_xy, num_xy]
        elif self.dataset[0].shape[0] == 50:
            self.stride = [5, 80, 80]
            self.valid_padding = [1, padding_xy, padding_xy]
            self.num_zyx = [10, num_xy, num_xy]
        elif self.dataset[0].shape[0] == 25:
            # for rapid inference
            self.stride = [5, 80, 80]
            self.valid_padding = [1, padding_xy, padding_xy]
            self.num_zyx = [5, num_xy, num_xy]
        elif self.dataset[0].shape[0] == 20:
            self.stride = [10, 80, 80]
            self.valid_padding = [4, padding_xy, padding_xy]
            self.num_zyx = [2, num_xy, num_xy]
        else:
            raise NotImplementedError

        # only for superhuman and the num-z = 10
        if self.num_z < 9:
            raise NotImplementedError

        self.crop_size = [9, 1024 // self.scale + self.valid_padding[1] * 2,
                              1024 // self.scale + self.valid_padding[2] * 2]
        self.out_size = self.crop_size


        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                       (self.valid_padding[1], self.valid_padding[1]), \
                                                       (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
            self.dataset_hr[k] = np.pad(self.dataset_hr[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                             (self.valid_padding[1] * self.scale,
                                                              self.valid_padding[1] * self.scale), \
                                                             (self.valid_padding[2] * self.scale,
                                                              self.valid_padding[2] * self.scale)), mode='reflect')
            self.labels[k] = np.pad(self.labels[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                     (self.valid_padding[1] * self.scale,
                                                      self.valid_padding[1] * self.scale), \
                                                     (self.valid_padding[2] * self.scale,
                                                      self.valid_padding[2] * self.scale)), mode='reflect')

        self.raw_data_shape = list(self.dataset[0].shape)
        self.label_shape = list(self.labels[0].shape)

        self.reset_output()
        self.weight_vol_hr = self.get_weight_hr()

        # the number of inference times
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        pos_data = index // self.num_per_dataset
        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
            if fromy < 0:
                fromy = 0
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
            if fromx < 0:
                fromx = 0
        self.pos = [fromz, fromy, fromx]
        hr_imgs = self.dataset_hr[pos_data][fromz:endz, fromx * self.scale:endx * self.scale,
                  fromy * self.scale:endy * self.scale].copy()
        lb = self.labels[pos_data][fromz:endz, fromx * self.scale:endx * self.scale,
             fromy * self.scale:endy * self.scale].copy()
        fromz = fromz - 2
        endz = endz + 2
        Z_max = self.raw_data_shape[0]
        if fromz < 0:
            pad_front = -fromz  
            fromz = 0
        else:
            pad_front = 0
        if endz > Z_max:
            pad_back = endz - Z_max  
            endz = Z_max
        else:
            pad_back = 0
        
        lr_imgs = self.dataset[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()


        
        if pad_front > 0:
            lr_front = self.dataset[pos_data][0:1, fromx:endx, fromy:endy].repeat(pad_front, axis=0)
            lr_imgs = np.concatenate([lr_front, lr_imgs], axis=0)
        if pad_back > 0:
            lr_back = self.dataset[pos_data][-1:, fromx:endx, fromy:endy].repeat(pad_back, axis=0)
            lr_imgs = np.concatenate([lr_imgs, lr_back], axis=0)


        if self.shift_channels is None:
            if self.output_nc == 3:
                lb_affs = seg_to_aff(lb).astype(np.float32)
            elif self.output_nc == 12:
                nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
                nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
                nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
                label111 = seg_to_aff(lb, pad='').astype(np.float32)
                label233 = seg_to_aff(lb, nhood233, pad='')
                label399 = seg_to_aff(lb, nhood399, pad='')
                label427 = seg_to_aff(lb, nhood427, pad='')
                lb_affs = np.concatenate((label111, label233, label399, label427), axis=0)
            else:
                raise NotImplementedError
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        if self.separate_weight:
            weightmap = np.zeros_like(lb_affs)
            for i in range(lb_affs.shape[0]):
                weightmap[i] = weight_binary_ratio(lb_affs[i])
        else:
            weightmap = weight_binary_ratio(lb_affs)

        lr_imgs = lr_imgs.astype(np.float32) / 255.0
        lr_imgs = lr_imgs[np.newaxis, ...]
        lr_imgs = torch.from_numpy(np.ascontiguousarray(lr_imgs, dtype=np.float32)).contiguous()
        hr_imgs = hr_imgs.astype(np.float32)
        hr_imgs = hr_imgs[np.newaxis, ...]
        hr_imgs = torch.from_numpy(np.ascontiguousarray(hr_imgs, dtype=np.float32)).contiguous()
        lb_affs = torch.from_numpy(np.ascontiguousarray(lb_affs, dtype=np.float32)).contiguous()
        weightmap = torch.from_numpy(np.ascontiguousarray(weightmap, dtype=np.float32)).contiguous()
        return {"lr": context_sections(lr_imgs), "hr": hr_imgs, "lb_affs": lb_affs, "weightmap": weightmap}

    def __len__(self):
        return self.iters_num

    def reset_output(self, default_c=None):
        if default_c is None:
            if self.shift_channels is None:
                channel = self.output_nc
            else:
                channel = len(self.shift_channels)
        else:
            channel = default_c
        self.out_affs = np.zeros(tuple([3] + self.label_shape), dtype=np.float32)
        self.out_sr = np.zeros(tuple(self.label_shape), dtype=np.float32)
        self.weight_map = np.zeros(tuple([3] + self.label_shape), dtype=np.float32)
        self.weight_map_sr = np.zeros(tuple(self.label_shape), dtype=np.float32)

    def get_weight(self, sigma=0.2, mu=0.0):
        if self.num_z < 9:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, 9, dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        else:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def get_weight_hr(self, sigma=0.2, mu=0.0):
        if self.num_z < 9:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, 9, dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[1] * self.scale, dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[2] * self.scale, dtype=np.float32), indexing='ij')
        else:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[1] * self.scale, dtype=np.float32),
                                     np.linspace(-1, 1, self.out_size[2] * self.scale, dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = np.repeat(weight[np.newaxis, ...], 3, axis=0)
        return weight

    def add_vol(self, affs_vol, sr_vol=None):
        fromz, fromy, fromx = self.pos
        if self.num_z < 9:
            raise NotImplementedError
        self.out_affs[:, fromz + 1:fromz + self.out_size[0], \
        fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
        fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += affs_vol[:, 1:] * self.weight_vol_hr[:, 1:]
        self.out_affs[1:3, fromz, \
        fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
        fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += affs_vol[1:3, 0] * self.weight_vol_hr[1:3, 0]
        self.w_sr = False
        if sr_vol is not None:
            self.w_sr = True
            for z_idx in range(fromz, fromz + self.out_size[0]):
                if np.sum(self.weight_map_sr[z_idx, fromx * self.scale:(fromx + self.out_size[1]) * self.scale,
                          fromy * self.scale:(fromy + self.out_size[2]) * self.scale]) == 0:
                    self.out_sr[z_idx, \
                    fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
                    fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += sr_vol[z_idx - fromz]
                    self.weight_map_sr[z_idx, \
                    fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
                    fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += 1

        self.weight_map[:, fromz + 1:fromz + self.out_size[0], \
        fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
        fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += self.weight_vol_hr[:, 1:]

        self.weight_map[1:3, fromz, \
        fromx * self.scale:(fromx + self.out_size[1]) * self.scale, \
        fromy * self.scale:(fromy + self.out_size[2]) * self.scale] += self.weight_vol_hr[1:3, 0]

    def get_results(self):
        if self.valid_padding[1] == 0:
            self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0]] / self.weight_map[:,
                                                                                             self.valid_padding[0]:-
                                                                                             self.valid_padding[0]]
            self.out_sr = self.out_sr[self.valid_padding[0]:-self.valid_padding[0]]
        else:
            self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0], \
                            self.valid_padding[1] * self.scale:-self.valid_padding[1] * self.scale, \
                            self.valid_padding[2] * self.scale:-self.valid_padding[2] * self.scale] / self.weight_map[:,
                                                                                                      self.valid_padding[
                                                                                                          0]:-
                                                                                                      self.valid_padding[
                                                                                                          0], \
                                                                                                      self.valid_padding[
                                                                                                          1] * self.scale:-
                                                                                                                          self.valid_padding[
                                                                                                                              1] * self.scale, \
                                                                                                      self.valid_padding[
                                                                                                          2] * self.scale:-
                                                                                                                          self.valid_padding[
                                                                                                                              2] * self.scale]
            self.out_sr = self.out_sr[self.valid_padding[0]:-self.valid_padding[0], \
                          self.valid_padding[1] * self.scale:-self.valid_padding[1] * self.scale, \
                          self.valid_padding[2] * self.scale:-self.valid_padding[2] * self.scale]



        if self.w_sr:
            return self.out_affs, self.out_sr
        return self.out_affs

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data]

    def get_gt_lb(self, num_data=0):
        # lbs = self.labels_origin[num_data].copy()
        # lbs = lbs[self.valid_padding[0]:-self.valid_padding[0], \
        #           self.valid_padding[1]:-self.valid_padding[1], \
        #           self.valid_padding[2]:-self.valid_padding[2]]
        return self.labels_origin[num_data]

    def get_lr_data(self, num_data=0):
        out = self.dataset[num_data].copy()
        if self.valid_padding[1] == 0:
            return out[self.valid_padding[0]:-self.valid_padding[0]]
        else:
            return out[self.valid_padding[0]:-self.valid_padding[0], \
               self.valid_padding[1]:-self.valid_padding[1], \
               self.valid_padding[2]:-self.valid_padding[2]]

    def get_hr_data(self, num_data=0):
        out = self.dataset_hr[num_data].copy()
        if self.valid_padding[1] == 0:
            return out[self.valid_padding[0]:-self.valid_padding[0]]
        else:
            return out[self.valid_padding[0]:-self.valid_padding[0], \
               self.valid_padding[1] * self.scale:-self.valid_padding[1] * self.scale, \
               self.valid_padding[2] * self.scale:-self.valid_padding[2] * self.scale]

