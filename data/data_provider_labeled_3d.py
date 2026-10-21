from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from emsrseg.temporal import context_sections
import os
import sys
import cv2
import h5py
import math
import time
import torch
import random
import numpy as np
from PIL import Image
import multiprocessing
from joblib import delayed
from joblib import Parallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .augmentation import Flip
from .augmentation import Elastic
from .augmentation import Grayscale
from .augmentation import Rotate
from .augmentation import Rescale
from .utils.seg_util import mknhood3d, genSegMalis
from .utils.aff_util import seg_to_affgraph
from .utils.utils import center_crop
from .data.data_segmentation import seg_widen_border, weight_binary_ratio
from .data.data_affinity import seg_to_aff
# from utils.affinity_official import seg2affs
from .utils.affinity_ours import gen_affs_mutex_3d

class Train(Dataset):
	def __init__(self, cfg, scale):
		super(Train, self).__init__()
		# multiprocess settings
		self.cfg = cfg
		self.if_dilate = cfg['if_dilate'] #True
		self.shift_channels = cfg['shift_channels'] #None
		self.output_nc = cfg['output_nc']
		self.separate_weight = cfg['separate_weight'] #False
		self.if_sparse = cfg['if_sparse'] #False
		self.scale = scale
		# basic settings
		# the input size of network
		# hr_crop: HR patch è¾¹é¿, é»è®¤ 256 (ä¸åè¡ä¸ºä¸è´);
		# å¯å¨ yml ç datasets.train éè®¾ hr_crop ä»¥éæ¾å­(å¦ IART_seg x4 ç¨ 128)
		hr_crop = int(cfg.get('hr_crop', 256)) if hasattr(cfg, 'get') else 256
		self.hr_crop = hr_crop
		self.crop_size = [9, hr_crop//scale, hr_crop//scale]
		self.net_padding = [0, 0, 0]


		# the output size of network
		self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

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
			self.train_datasets = ['wafer25_inputs.h5','wafer26_2_inputs.h5','wafer26_inputs.h5','wafer36_inputs.h5']
			self.train_labels = ['wafer25_labels.h5','wafer26_2_labels.h5','wafer26_labels.h5','wafer36_labels.h5']
		else:
			raise AttributeError('No this dataset type!')

		# the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
		self.folder_name_gt = cfg['data_folder_gt']
		self.folder_name_lr = cfg['data_folder_lr']

		assert len(self.train_datasets) == len(self.train_labels)

		# split training data
		self.train_split = cfg["train_split"]

		# augmentation
		self.if_scale_aug = cfg["if_scale_aug"] 
		self.if_filp_aug = cfg["if_filp_aug"]
		self.if_elastic_aug = cfg["if_elastic_aug"]
		self.if_intensity_aug = cfg["if_intensity_aug"]
		self.if_rotation_aug = cfg["if_rotation_aug"]

		# load dataset
		self.dataset = []
		self.dataset_hr = []
		self.labels = []
		for k in range(len(self.train_datasets)):
			f_raw = h5py.File(os.path.join(self.folder_name_lr, self.train_datasets[k]), 'r')
			data = f_raw['main'][:]
			f_raw.close()
			data = data[:self.train_split]
			self.raw_data_shape = list(data.shape)
			self.dataset.append(data)

			# load labels
			f_label = h5py.File(os.path.join(self.folder_name_gt, self.train_labels[k]), 'r')
			label = f_label['main'][:]
			f_label.close()
			label = label[:self.train_split]
			if self.if_dilate:
				label = seg_widen_border(label, tsz_h=1)
			self.labels.append(label)

			f_raw = h5py.File(os.path.join(self.folder_name_gt, self.train_datasets[k]), 'r')
			data = f_raw['main'][:]
			f_raw.close()
			data = data[:self.train_split]
			self.dataset_hr.append(data)



		# padding for random rotation
		self.crop_from_origin = [0, 0, 0]
		self.padding = 0
		self.crop_from_origin[0] = self.crop_size[0]
		self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
		self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

		# augmentation initoalization
		self.augs_init()
		#self.len_data = len(self.train_datasets)*(Z-self.crop_size[0]+1)

	def __getitem__(self, index):
		Z, H, W = self.raw_data_shape
		#random_z = index % (Z-self.crop_size[0]+1)
		random_z = index % (Z - self.crop_size[0] - 3) + 2
		# random select one dataset if contain many datasets
		#k = index // (Z-self.crop_size[0]+1)
		k = index // (Z - self.crop_size[0] - 3)
		lr_data = self.dataset[k]
		hr_data = self.dataset_hr[k]
		used_label = self.labels[k]
		if self.raw_data_shape[1]-self.crop_from_origin[1]<0:
			lr_imgs = lr_data[random_z - 2:random_z + 2 + self.crop_from_origin[0], \
				   0:self.raw_data_shape[1], \
				   0:self.raw_data_shape[2]].copy()
			hr_imgs = hr_data[random_z:random_z + self.crop_from_origin[0], \
					  0:self.raw_data_shape[1]*self.scale, \
					  0:self.raw_data_shape[2]*self.scale].copy()
			lb = used_label[random_z:random_z + self.crop_from_origin[0], \
				   0:self.raw_data_shape[1]*self.scale, \
				   0:self.raw_data_shape[2]*self.scale].copy()

		else:
			random_y = random.randint(0, self.raw_data_shape[1] - self.crop_from_origin[1])
			random_x = random.randint(0, self.raw_data_shape[2] - self.crop_from_origin[2])
			lr_imgs = lr_data[random_z-2:random_z+2+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
			hr_imgs = hr_data[random_z:random_z + self.crop_from_origin[0], \
					  random_y*self.scale:(random_y + self.crop_from_origin[1])*self.scale, \
					  random_x*self.scale:(random_x + self.crop_from_origin[2])*self.scale].copy()
			lb = used_label[random_z:random_z+self.crop_from_origin[0], \
						random_y*self.scale:(random_y+self.crop_from_origin[1])*self.scale, \
						random_x*self.scale:(random_x+self.crop_from_origin[2])*self.scale].copy()


		lr_imgs = lr_imgs.astype(np.float32) / 255.0
		hr_imgs = hr_imgs.astype(np.float32) / 255.0
		data = {'lr': lr_imgs, 'hr': hr_imgs, 'label': lb}
		# p=0.5 for augmentation
		if np.random.rand() < 0.5:
			data = self.augs_mix(data)
		lr_imgs = data['lr']
		hr_imgs = data['hr']
		lb = data['label']
		if self.raw_data_shape[1] - self.crop_from_origin[1] >= 0:
			lr_imgs = center_crop(lr_imgs, det_shape=[self.crop_size[0]+4] + self.crop_size[1:])
			hr_imgs = center_crop(hr_imgs, det_shape=[self.crop_size[0]] + [list_x * self.scale for list_x in self.crop_size[1:]])
			lb = center_crop(lb, det_shape=[self.crop_size[0]] + [list_x * self.scale for list_x in self.crop_size[1:]])

		if self.shift_channels is None:
			if self.if_sparse:
				if_bg = False
			else:
				if_bg = True
			if self.output_nc == 3:
				lb_affs = seg_to_aff(lb,if_bg=if_bg).astype(np.float32)
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
										padding=True, background=False)

		if self.separate_weight:
			weightmap = np.zeros_like(lb_affs)
			for i in range(lb_affs.shape[0]):
				weightmap[i] = weight_binary_ratio(lb_affs[i])
		else:
			weightmap = weight_binary_ratio(lb_affs)

		lr_imgs = lr_imgs[np.newaxis, ...]
		hr_imgs = hr_imgs[np.newaxis, ...]
		lb = lb.astype(np.float32)
		lr_imgs = torch.from_numpy(np.ascontiguousarray(lr_imgs, dtype=np.float32)).contiguous()
		hr_imgs = torch.from_numpy(np.ascontiguousarray(hr_imgs, dtype=np.float32)).contiguous()
		lb_affs = torch.from_numpy(np.ascontiguousarray(lb_affs, dtype=np.float32)).contiguous()
		weightmap = torch.from_numpy(np.ascontiguousarray(weightmap, dtype=np.float32)).contiguous()

		return {"lr": context_sections(lr_imgs), "hr": hr_imgs, "lb": lb, "lb_affs": lb_affs, "weightmap": weightmap}

	def __len__(self):
		Z, H, W = self.raw_data_shape
		return len(self.train_datasets)*(Z-self.crop_size[0]-3)

	def augs_init(self):
		# https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
		self.aug_rotation = Rotate(p=0.5)
		self.aug_rescale = Rescale(p=0.5)
		self.aug_flip = Flip(p=1.0, do_ztrans=0)
		self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
		self.aug_grayscale = Grayscale(p=0.75)

	# TO DO
	def augs_single(self, data):
		random_id = np.random.randint(1, 5+1)
		if random_id == 1:
			data = self.aug_rotation(data)
		elif random_id == 2:
			data = self.aug_rescale(data)
		elif random_id == 3:
			data = self.aug_flip(data)
		elif random_id == 4:
			data = self.aug_elastic(data)
		elif random_id == 5:
			data = self.aug_grayscale(data)
		else:
			raise NotImplementedError
		return data

	def augs_mix(self, data):
		if self.if_filp_aug and random.random() > 0.5:
			data = self.aug_flip(data)
		if self.if_rotation_aug and random.random() > 0.5:
			data = self.aug_rotation(data)
		if self.if_scale_aug and random.random() > 0.5:
			data = self.aug_rescale(data)
		if self.if_elastic_aug and random.random() > 0.5:
			data = self.aug_elastic(data)
		if self.if_intensity_aug and random.random() > 0.5:
			data = self.aug_grayscale(data)
		return data


def show(img3d):
	# only used for image with shape [18, 160, 160]
	num = img3d.shape[0]
	column = 5
	row = math.ceil(num / float(column))
	size = img3d.shape[1]
	img_all = np.zeros((size*row, size*column), dtype=np.uint8)
	for i in range(row):
		for j in range(column):
			index = i*column + j
			if index >= num:
				img = np.zeros_like(img3d[0], dtype=np.uint8)
			else:
				img = (img3d[index] * 255).astype(np.uint8)
			img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
	return img_all

