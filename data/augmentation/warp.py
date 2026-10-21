import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augmentor import DataAugment

class Elastic(DataAugment):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https://gist.github.com/erniejunior/601cdf56d2b424757de5.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    Args:
        alpha (float): maximum pixel-moving distance of elastic deformation. Default: 10.0
        sigma (float): standard deviation of the Gaussian filter. Default: 4.0
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self,
                 alpha=10.0,
                 sigma=4.0,
                 p=0.5):
        
        super(Elastic, self).__init__(p)
        self.alpha = alpha
        self.sigma = sigma
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        max_margin = int(self.alpha) + 1
        self.sample_params['add'] = [0, max_margin, max_margin]

    def __call__(self, data, random_state=np.random):
        if 'label' in data and data['label'] is not None:
            lr, hr, label = data['lr'],data['hr'],data['label']
        else:
            image = data['image']

        height, width = lr.shape[-2:] # (c, z, y, x)
        height2, width2 = hr.shape[-2:]  # (c, z, y, x)
        scale = height2//height

        dx_lr = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)
        dy_lr = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), self.sigma) * self.alpha)

        x_lr, y_lr = np.meshgrid(np.arange(width), np.arange(height))
        mapx_lr, mapy_lr = np.float32(x_lr + dx_lr), np.float32(y_lr + dy_lr)
        dx_hr = dx_lr*scale
        dy_hr = dy_lr*scale

        x_hr, y_hr = np.meshgrid(np.arange(width2), np.arange(height2))
        mapx_hr, mapy_hr = np.float32(x_hr + dx_hr), np.float32(y_hr + dy_hr)

        output = {}
        transformed_image_lr = []
        transformed_image_hr = []
        transformed_label = []

        for i in range(lr.shape[-3]):
            if lr.ndim == 3:
                transformed_image_lr.append(cv2.remap(lr[i], mapx_lr, mapy_lr,
                                    self.image_interpolation, borderMode=self.border_mode))
                transformed_image_hr.append(cv2.remap(hr[i], mapx_hr, mapy_hr,
                                                      self.image_interpolation, borderMode=self.border_mode))
            else:
                temp = [cv2.remap(image[channel, i], mapx, mapy, self.image_interpolation, 
                        borderMode=self.border_mode) for channel in range(image.shape[0])]     
                transformed_image.append(np.stack(temp, 0))          
            if 'label' in data and data['label'] is not None:
                transformed_label.append(cv2.remap(label[i], mapx_hr, mapy_hr, self.label_interpolation, borderMode=self.border_mode))

        if image.ndim == 3: # (z,y,x)
            transformed_image_lr = np.stack(transformed_image_lr, 0)
            transformed_image_hr = np.stack(transformed_image_hr, 0)
        else: # (c,z,y,x)
            transformed_image = np.stack(transformed_image, 1)

        transformed_label = np.stack(transformed_label, 0)
        output['lr'] = transformed_image_lr
        output['hr'] = transformed_image_hr
        output['label'] = transformed_label

        return output
