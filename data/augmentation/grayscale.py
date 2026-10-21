import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    """Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    and apply gamma correction.

    Args:
        contrast_factor (float): intensity of contrast change. Default: 0.3
        brightness_factor (float): intensity of brightness change. Default: 0.3
        mode (string): one of ``'2D'``, ``'3D'`` or ``'mix'``. Default: ``'mix'``
        p (float): probability of applying the augmentation. Default: 0.5
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters.
        """
        super(Grayscale, self).__init__(p=p)
        self._set_mode(mode)
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        # No change in sample size
        pass

    def __call__(self, data, random_state=np.random):

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'
        else:
            mode = self.mode

        # apply augmentations  
        if mode == '2D': 
            data = self._augment2D(data, random_state)
        if mode == '3D': 
            data = self._augment3D(data, random_state)
        return data

    def _augment2D(self, data, random_state=np.random):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs_lr = data['lr']
        imgs_hr = data['hr']
        transformedimgs_lr = np.copy(imgs_lr)
        transformedimgs_hr = np.copy(imgs_hr)
        ran = random_state.rand(transformedimgs_lr.shape[-3]*3)

        for z in range(transformedimgs_lr.shape[-3]):
            img_lr = transformedimgs_lr[z, :, :]
            img_lr *= 1 + (ran[z*3] - 0.5)*self.CONTRAST_FACTOR
            img_lr += (ran[z*3+1] - 0.5)*self.BRIGHTNESS_FACTOR
            img_lr = np.clip(img_lr, 0, 1)
            img_lr **= 2.0**(ran[z*3+2]*2 - 1)
            transformedimgs_lr[z, :, :] = img_lr

        for z in range(transformedimgs_hr.shape[-3]):
            img_hr = transformedimgs_hr[z, :, :]
            img_hr *= 1 + (ran[z * 3] - 0.5) * self.CONTRAST_FACTOR
            img_hr += (ran[z * 3 + 1] - 0.5) * self.BRIGHTNESS_FACTOR
            img_hr = np.clip(img_hr, 0, 1)
            img_hr **= 2.0 ** (ran[z * 3 + 2] * 2 - 1)
            transformedimgs_hr[z, :, :] = img_hr

        data['lr'] = transformedimgs_lr
        data['hr'] = transformedimgs_hr
        return data    

    def _augment3D(self, data, random_state=np.random):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        ran = random_state.rand(3)

        imgs_lr = data['lr']
        transformedimgs_lr = np.copy(imgs_lr)
        transformedimgs_lr *= 1 + (ran[0] - 0.5)*self.CONTRAST_FACTOR
        transformedimgs_lr += (ran[1] - 0.5)*self.BRIGHTNESS_FACTOR
        transformedimgs_lr = np.clip(transformedimgs_lr, 0, 1)
        transformedimgs_lr **= 2.0**(ran[2]*2 - 1)
        imgs_hr = data['hr']
        transformedimgs_hr = np.copy(imgs_hr)
        transformedimgs_hr *= 1 + (ran[0] - 0.5) * self.CONTRAST_FACTOR
        transformedimgs_hr += (ran[1] - 0.5) * self.BRIGHTNESS_FACTOR
        transformedimgs_hr = np.clip(transformedimgs_hr, 0, 1)
        transformedimgs_hr **= 2.0 ** (ran[2] * 2 - 1)
        
        data['lr'] = transformedimgs_lr
        data['hr'] = transformedimgs_hr
        return data

    def _invert(self, data, random_state=np.random):
        """
        Invert input images
        """
        lr = data['lr']
        hr = data['hr']
        transformedimgs_lr = np.copy(lr)
        transformedimgs_lr = 1.0-transformedimgs_lr
        transformedimgs_lr = np.clip(transformedimgs_lr, 0, 1)

        data['lr'] = transformedimgs_lr
        transformedimgs_hr = np.copy(hr)
        transformedimgs_hr = 1.0 - transformedimgs_hr
        transformedimgs_hr = np.clip(transformedimgs_hr, 0, 1)

        data['hr'] = transformedimgs_hr
        return data

    ####################################################################
    ## Setters.
    ####################################################################

    def _set_mode(self, mode):
        """Set 2D/3D/mix greyscale value augmentation mode."""
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode
