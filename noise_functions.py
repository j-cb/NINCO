''' All basic noise functions and SyntheticNoise instances should take a datapoint as first argument. A datapoint is a tuple of a PIL.Image.Image and a label.
Additional arguments should be keyword arguments.
Filter and scaling functions take a datapoint where the image is already processed with _np_image_from_raw_datapoint.
All noise functions should return an image sized numpy array with values between 0 and 1 and a label.
Standard label for noise is NL and set to 0.
'''

import numpy as np
import skimage

NL = 0 #label for noise

def _np_image_from_raw_datapoint(datapoint):
    return np.array(datapoint[0])/255



################ Basic noise functions ################


def noise_uniform(raw_datapoint):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    return np.random.uniform(low=0.0, high=1., size=image.shape), NL


def noise_pixel_permutation(raw_datapoint):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    flat_image = image.reshape((-1, image.shape[-1])).copy()
    np.random.shuffle(flat_image)
    shuffled = flat_image.reshape(image.shape)
    return shuffled, NL


def noise_rademacher(raw_datapoint):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    return np.random.random(image.shape) > 0.5, NL


def noise_monochrome(raw_datapoint, color):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    return image*0 + color, NL


def noise_binomial(raw_datapoint, p):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    return np.float32(np.random.binomial(n=1, p=p, size=image.shape)), NL     
        
    
def noise_gaussian_unclipped(raw_datapoint, sigma=1/4):
    image = _np_image_from_raw_datapoint(raw_datapoint)
    return sigma*np.random.randn(*image.shape)+.5, NL     


def noise_stripes(raw_datapoint, orientation, color_list):
    image = _np_image_from_raw_datapoint(raw_datapoint)*0
    n_stripes = len(color_list)
    if orientation == 'horizontal':
        stripe_height = image.shape[0]//n_stripes
        assert n_stripes > 0
        for i, c in enumerate(color_list):
            image[i*stripe_height:(i+1)*stripe_height,:,:] += c
        image[(i+1)*stripe_height:,:,:] += c
    elif orientation == 'vertical':
        stripe_width = image.shape[1]//n_stripes
        assert n_stripes > 0
        for i, c in enumerate(color_list):
            image[:,i*stripe_width:(i+1)*stripe_width,:] += c
        image[:,(i+1)*stripe_width:,:] += c
    return image, NL
        
################ Filter functions ################


def filter_gauss(datapoint, srange=[1,1], mode='wrap'):
    image = datapoint[0]
    sigma = srange[0] + np.random.random_sample()*(srange[1]-srange[0])
    imgn_gaussed = skimage.filters.gaussian(image, sigma=sigma, channel_axis=2, mode=mode)
    return imgn_gaussed, datapoint[1] 



################ Scaling functions ################

def scale_full_brightness_range(datapoint):
    image = datapoint[0]
    img_0_based = image - image.min()
    img_scaled = img_0_based/(img_0_based.max())
    return img_scaled, datapoint[1]


def scale_full_brightness_range_per_channel(datapoint):
    image = datapoint[0]
    img_0_based = image - np.amin(image, axis=(0,1))
    img_scaled = img_0_based/(np.amax(img_0_based,axis=(0,1)))
    return img_scaled, datapoint[1]


def scale_color_range(datapoint, color_center=(.5, .5, .5), color_deviation=(.1, .1, .1), quantile_in_deviation=0.95):
    image = datapoint[0]
    img_0_based = image - np.quantile(image, (1 - quantile_in_deviation)/2, axis=(0,1))
    img_normalized = img_0_based/(np.quantile(img_0_based, 0.5 + quantile_in_deviation/2, axis=(0,1)))
    img_scaled = 2*img_normalized*color_deviation + color_center
    img_clipped = np.clip(img_scaled, 0, 1)
    return img_clipped, datapoint[1]


def push_out(datapoint, lower=.75, upper=1.0):
    image = datapoint[0]
    image[image < lower] = 0.0
    image[image > upper] = 1.0
    return image, datapoint[1]


def clip(datapoint, lower=.0, upper=1.0):
    image = datapoint[0]
    return image.clip(lower, upper), datapoint[1]




class SyntheticNoise:

    def __init__(self, base_noise, noise_filter=None, scaling=None,
                 default_kwargs_base_noise={}, default_kwargs_noise_filter={}, default_kwargs_scaling={}
                ):
        self.base_noise = base_noise
        if noise_filter is not None:
            self.noise_filter = noise_filter
        else:
            self.noise_filter = lambda x: x
        if scaling is not None:
            self.scaling = scaling
        else:
            self.scaling = lambda x: x
        self.default_kwargs_base_noise = default_kwargs_base_noise
        self.default_kwargs_noise_filter = default_kwargs_noise_filter
        self.default_kwargs_scaling = default_kwargs_scaling
       
    def __call__(self, datapoint, kwargs_dict_base_noise=None, kwargs_dict_noise_filter=None, kwargs_dict_scaling=None):
        if kwargs_dict_base_noise is None:
            kwargs_dict_base_noise = self.default_kwargs_base_noise
        if kwargs_dict_noise_filter is None:
            kwargs_dict_noise_filter = self.default_kwargs_noise_filter
        if kwargs_dict_scaling is None:
            kwargs_dict_scaling = self.default_kwargs_scaling
        start_noise = self.base_noise(datapoint, **kwargs_dict_base_noise)
        filtered_noise = self.noise_filter(start_noise, **kwargs_dict_noise_filter)
        return clip(self.scaling(filtered_noise, **kwargs_dict_scaling))
    

    
def get_uniform_noise():
    return SyntheticNoise(base_noise=noise_uniform)


def get_low_freq_uni_fullscale_noise():
    return SyntheticNoise(base_noise=noise_uniform, noise_filter=filter_gauss, scaling=scale_full_brightness_range)


def get_low_freq_uni_channelfullscale_noise():
    return SyntheticNoise(base_noise=noise_uniform, noise_filter=filter_gauss, scaling=scale_full_brightness_range_per_channel)


def get_low_freq_uni_colorrange_noise():
    return SyntheticNoise(base_noise=noise_uniform, noise_filter=filter_gauss, scaling=scale_color_range)


def get_pixel_permutation_noise():
    return SyntheticNoise(base_noise=noise_pixel_permutation)


def get_low_freq_pixel_permutation_noise():
    return SyntheticNoise(base_noise=noise_pixel_permutation, noise_filter=filter_gauss)


def get_rademacher_noise():
    return SyntheticNoise(base_noise=noise_rademacher)


def get_monochrome():
    return SyntheticNoise(base_noise=noise_monochrome)


def get_black():
    return SyntheticNoise(base_noise=noise_monochrome, default_kwargs_base_noise={'color': (0,0,0)})


def get_white():
    return SyntheticNoise(base_noise=noise_monochrome, default_kwargs_base_noise={'color': (1,1,1)})


def get_blobs():
    return SyntheticNoise(base_noise=noise_binomial, default_kwargs_base_noise={'p': .7},
                          noise_filter=filter_gauss, default_kwargs_noise_filter={'srange': [2,2]},
                          scaling=push_out, default_kwargs_scaling={'lower': .75, 'upper': 1.0},
                         )


def get_gaussian():
    return SyntheticNoise(base_noise=noise_gaussian_unclipped)


def get_stripes():
    return SyntheticNoise(base_noise=noise_stripes)

