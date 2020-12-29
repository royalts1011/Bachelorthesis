from torchvision import transforms
import MyTransforms


# setting mean and std for normalization
norm_mean = [0.485, 0.456, 0.406] # imageNet mean
# [0.49139968, 0.48215841, 0.44653091] # original DLBIO
norm_std=[0.229, 0.224, 0.225] # imageNet std
# [0.24703223, 0.24348513, 0.26158784] # original DLBIO

normalize = transforms.Normalize( mean=norm_mean, std=norm_std )


def get_transform(dict_key, is_small):

    # dictionary to access different transformation methods
    transform_dict = {
        'train': transforms_train(get_resize(is_small=is_small) ),
        'valid_and_test': transforms_valid_and_test( get_resize(is_small=is_small) ),
        'siamese' : transforms_siamese( get_resize(is_small=is_small) ),
        'siamese_valid_and_test' : transforms_siamese_verification( get_resize(is_small=is_small) ),
        'size_only' : None
    }
    assert dict_key in transform_dict, "The string "+str(dict_key)+" was not found in dictionary"
    
    return transform_dict[dict_key]

# Returns boolean decision of small or bigger
def get_resize(is_small):
    if is_small: return 150, 100
    else: return 280, 230


def transforms_train(img_shape):
    mean_pil = tuple([int(x*255) for x in norm_mean])
    return transforms.Compose([
        MyTransforms.RandomScaleWithMaxSize(img_shape, 0.8),
        MyTransforms.PadToSize(img_shape, mean_pil),
        transforms.RandomAffine(degrees=15, fillcolor=mean_pil),
        MyTransforms.MyRandomCrop(crop_ratio=0.1, b_keep_aspect_ratio=True),
        transforms.Resize(img_shape),
        #MyTransforms.GaussianBlur(p=0.2, max_radius=4),
        MyTransforms.AddGaussianNoise(blend_alpha_range=(0., 0.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.02),
        #transforms.RandomHorizontalFlip(),
        #transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
        ])

def transforms_valid_and_test(img_shape):
    return transforms.Compose([
        transforms.Resize(img_shape),
        # transforms.Lambda(lambda x: x.convert('RGB')),
        #transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
        ])

def transforms_siamese(img_shape):
    return transforms.Compose([
        MyTransforms.RandomScaleWithMaxSize(img_shape, 0.8),
        transforms.RandomAffine(degrees=15),
        MyTransforms.MyRandomCrop(crop_ratio=0.1, b_keep_aspect_ratio=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.02),
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.Resize(img_shape),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

def transforms_siamese_verification( img_shape ):
        return transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor()
            ])

class UnNormalize(object):
    def __init__(self):
        self.mean = norm_mean
        self.std = norm_std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor