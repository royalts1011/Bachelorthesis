from torchvision import transforms
import MyTransforms

# setting mean and std for normalization
norm_mean = [0.485, 0.456, 0.406] # imageNet mean
# [0.49139968, 0.48215841, 0.44653091] # original DLBIO
norm_std=[0.229, 0.224, 0.225] # imageNet std
# [0.24703223, 0.24348513, 0.26158784] # original DLBIO

normalize = transforms.Normalize( mean=norm_mean, std=norm_std )

def transforms_train(img_shape):
    mean_pil = tuple([int(x*255) for x in norm_mean])
    transformations = transforms.Compose([
        MyTransforms.RandomScaleWithMaxSize(img_shape, 0.8),
        MyTransforms.PadToSize(img_shape, mean_pil),
        transforms.RandomAffine(degrees=15, fillcolor=mean_pil),
        MyTransforms.MyRandomCrop(crop_ratio=0.1, b_keep_aspect_ratio=True),
        transforms.Resize(img_shape),
        #MyTransforms.GaussianBlur(p=0.2, max_radius=4),
        MyTransforms.AddGaussianNoise(blend_alpha_range=(0., 0.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.02),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    return transformations

def transforms_valid_and_test(img_shape):
    transformations = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        normalize
        ])
    return transformations

def transforms_siamese(img_shape):
    transformations = transforms.Compose([
        transforms.Resize(img_shape),
        transforms.ToTensor()
        ])
    return transformations