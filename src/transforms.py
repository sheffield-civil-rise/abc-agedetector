import torch
import kornia as K
from torchvision import transforms as T

class DefaultTransformation:
    ''' This is the default transformation for the age detector '''
    def __init__(self, nb_patches=10, patch_size=128, crop_size=1024):
        self.nb_patches=nb_patches
        self.patch_size=patch_size
        self.crop_size=crop_size
        self._compose()

    def _compose(self):
        ''' creates a composition transformation '''
        self._transform = T.Compose([
            T.CenterCrop((self.crop_size, self.crop_size)),
            T.ToTensor(),
            ExtractPatchUnfold(self.nb_patches, self.patch_size)],
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __call__(self, x):
        ''' propogate x through transform and return result '''
        return self._transform(x)



class ExtractPatchUnfold:
    ''' Transformation that can be used to extract patches and create stacked tensor '''
    def __init__(self, nb_patches=10, patch_size=128):
        self.nb_patches=nb_patches
        self.patch_size=patch_size

    def __call__(self, x):
        ''' currently this function repeats the input nb_patches along the newly created 1st dimension and applies an MxM random crop'''
        dim, M = 0, self.patch_size
        x = torch.repeat_interleave(
            x.unsqueeze(dim),
            repeats=self.nb_patches,
            dim=dim)
        sequence = K.augmentation.ImageSequential(
            K.augmentation.RandomCrop((M, M)),
            same_on_batch=False)
        return sequence(x)
