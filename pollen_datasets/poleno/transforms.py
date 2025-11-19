import random
import torchvision.transforms.functional as TF


class PairTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img0, img1):
        for transform in self.transforms:
            img0, img1 = transform(img0, img1)
        return img0, img1
    

class SwapRotate180:
    """
    Swap images + 180° rotation. 
    This corresponds to flipping the 3D object upside-down.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        swapped = False
        if random.random() < self.p:
            img1, img2 = img2, img1
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)
            swapped = True
        return img1, img2, {"swapped": swapped}