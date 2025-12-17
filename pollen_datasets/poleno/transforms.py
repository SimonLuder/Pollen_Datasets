import random
import torchvision.transforms.functional as TF


class PairTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img0, img1):
        for transform in self.transforms:
            img0, img1, meta = transform(img0, img1)
        return img0, img1, meta
    

class SwapRotate180:
    """
    Swap images + 180° rotation. 
    This corresponds to flipping the 3D object upside-down.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2, meta=None):
        if meta is None:
            meta = {}
        swapped = False
        if random.random() < self.p:
            img1, img2 = img2, img1
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)
            swapped = True
            meta.update({"swapped": swapped})
        return img1, img2, meta


class RotatePairKx90:
    """
    Rotate both images by k*90 degrees counter-clockwise.
    """
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img1, img2, meta=None):
        if meta is None:
            meta = {}
        k = 0
        if random.random() < self.p:
            k = random.randint(0, 3)  # 0,1,2,3 → 0°, 90°, 180°, 270°
            deg = 90 * k
            img1 = TF.rotate(img1, deg)
            img2 = TF.rotate(img2, deg)
            meta.update({"rotation_deg": deg})
        return img1, img2, meta
    
