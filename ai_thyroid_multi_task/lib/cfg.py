from torchvision.transforms import transforms

BASE_PATH = '/home/martin/JupyterLab/ai_thyroid/data/'
PATH_IMAGE = BASE_PATH + 'CEUS1012_ex_sight/' # image
PATH_CALCIFICATION_MASK = BASE_PATH + 'mask_calcification/' # calcification mask
PATH_LESION_MASK = BASE_PATH + 'mask_lesion/' # lesion mask (used visualization tools only)

IMAGE_RESIZE_DIM = (512, 512)

# numpy -> tensor + normalize pixel values
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.218, 0.218, 0.218], [0.181, 0.181, 0.181])
])

# numpy -> tensor
mask_transform = transforms.ToTensor()
