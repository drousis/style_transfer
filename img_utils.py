import torch
from torchvision import utils
import torchvision.transforms as transforms
from IPython.core.display import display, HTML
from matplotlib import pyplot as plt

device = torch.device("cuda")
unloader = transforms.ToPILImage()

def imload(img, height, ar=1, zoom=None):
    """Prepare a PIL image for pytorch."""
    if zoom:
        img = img.resize([int(_*zoom) for _ in img.size])
        
    loader = transforms.Compose([
        transforms.CenterCrop((height, height * ar)),
        transforms.ToTensor()
    ])
    
    return loader(img).unsqueeze(0).to(device, torch.float)
        

def imshow(l, w, *args):
    """Show images as a row in a notebook."""
    plt.figure(figsize=(l, w))
    for i, tensor in enumerate(args):
        plt.subplot(1, len(args), i+1) # .axis("off")
        image = tensor.cpu().clone().squeeze(0)
        image = unloader(image)
        plt.imshow(image)


def imsave(tensor, filepath, download=True):
    """Save an image to disk and print download link."""
    utils.save_image(tensor, filepath)
    
    if download:
        display(HTML(f'<a href="{filepath}" download>Download</h1>'))