import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet101
from PIL import Image
import os, glob
import numpy as np

def compute_fcn_score(label_dir, fake_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fcn_resnet101(pretrained=True).eval().to(device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    ious, counts = [], []
    for real_path in glob.glob(os.path.join(label_dir, '*.png')):
        name = os.path.basename(real_path)
        fake_path = os.path.join(fake_dir, name.replace('real','fake'))
        if not os.path.exists(fake_path): continue

        real = Image.open(real_path).convert('RGB')
        fake = Image.open(fake_path).convert('RGB')
        with torch.no_grad():
            out_real = model(tf(real).unsqueeze(0).to(device))['out'].argmax(1).cpu().numpy()[0]
            out_fake = model(tf(fake).unsqueeze(0).to(device))['out'].argmax(1).cpu().numpy()[0]
    
        intersection = np.logical_and(out_real==out_fake, out_real>0)
        union        = np.logical_or(out_real>0, out_fake>0)
        ious.append(intersection.sum()/union.sum() if union.sum()>0 else 0)
    print("Mean IOU:", np.mean(ious))

if __name__=='__main__':
    compute_fcn_score('results/horse2zebra/test_latest/images', 'results/horse2zebra/test_latest/images')
