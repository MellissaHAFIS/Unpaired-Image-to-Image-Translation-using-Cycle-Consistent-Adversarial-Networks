# Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks

This project implements **CycleGAN**,[^1][^2] an unpaired image-to-image translation method. Leveraging GANs and cycle-consistency, it converts images from one domain to another (e.g., horses ↔ zebras, summer ↔ winter) without requiring paired images.

## Setup your environment
To install requirements, do : 
~~~bash
pip install -r requirements.txt
~~~

## Download a dataset
Run in a terminal
~~~bash
python download_dataset.py
~~~
Then type in terminal which one you want to test.

Available datasets :

ae_photos, apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, mini, mini_pix2pix, mini_colorization

[^1]: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, 
*Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros*](https://arxiv.org/abs/1703.10593)

[^2]: [Pytorch implementation by original authors](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)