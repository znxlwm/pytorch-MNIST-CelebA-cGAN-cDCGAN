# pytorch-MNIST-CelebA-cGAN-cDCGAN
Pytorch implementation conditional Generative Adversarial Networks (cGAN) [1] and conditional Generative Adversarial Networks (cDCGAN) for MNIST [2] and CelebA [3] datasets.

* The network architecture (number of layer, layer size and activation function etc.) of this code differs from the paper.

* CelebA dataset used gender lable as condition.

* you can download
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/
  - CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Resutls
### MNIST
* Generate using fixed noise (fixed_z_)

<table align='center'>
<tr align='center'>
<td> cGAN</td>
<td> cDCGAN</td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/generation_animation.gif'>
<td><img src = 'MNIST_cDCGAN_results/MNIST_cDCGAN_generation_animation.gif'>
</tr>
</table>

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> cGAN after 50 epochs </td>
<td> cDCGAN after 20 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_cGAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_cGAN_results/MNIST_cGAN_50.png'>
<td><img src = 'MNIST_cDCGAN_results/MNIST_cDCGAN_20.png'>
</tr>
</table>

* Learning Time
  * MNIST cGAN - Avg. per epoch: 9.13 sec; Total 50 epochs: 937.06 sec
  * MNIST cDCGAN - Avg. per epoch: 47.16 sec; Total 20 epochs: 1024.26 sec

### CelebA
* Generate using fixed noise (fixed_z_)

![CelebA](CelebA_cDCGAN_results/CelebA_cDCGAN_generation_animation.gif)

* CelebA vs Generated images

<table align='center'>
<tr align='center'>
<td> CelebA </td>
<td> cDCGAN after 20 epochs </td>
</tr>
<tr>
<td><img src = 'CelebA_cDCGAN_results/raw_CelebA.png'>
<td><img src = 'CelebA_cDCGAN_results/CelebA_cDCGAN_20.png'>
</tr>
</table>

* CelebA cDCGAN morphing (noise interpolation)
![morphing](CelebA_cDCGAN_results/CelebA_cDCGAN_morp.png)

* Learning Time
  * CelebA cDCGAN - Avg. per epoch: 826.69 sec; total 20 epochs ptime: 16564.10 sec

## Development Environment

* Ubuntu 14.04 LTS
* NVIDIA GTX 1080 ti
* cuda 8.0
* Python 2.7.6
* pytorch 0.1.12
* torchvision 0.1.8
* matplotlib 1.3.1
* imageio 2.2.0

## Reference

[1] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

(Full paper: https://arxiv.org/pdf/1411.1784.pdf)

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

[3] Liu, Ziwei, et al. "Deep learning face attributes in the wild." Proceedings of the IEEE International Conference on Computer Vision. 2015.
