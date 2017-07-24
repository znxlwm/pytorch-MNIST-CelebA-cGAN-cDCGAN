# pytorch-MNIST-cGAN
Pytorch implementation conditional Generative Adversarial Networks (cGAN) [1] for MNIST [2] dataset

* you can download
  - MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Resutls
### MNIST
* Generate using fixed noise (fixed_z_)

![MNIST](MNIST_cGAN_results/generation_animation.gif)

* MNIST vs Generated images

<table align='center'>
<tr align='center'>
<td> MNIST </td>
<td> cGAN after 50 epochs </td>
</tr>
<tr>
<td><img src = 'MNIST_GAN_results/raw_MNIST.png'>
<td><img src = 'MNIST_cGAN_results/MNIST_cGAN_50.png'>
</tr>
</table>

* Learning Time
  * MNIST cGAN - Avg. per epoch: 9.13 sec; Total 50 epochs: 937.06 sec


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
