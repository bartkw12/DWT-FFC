**My understanding of your assignment**

For Baseline 1 (Breaking through the Haze: An advanced non-homogeneous dehazing method based on the fast fourier convolution and ConvNeXt), your PhD student wants you to do more than just read the paper. You need to:

1. Understand the model architecture (how the full network is built).
2. Understand the optimization / training setup.
3. Understand the dataset and dataloader construction.
4. Answer what to do if the test image is too large (i.e., how to avoid OOM).
5. Explain the benefit of using ConvNeXt in this method.
6. At a practical level, you’re also expected to at least run evaluation/inference, inspect the model classes and submodules carefully, and do a simple shortened training run rather than full-scale training. 

So your task is really a mix of:

* paper understanding,
* repo/code understanding, and
* light reproduction / implementation understanding.

**My understanding of Baseline 1 itself**

The paper proposes a two-branch GAN-style dehazing network for non-homogeneous haze, where haze is spatially irregular and much harder than standard homogeneous haze. The authors argue that existing methods struggle mainly because:
* non-homogeneous haze is structurally complex and hard to restore in dense regions, and
* the available datasets are too small for reliable supervised learning. 

The network has two main branches

1. **DWT-FFC frequency branch**
* This branch is responsible for learning the mapping from hazy image → clear image. It is an encoder–decoder with skip connections, and it uses:
  * DWT (Discrete Wavelet Transform) to preserve and propagate high-frequency information like edges and fine textures,
  * FFC (Fast Fourier Convolution) residual blocks to give the model a large/global receptive field by combining local spatial processing with frequency-domain processing. 

This branch is basically trying to solve:
“How do we reconstruct image structure and details when haze is spatially irregular and sometimes very dense?” 

**2. Prior knowledge branch**

* This is the transfer-learning branch. It uses the first three stages of an ImageNet-pretrained ConvNeXt encoder, with a decoder that upsamples features back to image resolution using pixel shuffle and attention blocks. This branch injects prior visual knowledge to reduce overfitting on small dehazing datasets.

**Final output**
The outputs of the two branches are fused, and a discriminator is used during training to improve perceptual realism via adversarial loss.











































