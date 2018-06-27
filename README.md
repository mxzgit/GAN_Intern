# Generative Adversarial networks

The following repository contains an ensemble of GANs algorithms based on a theoritical study of the following papers :
1. Ian J. Goodfellow,Jean Pouget-Abadie,Mehdi Mirza,Bing Xu,David Warde-Farley,Sherjil Ozair,Aaron C. Courville, Yoshua Bengio, “Generative Adversarial Networks,” 2014; https://arxiv.org/abs/1406.2661
2. Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel,“InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets,”in NIPS,2016; https://arxiv.org/pdf/1606.03657.pdf
3. Alec Radford, Luke Metz, Soumith Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks,” in ArXiv, 2015; https://arxiv.org/pdf/1511.06434.pdf
4. Abhishek Kumar, Prasanna Sattigeri, Tom Fletcher, “Semi-supervised Learning with GANs: Manifold Invariance with Improved Inference,” in NIPS, 2017; https://papers.nips.cc/paper/7137-semi-supervised-learning-with-gans-manifold-invariance-with-improved-inference.pdf
5. Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, XiChen, “Improved Techniques for Training GANs,” in CVPR, 2016; https://arxiv.org/abs/1606.03498
6. Martín Arjovsky, Soumith Chintala, Léon Bottou,”Wasserstein GAN,” in ICML,2017; https://arxiv.org/abs/1701.07875
7. Alec Radford, Luke Metz, Soumith Chintala, “Unsupervised Representation Learningwith Deep Convolutional Generative Adversarial Networks,” in ArXiv, 2015; https://arxiv.org/abs/1511.06434

The downlod script helpd to download the most dataset used during the training and testing the aglorithms. The available datasets are :
1. MNIST dataset: dataset of digit images, size 28*28
2. CelebA dataset: dataset of celebrities faces, size 178*218
3. Cat dataset: dataset containing images of cats, size 
4. Dog dataset: dataset containing images of dogs, size

For the sake of minimizing the computation time and because of the lack of low hardware resurces, we used a script to resize the images size to be 28*28 as the MNIST Dataset. the script helper.py from https://github.com/zackthoutt/face-generation-gan .
