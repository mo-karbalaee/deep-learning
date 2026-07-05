---
course: deep-learning
title: Self-Supervised Learning
source: deep-learning/slides/selfSupervisedLearning/ssl.tex
---

# Self-Supervised Learning

## Definition

### Motivation  

Researchers have repeatedly warned that the field of artificial intelligence relies too heavily on manually provided supervision.  Jitendra Malik famously stated, “Supervision is the opium of the AI researcher.”  Likewise, Alyosha Efros has predicted that “the AI revolution will not be supervised.”  Yann LeCun reinforced this view with a memorable metaphor:  

> *“Unsupervised learning is the cake, supervised learning is the icing, and reinforcement learning is the cherry.”*  

The accompanying illustration shows LeCun’s portrait (affiliated with Carnegie Mellon University’s Machine Learning department) together with the quote, and the logos of Facebook AI Research and Carnegie Mellon University.  The visual metaphor emphasizes that the bulk of learning should come from unsupervised (or self‑supervised) methods, while supervised and reinforcement learning provide comparatively smaller contributions.  Together, these statements challenge the prevailing dominance of supervised learning and motivate a shift toward learning paradigms that do not require explicit human‑provided labels.

---

### Idea  

Self‑supervised learning exploits the inherent structure of raw data to generate its own training signals.  The figure visualizes this idea as a series of stacked three‑dimensional blocks aligned along a horizontal “Time” axis.  Each block consists of teal and magenta layers that represent different portions of an input sample (for instance, past vs. future video frames, or visible vs. occluded image regions).  Arrows labeled “Past” and “Future” indicate the direction in which the model must make predictions.  

Typical pretext tasks illustrated in the diagram include:  

- Predicting the future from the past.  
- Predicting the near future from the recent past.  
- Reconstructing the past from the present.  
- Inferring the top part of an image from the bottom.  
- Filling in occluded regions from the visible surrounding pixels.  
- Pretending that a part of the input is missing and predicting it.  

In each case the model is trained to reconstruct or anticipate the missing portion using the observable part as a cue.  Because the missing portion is known during training, it supplies an automatic supervisory signal without any external annotation.  This strategy, emphasized by LeCun, converts raw data’s latent structure into a supervised‑style learning problem.

---

### Self‑Supervised Learning: Definition  

Self‑supervised learning (SSL) is formally defined as a **sub‑category of unsupervised learning** in which the learning algorithm constructs *pretext*, *surrogate*, or *pseudo* tasks that can be solved using the standard supervised learning framework.  The key characteristics of SSL are:

1. **Automatically generated labels.**  
   The pretext task itself provides a label for each training example (e.g., the next video frame, the missing image patch, or the rotation angle applied to an image).  No human annotation is required.

2. **Explicit measurement of correctness.**  
   Since the label is known by construction, a loss function can be defined that quantifies how well the model’s prediction matches the automatically generated target (e.g., mean‑squared error for reconstruction, cross‑entropy for classification of a rotation angle).

3. **Purpose as a representation learner.**  
   The representations learned during the pretext phase are not the final goal; they serve as a *precursor* to downstream tasks such as image retrieval, fully supervised classification, or semi‑supervised classification.  After pretraining, the learned encoder is typically fine‑tuned or frozen and plugged into a downstream model.

4. **Generative models as SSL methods.**  
   Techniques that learn to generate data—most prominently Generative Adversarial Networks (GANs)—fit within the SSL framework because they also rely on automatically derived objectives (e.g., discriminating real from generated samples) without external labels.

#### Two‑Stage Training Process  

> **Figure description.** The diagram shows a convolutional neural network (ConvNet) undergoing two distinct training stages.  The upper stage (enclosed by a dashed orange border) depicts *self‑supervised pretext task training* on an unlabeled dataset containing assorted animal images.  A “Pretext Task” block supplies an automatic learning signal to the ConvNet.  The lower stage (also within a dashed orange border) illustrates *supervised downstream task training* on a labeled dataset that includes images of tools and birds.  A “Downstream Task” block consumes the ConvNet’s output.  An arrow labeled “Knowledge Transfer” connects the ConvNet from the pretraining stage to the downstream stage, indicating that the representations learned without labels are reused for the supervised task.

During the **pretext stage**, the ConvNet learns to encode useful visual features by solving an artificial prediction problem on the unlabeled data.  In the **downstream stage**, those learned features are transferred to a conventional supervised setting, often by attaching a small classification head and fine‑tuning on a modestly sized labeled dataset.  This two‑step procedure is the operational backbone of most modern SSL approaches.

---

### Advantages of Self‑Supervised Learning  

#### Comparison with Traditional Supervised Learning  

> **Figure description (supervised workflow).** Three input images of cats and dogs pass through a “Bottleneck” labeled “Manual Annotation” (teal region).  Human annotators assign categorical labels (“Cat” or “Dog”), yielding paired data $(X, Y)$.  These pairs are fed to a ConvNet, which learns representations that can later be used for transfer learning and other downstream tasks.  The diagram shows a cyclic arrow indicating repeated annotation and model training.

In the classic supervised pipeline, **manual annotation** constitutes the primary bottleneck: every new sample must be examined by a human, which is costly, slow, and often infeasible for large‑scale datasets.  The resulting labeled dataset $(X, Y)$ is then used to train a model with a loss such as cross‑entropy.  While effective, this approach scales poorly with the desired data volume.

#### Self‑Supervised Alternative  

> **Figure description (self‑supervised workflow).** Images of animals (a dog, two cats, a puppy) enter an “Automated” block (highlighted in green).  The block produces predictions (“Cat” or “Dog”) that are turned into pseudo‑labels, forming synthetic pairs $(X, Y)$.  These pairs train a ConvNet, which outputs learned representations.  A large blue circular arrow indicates a feedback loop: the learned representations are fed back into the automated prediction process, allowing the system to improve its own pseudo‑labeling over time.  The final representations can be employed for transfer learning on downstream tasks.

The self‑supervised pipeline **eliminates human annotation** by leveraging structural regularities in the data itself (e.g., spatial continuity, temporal coherence).  Automated pretext tasks generate pseudo‑labels on the fly, producing a virtually unlimited supply of training pairs.  Moreover, the feedback loop allows the model to refine its own predictions: as the encoder improves, the quality of the pseudo‑labels increases, leading to a virtuous cycle of representation learning.  Consequently, SSL dramatically reduces the annotation cost while still yielding high‑quality features suitable for downstream applications.

---

### Pretext Tasks Overview  

Self‑supervised learning relies on a diverse set of *pretext tasks* that each exploit a different type of inherent data relationship.  The taxonomy presented in the figure groups these tasks into four broad categories, each enclosed in a distinct colored rounded rectangle.

1. **Generation‑Based Methods (teal).**  
   - *Image Generation*: The model learns to synthesize realistic images from latent codes (e.g., auto‑encoders, GANs).  
   - *Video Generation*: Extends image generation to the temporal domain, requiring the model to predict future frames conditioned on past frames.

2. **Context‑Based Methods (orange).**  
   - *Spatial Context Structure*: Tasks such as jigsaw puzzle solving or predicting the relative position of image patches force the encoder to understand spatial arrangement.  
   - *Temporal Context Structure*: Predicting the order of video clips or forecasting future frames leverages the temporal continuity of sequences.

3. **Free Semantic Label‑Based Methods (green).**  
   - *Semantic Label Segmentation*: The model predicts semantic segmentation masks derived from inexpensive cues (e.g., edge detection, saliency).  
   - *Depth Estimation*: Uses geometric cues to infer a depth map from a single image, providing a supervisory signal without ground‑truth depth annotations.

4. **Cross‑Modal‑Based Methods (purple).**  
   - *Flow‑RGB Correspondence*: Aligns optical flow information with RGB frames, encouraging the network to learn motion‑aware features.  
   - *Audio‑Visual Correspondence*: Forces the model to associate sound clips with matching video frames, thereby exploiting the natural synchronization between modalities.

> **Figure description.** The diagram visualizes the four categories as colored clusters, each containing specific pretext tasks connected by arrows that indicate a hierarchical or functional relationship (e.g., “Image Generation” → “Video Generation” under Generation‑Based Methods).  The overall layout emphasizes that SSL can be instantiated through a wide spectrum of tasks, each tailored to the modality and structure of the available data.

By selecting an appropriate pretext task—or a combination thereof—researchers can guide a neural network to learn representations that capture the most informative aspects of the data, even in the complete absence of human‑provided labels.  These representations become valuable foundations for downstream problems ranging from classification and detection to segmentation and reinforcement learning.

## Image-based SSL for Representation Learning

### Generative

#### Image Colorization

In image colorization, the goal is to predict the color version of an image given its grayscale representation. The data generation process involves creating pairs of images where each pair consists of a color image and its corresponding grayscale version. The figure displays such pairs, arranged in a grid, with each pair connected by an arrow indicating the transformation from color to grayscale. The pretext task for this process is defined by an $l_2$ loss function, which measures the difference between the generated color image and the original color image.

The figure also depicts a convolutional neural network (CNN) architecture used for image colorization. The process begins with an initial grayscale image, which is input to a CNN encoder. The output of the encoder is then fed into a CNN decoder, which produces a predicted colorized image. The predicted image is compared to the actual colorized image using a loss function, which is used to train the network. This architecture allows the network to learn the mapping from grayscale to color images, effectively colorizing the input grayscale images.

#### Image Inpainting

Image inpainting is a technique used to reconstruct missing or damaged parts of an image. The data generation process for image inpainting involves creating images with masked regions, where the goal is to reconstruct the missing content based on the surrounding context. The figure shows a photograph of a football game with a large white rectangular region obscuring a portion of the scene, illustrating the scenario for image inpainting. The pretext task for this process involves using a Generative Adversarial Network (GAN) to fill in the missing regions with content that is visually consistent with the surrounding image.

The figure illustrates the architecture of a GAN applied to image inpainting. The input image, which contains a masked region, is fed into a generator component. The generator outputs a predicted image, which attempts to fill in the missing region with content consistent with the surrounding context. Both the predicted and actual images are then inputted into a discriminator component, which performs a binary classification to determine whether the input image is real or fake. The loss from this classification is then fed back to the generator to improve its performance, effectively training the network to reconstruct the missing regions accurately.

### Spatial Context

#### Solve Jigsaw Puzzle \[@Doersch15\]

The solve jigsaw puzzle task involves using a convolutional neural network (CNN) to predict the correct permutation of image patches. The figure depicts a photograph of a cat lying on a multi-colored blanket, with red dashed boxes overlaid on various parts of the cat and blanket, indicating regions used in the jigsaw puzzle task. The equation $X = (\text{image fragment 1}, \text{image fragment 2})$; $Y = 3$ suggests that 'X' represents a pair of image fragments, and 'Y' is assigned a value of 3, potentially representing the puzzle complexity or a score.

The figure also shows the architecture of a CNN used in this task. The network consists of convolutional layers ($conv_i$), pooling layers ($pool_i$), and fully connected layers ($fc_i$). Each convolutional layer is labeled with its kernel size, number of feature maps, and input channel count. The diagram shows two identical network branches, labeled “Patch 1” and “Patch 2” at the input, processing separate patches. The network learns to recognize the permutations of these patches, with the goal of solving the jigsaw puzzle.

**Attention:** Trivial solution possible

- Boundary patterns and continuing textures can lead to trivial solutions. To avoid this, large enough gaps should be used.
- Chromatic aberration can also be an issue. To mitigate this, images can be pre-processed by shifting green and magenta toward gray, or randomly dropping 2 color channels.

#### Solve Jigsaw Puzzle++ \[@Noroozi16\]

The solve jigsaw puzzle++ task involves using a CNN to predict the correct permutation of nine image patches. The figure depicts a CNN architecture designed to solve this task, with the input consisting of nine image patches arranged in a $3 \times 3$ grid. The CNN is composed of multiple convolutional layers, followed by fully connected layers labeled “fc7” and “softmax”. The convolutional layers utilize shared weights, and a permutation set is used to generate different patch arrangements. The network learns to recognize these permutations, with the goal of predicting the correct arrangement of the nine patches.

The table presents the performance of the solve jigsaw puzzle++ task, with columns for the number of permutations, average hamming distance, minimum hamming distance, jigsaw task accuracy, and detection performance. The data shows that as the number of permutations decreases, the jigsaw task accuracy increases, but the detection performance may vary. This illustrates the trade-off between the complexity of the task and the accuracy of the predictions.

#### Rotation \[@Gidaris18\]

The rotation task involves using a CNN to predict the rotation angle of an object within an image. The figure depicts a process for determining the rotation angle of a bird, with four rotated versions of the same image created, representing 0, 90, 180, and 270-degree rotations. Each rotated image is fed into a CNN model, and the output is used to maximize the probability of a specific rotation angle. The figure illustrates how a CNN can be employed to learn rotation-invariant features or to explicitly predict the rotation angle of an object.

### Context Similarity

#### Distortions \[@Dosovitskiy16\] (Exemplar-CNN)

The distortions task involves creating distorted images from a single input image to form a class of surrogate images for a larger set of pseudo-classes. The figure displays a grid of images showcasing variations of animal depictions, with a red bounding box highlighting the first image. The goal is to create $N$ distorted images for each input patch, with all these distorted images forming one class. The network then discriminates between a set of surrogate classes, using the distorted images as training data.

#### Clustering \[@Christlein17ICDAR\]

The clustering task involves training a ResNet network using clustered features extracted from images. The figure depicts a pipeline for this process, beginning with identifying keypoints in an image and extracting $32 \times 32$ pixel patches centered around those keypoints. SIFT features are then computed for each patch, and these features are clustered using $k$-means, resulting in $N=5000$ clusters. The cluster indices are used as targets for ResNet training, with intermediate features or patches located between clusters removed before proceeding to the ResNet training stage.

#### Clustering \[@Caron18\] (DeepCluster)

The DeepCluster task involves a semi-supervised learning pipeline using a CNN and clustering. The figure illustrates this process, with an input set of diverse images fed into the CNN. The CNN's output is used for $k$-means clustering on PCA-whitened and $\ell_2$-normalized features. The process cycles between CNN training and clustering, producing pseudo-labels for the input images. The goal is to avoid trivial solutions by re-assigning empty clusters and weighting the contribution of an input by the inverse of the size of its assigned cluster.

#### Clustering \[@Asano20\]

The clustering task using optimal transport involves generating an optimal matrix $Q$ that allocates $N$ unlabeled images into $K$ clusters. The figure depicts a matrix $Q$ with dimensions $K$ rows by $N$ columns, where each element $Q_{ij}$ is either 0 or 1, signifying the assignment of image $j$ to cluster $i$. The equipartition constraint ensures that each cluster contains an equal number of images. The cost matrix is given by the model performance when trained using these clusters as the labels. The fast-variant of the Sinkhorn-Knopp algorithm is used to generate the optimal matrix $Q$, with a single matrix-vector multiplication that scales linearly with the number of images $N$.

**Self-labelling with Optimal Transport**

- **Problem:** Generate optimal matrix $Q$ that allocates $N$ unlabeled images into $K$ clusters.
- **Equipartition constraint:** The unlabeled images should be divided equally into the $K$ clusters.
- **Cost matrix:** Cost of allocating each image to a cluster is given by the model performance when trained using these clusters as the labels.
- **Fast-variant of Sinkhorn-Knopp algorithm:**
  - Single matrix-vector multiplication.
  - Scales linearly with the number of images $N$.

**Comparison to DeepCluster**

- No separate clustering loss can lead to degenerate solutions.
- Clustering approach that minimizes the same cross-entropy loss that the network also optimizes.

#### Multi-task SSL using Synthetic Imagery \[@Ren18\]

The multi-task SSL using synthetic imagery task involves using both synthetic and real-world images for multi-task self-supervised learning. The figure depicts a system that takes input from both synthetic and real-world images, processing them through a base module with shared weights. The synthetic stream branches into three output predictions: surface normal, depth, and domain D. The real-world stream is directly connected to the domain D prediction. The goal is to minimize the feature space domain differences between real and synthetic data, using a contrastive loss to align the feature spaces.

- Given: input synthetic RGB image.
- Network simultaneously predicts: surface normal, depth, instance contour.
- Additionally: minimize feature space domain differences between real and synthetic data.

### Contrastive SSL

#### Contrastive Learning

Contrastive learning involves learning representations by distinguishing between positive and negative sample pairs. The figure depicts a simple flowchart where an input $x_0$ and a positive sample $x_1$ are fed into a contrastive learning module, which outputs whether the samples are the same or different. This approach has advantages over generative and context models, as pixel-level losses could overly focus on pixel-based details rather than more abstract latent factors. Pixel-based objectives often assume pixel independence, reducing the ability to model correlations or complex structure.

#### Contrastive Loss

The contrastive loss, also known as InfoNCE loss, aims to maximize the similarity between a positive sample pair and minimize the similarity between a sample and its negative samples. The goal is to ensure that the similarity between the positive sample pair is greater than the similarity between the sample and any negative sample. The contrastive loss is formulated as a cross-entropy loss for an $(N)$-way softmax classifier, with the loss function given by:

$$\begin{align*}
        \mathcal{L}_{N}
        &=-\mathbb{E}_{\mathcal{X}}
\left[\log \frac{\exp \left(
    \textcolor{darkgreen}{s(f(\bx),f(\bx^{+}))}
\right)}
{\exp
    \left(
    \textcolor{darkgreen}{s(f(\bx),f(\bx^{+}))}
    \right) +
    \sum_{j=1}^{N-1} \exp \left(
    \textcolor{red}{s(f(\bx),f(\bx_{j}^{-}))}
\right)}\right]\\
&= 
-\mathbb{E}_{\mathcal{X}}
\left[\log
    \frac{\exp
        \left(
            \textcolor{darkgreen}{s(f(\bx), f(\bx^{+}))}
        \right)}
            {               \sum_{j=1}^{N} \exp
                \left(s(f(\bx),f(\bx_{j}))\right)
            }
\right]
\end{align*}$$

A common variation of the contrastive loss includes a temperature hyperparameter $\tau$:

$$\begin{equation*}
        \mathcal{L}_{N} =
-\mathbb{E}_{\mathcal{X}}
\left[\log
    \frac{\exp
        \left(
            \textcolor{darkgreen}{s(f(\bx), f(\bx^{+}))} / \tau
        \right)}
            { 
                \sum_{j=1}^{N+1} \exp
                \left(s(f(\bx),f(\bx_{j})) / \tau\right)
            }
\right]
\end{equation*}$$

#### Effectivity of Contrastive Loss

The figure illustrates the effectivity of the contrastive loss, showing how it can be used to learn meaningful representations by distinguishing between positive and negative sample pairs. The contrastive loss is effective in capturing the underlying structure of the data, making it a powerful tool for self-supervised learning.

#### Examples: SimCLR \[@Chen20\]

SimCLR is a method for contrastive learning that involves creating positive pairs by applying different data augmentation operations to the same sample. The figure depicts a flowchart where an input sample $x$ is augmented twice to create two positive samples, $x_i$ and $x_j$. These samples are then processed through a base encoder $f$ to produce representations $h_i$ and $h_j$. A representation head $g$ is used to compute the contrastive loss, which maximizes the agreement between the positive pairs. The loss function for SimCLR is given by:

$$\mathcal{L}_{i, j}=-\log \frac{\exp \left(s\left(\mathbf{z}_{i},
    \mathbf{z}_{j}\right) / \tau\right)}{\sum_{k=1}^{2 n} \mathbf{1}_{[k \neq i]}
    \exp \left(s\left(\mathbf{z}_{i}, \mathbf{z}_{k}\right) /
    \tau\right)}$$

#### Examples: Prototypical Contrastive Learning \[@Li20\]

Prototypical contrastive learning involves learning representations by distinguishing between samples and their corresponding prototypes. The figure depicts a conceptual illustration of prototypical contrastive learning, with clusters of images representing different prototypes. The goal is to minimize the distance between instances and their corresponding prototypes while maximizing the distance to other prototypes. This approach can be combined with clustering to form a system like ProtoNCE, which iterates between clustering and contrastive learning to improve the representations.

**Idea:** Combine Clustering with Contrastive Learning

**Iterate:**

- **E-Step:** $k$-means clustering.
- **M-Step:** Contrastive loss between sample, associated cluster center, and all other clusters centers.

### Supervised Contrastive Learning

#### Supervised Contrastive Learning \[@Khosla20\]

Supervised contrastive learning extends the contrastive learning framework by incorporating class labels. The figure depicts a contrastive learning setup with two classes: dogs and cats. The goal is to pull images of the same class closer together in the embedding space and push images of different classes apart. This approach leverages the class labels to create more informative positive pairs, improving the quality of the learned representations.

#### Supervised Contrastive Loss

The supervised contrastive loss computes the loss between any sample $z_j$ having the same class as the anchor $z_i$. The loss function is given by:

$$\begin{equation*}
      L_{\text{sup}} = \sum_{i=1}^{2N}
        -\ldots {\color{red}\sum_{j=1}^{2N} \mathbb{1}_{i\neq j} \cdot \mathbb{1}_{\by_i=\by_j}} \cdot
        \log \frac{
          \exp\left(
            \bz_i^\top \bz_{j} / \tau
          \right)
        }{
          \sum_{k=1}^{2N} \mathbb{1}_{i\neq k} \cdot \exp\left(
            \bz_i^\top \bz_{k} / \tau
          \right)
        }
\end{equation*}$$

The vectors $z$ need to be normalized, and the gradient with respect to $w$ is high for hard positives and negatives and small otherwise, effectively incorporating a focal loss. For one positive and one negative, the loss simplifies to:

$$\begin{equation*}
                      L_{\text{sup}} \propto \lVert\bz_a - \bz_p\lVert^2 - \lVert\bz_a - \bz_n\lVert^2 + 2\tau
  \end{equation*}$$

This loss is common in siamese networks and has been shown to improve the stability of the learning process with respect to non-optimal hyperparameters.

#### Hyperparameter stability

The figure presents a boxplot comparison of hyperparameter stability across different components of a machine learning pipeline: augmentation, optimizer, and learning rate. The data shows that supervised contrastive learning (represented by purple boxes) has increased stability with respect to non-optimal hyperparameters compared to cross-entropy (represented by light blue boxes). This improved stability can lead to better performance and more reliable training.

- Increased stability with respect to non-optimal hyperparameters.
- Training is about 50% slower than cross-entropy.
- Enables unsupervised clustering in latent space, leading to new possibilities for semi-supervised learning and correction of label noise.

### Bootstrap SSL – A paradigm change

#### BYOL \[@Grill20\] Overview

BYOL (Bootstrap Your Own Latent) is a self-supervised learning method that avoids the need for negative pairs and contrastive loss. Instead, it uses a target network to provide a moving target for the online network, making it more resilient to changes in batch size and the set of image augmentations compared to its contrastive counterparts.

#### BYOL \[@Grill20\] Method

The BYOL method involves two networks: an online network and a target network. The online network generates a representation of the input image, which is then projected and used for prediction. The target network receives the same input image and generates a representation, which is used as the prediction target. The loss function is based on the mean squared error (MSE) between the $\ell^2$-normalized predictions, related to cosine distance. To prevent a trivial solution, a slow-moving average of the online network weights is used to update the target network.

- Two networks: **online** and **target** network.
- In theory: trivial solution possible (e.g., zero for all images).
- Use slow-moving average of the online network as the target network.
- Loss: MSE of $\ell^2$-normalized predictions (proportional to cosine distance).

#### SSL State of the Art

The figures present the top-1 accuracy of self-supervised learning (SSL) methods over time and as a function of the number of parameters in a model. The data shows a general increasing trend in top-1 accuracy for both groups of methods, with state-of-the-art methods consistently achieving higher accuracy. The performance improves with increasing model size for different SSL methods, and the figures visually compare their effectiveness against supervised learning.

#### Further Reading

For further reading on self-supervised learning, the following resources are recommended:

- Blogs:
  - <https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html>
  - <https://amitness.com/2020/02/illustrated-self-supervised-learning/>
  - <https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html>
- Others:
  - <https://github.com/jason718/awesome-self-supervised-learning>
  - <https://www.youtube.com/watch?v=7I0Qt7GALVk>

## References
#### References

## Bibliography

- **Asano20** — YM. et al. (2020) "Self-labelling via simultaneous clustering and representation learning." *International Conference on Learning Representations*. [https://openreview.net/forum?id=Hyx-jyBFPr](https://openreview.net/forum?id=Hyx-jyBFPr).
- **Caron18** — Caron et al. (2018) "Deep Clustering for Unsupervised Learning of Visual Features." *Computer Vision -- ECCV 2018*. [arxiv:http://arxiv.org/abs/1807.05520v2](http://arxiv.org/abs/1807.05520v2).
- **Chen20** — Chen et al. (2020) "A Simple Framework for Contrastive Learning of Visual Representations." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/2002.05709v3](http://arxiv.org/abs/2002.05709v3).
- **Christlein17ICDAR** — Christlein et al. (2017) "Unsupervised Feature Learning for Writer Identification and Writer Retrieval." *2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR)*. DOI: [10.1109/ICDAR.2017.165](https://doi.org/10.1109/ICDAR.2017.165).
- **Doersch15** — Doersch et al. (2015) "Unsupervised Visual Representation Learning by Context Prediction." *2015 IEEE International Conference on Computer Vision (ICCV)*. DOI: [10.1109/ICCV.2015.167](https://doi.org/10.1109/ICCV.2015.167).
- **Dosovitskiy16** — Dosovitskiy et al. (2016) "Discriminative Unsupervised Feature Learning with Exemplar Convolutional Neural Networks." *IEEE Transactions on Pattern Analysis and Machine Intelligence*. DOI: [10.1109/TPAMI.2015.2496141](https://doi.org/10.1109/TPAMI.2015.2496141).
- **Gidaris18** — Gidaris et al. (2018) "Unsupervised Representation Learning by Predicting Image Rotations." *International Conference on Learning Representations*. [https://openreview.net/forum?id=S1v4N2l0-](https://openreview.net/forum?id=S1v4N2l0-).
- **Grill20** — Grill et al. (2020) "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/2006.07733v3](http://arxiv.org/abs/2006.07733v3).
- **Jing19** — Jing & Tian (2019) "Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/1902.06162v1](http://arxiv.org/abs/1902.06162v1).
- **Khosla20** — Khosla et al. (2020) "Supervised Contrastive Learning." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/2506.04411v2](http://arxiv.org/abs/2506.04411v2).
- **Li20** — Li et al. (2020) "Prototypical Contrastive Learning of Unsupervised Representations." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/2005.04966v5](http://arxiv.org/abs/2005.04966v5).
- **Noroozi16** — Noroozi & Favaro (2016) "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles." *Computer Vision -- ECCV 2016*. [arxiv:http://arxiv.org/abs/1603.09246v3](http://arxiv.org/abs/1603.09246v3).
- **Oord18** — van den Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/1807.03748v2](http://arxiv.org/abs/1807.03748v2).
- **Pathak16** — Pathak et al. (2016) "Context Encoders: Feature Learning by Inpainting." *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [arxiv:http://arxiv.org/abs/1604.07379v2](http://arxiv.org/abs/1604.07379v2).
- **Poole19** — Poole et al. (2019) "On Variational Bounds of Mutual Information." *Proceedings of the 36th International Conference on Machine Learning*. [http://proceedings.mlr.press/v97/poole19a.html](http://proceedings.mlr.press/v97/poole19a.html).
- **Ren18** — Ren & Lee (2018) "Cross-Domain Self-Supervised Multi-task Feature Learning Using Synthetic Imagery." *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*. DOI: [10.1109/CVPR.2018.00086](https://doi.org/10.1109/CVPR.2018.00086).
- **Wang20** — Wang & Isola (2020) "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." *arXiv e-prints*. [arxiv:http://arxiv.org/abs/2005.10242v10](http://arxiv.org/abs/2005.10242v10).