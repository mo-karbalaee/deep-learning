---
course: deep-learning
title: Weakly and Self-Supervised Learning
source: deep-learning/slides/12_Weakly/12_Weakly.tex
---
# Weakly and Self-Supervised Learning

## Learning with Limited Annotations

### Supervised Learning

State‑of‑the‑art performance in computer vision has historically relied on two essential ingredients:

1. **Large quantities of training data** – modern deep networks require many examples to learn robust feature representations.
2. **Consistent, high‑quality annotations** – the labels must be accurate and detailed enough for the loss function to provide a useful learning signal.

An illustrative example is the output of an object‑detection model (Mask R‑CNN). The model localizes multiple objects in an outdoor scene by drawing bounding boxes around each detected instance and attaches a confidence score (e.g., 0.945 for the airplane body, 0.997 for its tail fin, 0.991–0.997 for each person). Different colors are used to distinguish the individual detections. The confidence scores represent the model’s estimated probability that the predicted class is correct. This demonstration underscores how precise, image‑level supervision enables both classification and localization.

---

### The Cost of Annotation

The amount of human effort required to produce different types of image annotations varies dramatically. Empirical measurements from the MS‑COCO dataset[^Lin14‑MSCoco] are reproduced below:

- **Image‑level class labels**: ≈ 20 seconds per image, and ≈ 27 seconds per image.
- **Instance spotting** (i.e., drawing a single bounding box around each object): ≈ +14 seconds on top of the image‑level label cost.
- **Instance segmentation** (pixel‑accurate masks for each object): ≈ +80 seconds beyond spotting.
- **Dense pixel‑level annotation** (full semantic segmentation of a scene): ≈ 1.5 hours per image[^Cordts16‑Citiscapes].

> **Figure (left)** – A wine‑cellar scene with a dog and a bottle. Two simple bounding boxes illustrate the difference between *image‑level labeling* (quick, coarse) and *instance spotting* (bounding boxes).
>
> **Figure (center)** – The same dog with a detailed polygon mask, visualizing the additional effort required for *instance segmentation*.
>
> **Figure (right)** – A street view from the Cityscapes dataset with dense, color‑coded segmentation masks for every pixel, exemplifying the most time‑consuming annotation level.

These illustrations make clear that the granularity of supervision is directly proportional to the annotation cost.

---

### Strongly vs. Weakly Supervised Learning

Supervision can be placed on a two‑dimensional continuum:

- **Horizontal axis**: from coarse **image‑level labels** to fine **pixel‑level masks**.
- **Vertical axis**: from weak **image-source information** (e.g., image tags) to strong **bounding‑box supervision**.

The lower‑right corner of the diagram corresponds to **strong supervision** (pixel‑accurate masks together with bounding boxes). The upper‑left corner represents **weak supervision**, where only image‑level tags are available and spatial localization is vague. Colored arrows of varying intensity indicate intermediate supervision levels, with darker arrows denoting more precise annotations. A blue circle marks an intermediate point on the continuum, emphasizing that many practical settings lie between the two extremes. The figure conveys that the richness of the label set directly influences the design of loss functions and ultimately the performance of object‑detection systems.

---

### Key Ingredients for Weakly Supervised Learning

Weak supervision is made feasible by exploiting **priors** (knowledge that is either explicitly encoded or implicitly learned) and **hints** (lightweight annotations that guide the learning process).

#### Priors (Explicit & Implicit)

- **Shape & size** – typical object silhouettes or expected aspect ratios.
- **Contrast** – objects often stand out from background in intensity or color.
- **Motion** – temporal continuity can reveal object boundaries in video.
- **Class distribution** – the relative frequency of classes can regularize predictions.
- **Similarity across images** – co‑occurring visual patterns suggest common objects.

#### Hints (Weak Labels)

- **Image‑level class tags** – indicate which categories are present.
- **Bounding boxes** – provide coarse spatial constraints.
- **Image captions** – textual descriptions that can be aligned with visual features.
- **Sparse temporal labels** – occasional frame‑wise annotations in video streams.
- **Scribbles** – rough strokes that roughly outline object extents.
- **Clicks inside objects** – single point annotations (e.g., a click on a toddler’s cheek) that signal object presence.

> **Figure** – A photograph of two children with four colored circles (cyan, magenta, dark blue) placed on their faces and on a baby bottle. The circles act as *point hints*, demonstrating that even minimal user input can greatly aid weakly supervised detectors such as Mask R‑CNN.
>
> **Figure** – A sleeping cat with a bright cyan contour drawn around its silhouette. The contour is a *scribble* that supplies a weak shape prior for segmentation without requiring a full pixel mask.

These priors and hints together form the “knowledge base” that compensates for the lack of dense supervision.

---

### From Labels to Localization – Approach 1: Use a Pre‑trained Classification Network

A common baseline is to repurpose a network that was originally trained for image‑level classification (e.g., VGG‑16[^Simonyan14]) to obtain spatial information:

1. **Visualize how input perturbations affect the classification score** – techniques such as guided back‑propagation or occlusion maps reveal which image regions contributed most to the decision.
2. **Interpret the resulting activation map as a coarse segmentation** – the map highlights discriminative areas but is not guaranteed to cover the full object extent.

Two fundamental problems arise:

- **Problem 1 – Lack of localization training**: The classifier was never exposed to pixel‑wise supervision, so its internal representations are optimized for *whether* a class is present, not *where* it appears.
- **Problem 2 – Good classifiers ≠ good maps**: High classification accuracy does not imply that the activation maps align well with object boundaries; they often focus on the most distinctive parts (e.g., the dog’s head) while ignoring the rest.

> **Figure** – A terrier dog image processed by a classification network. The network’s raw output is a grayscale activation map where brighter pixels indicate higher confidence for the “dog” class. This illustrates the limited spatial precision obtained from a naïve classifier‑to‑segmentation conversion.

---

### From Labels to Localization – Approach 2: Use a Classification Network, but Smarter

Zhou et al.[^zhou2016learning] introduced a principled way to extract localization cues from a classification model by **global average pooling (GAP)**:

1. **Replace the final fully‑connected layer with GAP** – each feature map of size \(m \times n\) is reduced to a single scalar by averaging over all spatial positions.
2. **Inspect the penultimate convolutional layer** – the feature maps retain spatial information.
3. **Compute Class Activation Maps (CAMs)** – each class’s score is a weighted sum of the GAP outputs; the same weights can be applied back to the feature maps to produce a heatmap that highlights class‑specific regions.
4. **Generalize with Gradient‑based CAM (Grad‑CAM)**[^Selvaraju16‑GradCAM] – back‑propagate the class score gradient to obtain importance weights for each feature map, allowing CAMs to be generated for any CNN architecture.

#### Class Activation Maps (CAM) Derivation

Given feature maps \(F_k(x, y)\) for channel \(k\) and GAP weights \(w_k^c\) for class \(c\), the class score is

\[
S_c = \sum_k w_k^c \cdot \frac{1}{m n}\sum_{x,y} F_k(x, y) .
\]

The corresponding CAM is obtained by linearly combining the original feature maps with the same weights:

\[
\text{CAM}_c(x, y) = \sum_k w_k^c \, F_k(x, y) .
\]

The heatmap \(\text{CAM}_c\) can be up‑sampled to the input resolution and overlaid on the image, revealing the regions that contributed most to the classification.

> **Figure** – A person holding an Australian terrier passes through a CNN consisting of several convolutional layers, a GAP layer, and a final linear classifier. The right‑most panel shows three example filter activations, their weighted combination (the CAM), and the CAM overlaid on the original image, highlighting the dog’s head and body.

---

#### Fully Convolutional Networks: Revisited

Traditional CNNs contain fully‑connected (FC) layers that fix the input size. Two strategies make networks fully convolutional and thus size‑agnostic:

- **Replace each FC layer by a convolution of spatial size \(m \times n\)** (the same spatial dimensions as the preceding feature map). The convolution’s kernel size matches the FC weight matrix, and the number of output channels equals the number of neurons in the original FC layer.
- **Alternatively, apply pooling (e.g., global average pooling) to collapse the spatial dimensions before the FC layer**, ensuring a fixed‑size vector regardless of the input resolution.

The accompanying schematic illustrates these ideas:

- **Left block**: two feature tensors of size \(m \times n\) with channel depth \(k\) are convolved (or pooled) to a \(1 \times 1\) tensor, effectively mimicking a fully connected operation.
- **Right block**: a pooling layer (“Pool”) reduces an \(m \times n\) tensor to \(1 \times 1\), after which two standard FC layers (both \(1 \times 1\)) produce the final class scores.

These modifications enable dense prediction (e.g., segmentation) without being constrained by a predetermined input size.

---

### From Bounding Boxes to Segmentation

Fully supervised semantic segmentation requires expensive pixel‑wise masks. An attractive alternative is to **learn segmentation solely from bounding‑box annotations**, which are far cheaper to obtain.

- **Expensive annotation**: manual masks (as in the cow image) demand considerable time.
- **Cheap annotation**: a simple rectangular box around a horse and rider (as in the horse‑rider image) can be collected quickly.
- **Weakly supervised scenario**: only the bounding box is provided; the interior of the box is ambiguous, as indicated by a large, blurred pink region with a question mark over a horse. The question mark visualizes the challenge of inferring precise object boundaries from coarse boxes.

> **Figure** – Three panels: (1) Fully supervised mask of a cow, (2) Bounding‑box annotation of a horse and rider, (3) Weakly supervised setting where a loose pink region and a question mark highlight the uncertainty inside a bounding box.

#### Exploiting Noise Robustness

Convolutional neural networks (CNNs) show a degree of robustness to label noise. This property can be leveraged as follows[^fromBoxesToSegmentation]:

1. **Treat the bounding box as an initial target mask** (all pixels inside the box = foreground, outside = background).
2. **Train a segmentation CNN on this noisy target**.
3. **Iteratively refine the target** by feeding the network’s predictions back as supervision for the next training round.

The process is visualized by a sequence of five images of a horse:

- **Input** – The original photograph with a pink bounding rectangle.
- **After 1 training round** – A coarse magenta mask that roughly follows the horse silhouette.
- **After 5 rounds** – The mask becomes tighter and discards obvious background.
- **After 10 rounds** – The mask closely matches the ground‑truth shape.
- **Ground truth** – The reference pixel‑accurate segmentation.

#### Post‑processing & Suppression

Training with boxes alone can quickly deteriorate due to accumulated errors. Proposed remedies include:

- **Suppress detections** that are unlikely to belong to the target class, lie outside the original box, occupy less than a certain percentage of the box area, or fall outside the borders of a Conditional Random Field (CRF) segmentation.
- **Use smaller, tighter boxes** because objects are on average roughly round; corners and edges typically contain fewer true positives.
- **Define “ignore” regions** where labels are unknown (e.g., outside the box or in uncertain border zones). These regions are excluded from loss computation, reducing penalization of ambiguous predictions.

> **Figure** – An illustration of a rider on a horse with a bounding‑box overlay, showing how false detections (outside the box or wrong class) are suppressed and how “ignore” zones are marked.
>
> **Figure** – A set of colored rectangles representing different ignore‑region categories (gray, dark gray, reddish‑pink, bright pink). The layout demonstrates how such regions can be partitioned spatially to guide the loss function.

#### Improved Recursive Training

A quantitative comparison of several segmentation strategies over multiple training rounds is depicted below. The **mean Intersection‑over‑Union (mIoU)** is plotted on the vertical axis (≈ 4

## Lecture Notes Sources

These integrated lecture notes were transcribed from voice recordings of the lecture (FAU LME). Follow the links for the original blog posts:

- [Weakly And Self Supervised Learning Part 1](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-weakly-and-self-supervised-learning-part-1/)
- [Weakly And Self Supervised Learning Part 2](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-weakly-and-self-supervised-learning-part-2/)
- [Weakly And Self Supervised Learning Part 3](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-weakly-and-self-supervised-learning-part-3/)
- [Weakly And Self Supervised Learning Part 4](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-weakly-and-self-supervised-learning-part-4/)
