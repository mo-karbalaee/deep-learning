---
course: deep-learning
title: Common Practices
source: deep-learning/slides/5_CommonPractices/5_CommonPractices.tex
---

# Common Practices

## Recap

### Training a Neural Network
In the previous lectures we covered the fundamental components required to train a neural network. These “nuts and bolts” can be summarized as follows:

- **Fully connected and convolutional layers** – the building blocks that define how information is transformed and propagated through the network. Fully connected layers perform a linear mapping of all input units to all output units, whereas convolutional layers apply learnable filters locally, exploiting spatial structure in data such as images.  
- **Activation function** – a non‑linear mapping applied element‑wise after each linear operation. Common choices include ReLU, sigmoid, and tanh, each influencing gradient flow and model expressivity.  
- **Loss function** – a scalar measure of discrepancy between the network’s predictions and the true targets. The loss guides learning; typical examples are cross‑entropy for classification and mean‑squared error for regression.  
- **Optimization** – an algorithmic procedure (e.g., stochastic gradient descent, Adam) that iteratively updates the network parameters to minimize the loss.  
- **Regularization** – techniques that constrain the model’s capacity to improve generalization, such as weight decay, dropout, or data augmentation.

Having established these components, we now turn to the practical question of **how to select an architecture, train it, and evaluate its performance** in a rigorous manner.

---

### First Things First: Test Data
A central principle of sound machine‑learning practice is to keep the test dataset isolated from all stages of model development. This idea is often illustrated with a vault metaphor: imagine a massive, circular bank vault door made of thick steel, reinforced with multiple layers, bolts, and a barred gate. The door’s weight (22.12 tons) and thickness (22 inches) emphasize its resistance to tampering. Likewise, the test set should be stored in a “vault” that is only opened **once**, after the final model has been chosen, to obtain an unbiased estimate of its generalization performance.

> *“Ideally, the test set should be kept in a vault, and be brought out only at the end of the data analysis.”* – T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*  

The vault analogy underscores two crucial points:

1. **Preventing data leakage** – any exposure of test data during model selection can leak information about the true distribution, leading to overly optimistic performance estimates.  
2. **Ensuring reproducibility** – a sealed test set guarantees that future researchers can evaluate the same model under identical conditions.

#### Why the Test Set Must Remain Untouched
Neural networks are highly expressive models that can easily overfit even seemingly random labels. For example, training a deep network on ImageNet with randomized class assignments still yields near‑zero training error, demonstrating the capacity of modern architectures to memorize data [@randomLabels]. Consequently, if the test set is consulted during hyper‑parameter tuning or architecture search, the reported test error will **substantially underestimate** the true generalization error.

#### Proper Workflow for Model Selection
- **Never use the test set for architecture selection.** The choice of network depth, width, layer types, and other architectural decisions belongs to the model‑selection phase, which must rely only on training data and a *validation* set.  
- **Reserve a validation set** (or employ cross‑validation) to guide hyper‑parameter tuning, early stopping, and other design choices.  
- **Perform initial experiments on a smaller subset** of the training data. This allows rapid prototyping and debugging without expending computational resources on the full dataset. Once a promising configuration is identified, scale up training on the complete training set while still keeping the test set untouched.

By adhering to this disciplined separation of data, we obtain a reliable estimate of how the final model will perform on truly unseen inputs.

#### Implementation Sanity Checks Before Full‑Scale Training
Before committing resources to a full training run, it is good practice to verify that the code and the model behave as expected on a tiny problem:

These checks catch subtle implementation errors (e.g., mismatched tensor shapes or wrong sign conventions) that would otherwise manifest only after many epochs of training.

#### Overfitting a Small Subset as a Debugging Probe
A quick sanity test for a newly designed architecture is to ask whether it can **overfit a handful of samples** (e.g., 5–20). Train on this tiny subset until the training loss reaches (or approaches) zero. If the network fails to memorize the data, the likely culprits are:

- bugs in the forward or backward pass,
- insufficient model capacity,
- an unsuitable optimizer or learning‑rate choice.

Turning off regularization for this experiment is advisable, because regularization explicitly discourages memorization.

#### Monitoring Training Dynamics
During the main training run, continually inspect the behaviour of several quantities:

- **Loss curves** – look for smooth, monotonically decreasing training loss. Exploding or vanishing gradients appear as sudden spikes or plateaus.  
- **Learning‑rate effects** – large jumps in the loss often indicate a learning rate that is too high; noisy curves may suggest that the mini‑batch size is too small. Increasing the batch size generally reduces stochastic noise.  
- **Validation loss** – use the held‑out validation set as a proxy for the test loss. If training loss continues to fall while validation loss rises, the model is overfitting; consider stronger regularization or early stopping. Conversely, if both losses are high and close together, the model may be underfitting and could benefit from increased capacity or reduced regularization.  
- **Weight and activation statistics** – monitor the magnitude of weight updates (typical range ≈ 10⁻³) and check for pathological behaviours such as saturated activations or “dying ReLUs.” Visualising the first‑layer convolutional filters can reveal whether they evolve toward smooth, structured patterns rather than noisy artefacts.  

Recording these diagnostics enables rapid identification of optimisation problems and informs decisions about learning‑rate schedules (stepwise decay, exponential decay, or 1/t decay) and momentum settings.

#### Optimizer and Learning‑Rate Strategy
A pragmatic progression that works well in practice is:

1. **Start with stochastic gradient descent** (mini‑batch size chosen to balance gradient noise and computational efficiency) **plus momentum**. This combination provides stable updates while helping to traverse shallow local minima.  
2. **Fine‑tune the learning‑rate schedule** – begin with a relatively large learning rate and decay it over time (stepwise decay every *n* epochs, exponential decay, or 1/t decay). Observe the loss curve to confirm that the decay reduces oscillations without stalling progress.  
3. **Switch to an adaptive optimizer** such as Adam once a reasonable set of hyper‑parameters (batch size, momentum, weight decay) has been identified. Adam’s per‑parameter learning rates can accelerate convergence, especially on problems with heterogeneous gradient magnitudes.  

This staged approach leverages the interpretability of SGD‑based training while still benefiting from the speed of adaptive methods once the search space has been narrowed.

#### Summary of Practical Checklist
Putting the above points together yields a concise checklist for a disciplined training pipeline:

- ✔️ Verify gradient correctness with numerical checks (central differences, double precision).  
- ✔️ Confirm that random initialization yields a loss comparable to random guessing.  
- ✔️ Overfit a tiny subset to ensure the model can memorize data.  
- ✔️ Monitor training and validation loss curves, adjusting learning rate or batch size as needed.  
- ✔️ Track weight update magnitudes and first‑layer filter visualisations.  
- ✔️ Use a validation set for early stopping and hyper‑parameter tuning; keep the test set sealed until final evaluation.  
- ✔️ Begin with SGD + momentum, employ a sensible learning‑rate decay, then consider Adam for final training runs.  

Following this workflow helps to detect bugs early, avoid over‑optimistic performance estimates, and ultimately leads to more reliable and reproducible deep‑learning experiments.

## Training Strategies

### Before Training: Gradient Checks

When implementing a new loss function or a custom layer, it is essential to verify that the analytical gradients produced by back-propagation match the true numerical gradients. A common method is to compute a **finite-difference** approximation of the gradient and compare it to the analytically derived one.

*Use centered differences for the numeric gradient.*
For a scalar parameter $\theta$, the centered difference formula is

$$
\frac{\partial L}{\partial \theta} \approx
\frac{L(\theta + h) - L(\theta - h)}{2h},
$$

where $L$ is the loss and $h$ is a small perturbation. Centered differences are more accurate than forward or backward differences because the truncation error is $O(h^{2})$ rather than $O(h)$.

*Measure the discrepancy with a relative error.*
Instead of comparing absolute differences, compute

$$
\text{relative error} =
\frac{\left|\,\nabla_{\text{analytic}} - \nabla_{\text{numeric}}\,\right|}
     {\max\!\bigl( \,|\nabla_{\text{analytic}}|,\;|\nabla_{\text{numeric}}|,\,\varepsilon \bigr)},
$$

where $\varepsilon$ is a tiny constant to avoid division by zero. A relative error below, for example, $10^{-5}$ usually indicates a correct implementation.

*Numerical considerations.*

- Perform the check in **double precision** (64-bit floating point) to minimize round-off error.
- If the loss values are extremely small (e.g., $<1\times10^{-9}$), temporarily **scale the loss** by a constant factor so that the finite-difference computation is numerically stable.
- Choose the perturbation size $h$ carefully; typical values are $10^{-4}$ to $10^{-2}$, but the optimal $h$ may depend on the scale of the parameter.

*Practical recommendations.*

- Use **only a few data points** when performing gradient checks. This reduces the likelihood of encountering non-differentiable regions of the loss (e.g., due to clipping or discrete operations).
- Run the network for a **short warm-up** (a few iterations) before

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Common Practices – Part 1

## Optimizers & Learning Rates

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome everybody to today’s deep learning lecture! Today, we want to talk a bit about common practices. The stuff that you need to know to get everything implemented in practice,

So, I have a small outline over the next couple of videos and the topics that we will look at. So, we will think about the problems that we currently have and how far we went. We will talk about training strategies, again optimization and learning rates, and a couple of tricks on how to adjust them, architecture selection, and hyperparameter optimization. One trick that is really useful is ensembling. Typically people have to deal with the class imbalance and of course, there are also very interesting approaches to deal with this. So finally, we look into the evaluation and how to get a good predictor. We also estimate how well our network is actually performing.

So far, we have seen all the nuts and bolts of how to train the network. We have to fully connected and convolutional layers, the activation function, the loss function, optimization, regularization, and today we will talk about how to choose the architecture, train, and evaluate a deep neural network.

The very first thing is testing. “Ideally, the test data should be kept in a vault and be brought only out at the end of the data analysis.” as Hastie and colleagues are teaching in the elements of statistical learning.

So, first things first: Overfitting is extremely easy with neural networks. Remember the ImageNet random labels. The true test set error and generalization can be underestimated substantially when you use the test set for model selection. So, when we choose the architecture – typically the first element in the model selection – this should never be done on the test set. We can do initial experimentation on a smaller subset of the data to try to figure out what works. Never work on the test set when you’re doing these things.

Let’s look at a couple of training strategies: Before the training check your gradients, check the loss function, check own layer implementations that they compute results correctly. If you implemented your own layer, then compare the analytic and the numeric gradient. You can use central differences for the numeric gradient . You can use relative errors instead of absolute differences and consider the numerics. Use double precision for checking, temporally scale the loss function, and if you observe very small values, choose your h for the step size appropriately.

Then, we have a couple of additional and recommendations: If you only use a few data points, then you will have fewer issues with non-differentiable parts of the loss function. You can train the network for a short period of time and only then perform the gradient checks. You can check the gradient first, then with regularization terms. So, you first turn the regularization terms off, check the gradient, and in the end with the regularization terms. Also, turn off data augmentation and drop out. So, you typically make this check on rather small data sets.

The goal of the initialization is that you have a correct random initialization of the layers. So, you can compute the loss for each class on the untrained network with regularization turned off and of course, that should give a random classification. So here, one can compare the loss with the loss achieved when deciding for class randomly. They should be the same because you randomly initialize. Repeat this with multiple random initializations just to check that there’s nothing wrong with the initialization.

Let’s go to training. First, you check whether the architecture is in general capable of learning the task. So, before training the network on the full data set, you take a small subset of the data. Maybe five to 20 samples and then try to overfit the network to get a zero loss. With such few samples, you should be able to memorize the entire data set. Try to get a zero loss. Then, you know that your training procedure actually works and you can really go down to the zero loss. Optionally, you can turn off the regularization because it may hinder this overfitting procedure. Now, if the network can’t overfit, you may have a bug in the implementation, or your model may be too small. So, you may want to increase the parameters / the model capacity or simply the model may not be suitable for this task. Also, get a first idea about how the data, the loss, and the network behave.

Remember, we should monitor the loss function. These are typical loss curves. Make sure you don’t have an exploding or vanishing gradient. You want to have the appropriate learning rate, so check the learning rate to identify large jumps in the learning curve. If you have very noisy curves, try to increase the batch size. Noisy loss curves can be associated with too small mini-batches.

Next, get a validation data set and monitor the validation loss. You remember, this image here: Over the epochs, your training loss will, of course, go down but the test loss would go up. You never compute, of course, this on the test data set but you take the validation set as a surrogate for the test loss. Then, you can identify whether overfitting occurs in your network. If training and validation diverge, you have overfitting. So, you may want to increase the regularization or try early stopping. If training and validation loss are close but very high, you may have underfitting. So, decrease the regularization and increase the model size. You may want to save intermediate models because you can use them for testing later.

Further, during training monitor the weights and the activations. Keep track of the relative magnitude of the weight update. They should be in a sensible range, maybe 10⁻³. In the convolutional layers, you can check the filters of the first few layers. They should develop towards smooth and regular filters. You may want to check that. You want to get filters like here, on the right-hand side. The ones on the left-hand side, contain considerable amounts of noise and this may be not very reliable features. You may start building a noise detector here. So this can be a problem. Also, check for largely saturated activations. Keep in mind that dying ReLUs may happen.

So let’s look a bit at optimization and the learning rate. You want to choose an optimizer. Now, batch gradient descent requires large memory, is too slow, and has too few updates. So what people go for is typically stochastic gradient descent. Here, the loss function and the gradient become very noisy, in particular, if you only use one of your samples. You want to go with the mini-batch. The mini-batch is the best of both worlds. It has frequent but stable updates and the gradient is noisy enough to escape local minima. So, you want to adapt the mini-batch size to yield smoother or more noisy gradients, depending on your problem and the optimization. In addition, you may want to use momentum to prevent oscillations and speed up the optimization. The effect of hyper-parameters is relatively straightforward. The recommendation from us is you start with mini-batch, gradient descent, and momentum. Once, you have a good parameter set, you then change to Adam or other optimizers that can optimize the different weights with an adaptive learning rate.

Keep in mind to observe the loss curve. If your learning rate is not set correctly, you have trouble in the training of the network. For almost all gradient-based optimizers, you have to set η. So, we often see that directly in the lost curve, but this is a simplified view. So we actually want to have an adaptive learning rate and then progressively have smaller steps to find the optimum. So as we already discussed, you want to anneal the learning rate.

Now, the learning rate decay is yet another hyper-parameter that you have to set somehow. You want to avoid oscillations as well as a too-fast cooldown. So, there’s a couple of decay strategies. Stepwise decay every n epochs, you reduce the learning rate by a certain factor, like 0.5, a constant value like 0.01, or you reduce the learning rate when the validation error is no longer reducing. There’s exponential decay at every epoch where you actually use this exponential function here that can control the decay. There’s also the 1/t decay that at epoch t, you essentially scale the initial learning rate with 1 / (1 + kt ). The stepwise decay is most common and also the hyper-parameters are easy to interpret. Second-order methods are currently uncommon in practice as they don’t scale very well. So much about learning rates and a couple of the associated hyper-parameters.

Next time in deep learning, we will look further into how to adjust all those hyper-parameters that we’ve just discovered. You will find those hints to be really valuable for your own experimentation. So thank you very much for listening and see you in the next lecture!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] M. Aubreville, M. Krappmann, C. Bertram, et al. “A Guided Spatial Transformer Network for Histology Cell Differentiation”. In: ArXiv e-prints (July 2017). arXiv: 1707.08525 [cs.CV]. [2] James Bergstra and Yoshua Bengio. “Random Search for Hyper-parameter Optimization”. In: J. Mach. Learn. Res. 13 (Feb. 2012), pp. 281–305. [3] Jean Dickinson Gibbons and Subhabrata Chakraborti. “Nonparametric statistical inference”. In: International encyclopedia of statistical science. Springer, 2011, pp. 977–979. [4] Yoshua Bengio. “Practical recommendations for gradient-based training of deep architectures”. In: Neural networks: Tricks of the trade. Springer, 2012, pp. 437–478. [5] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016). [6] Boris T Polyak and Anatoli B Juditsky. “Acceleration of stochastic approximation by averaging”. In: SIAM Journal on Control and Optimization 30.4 (1992), pp. 838–855. [7] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. “Searching for Activation Functions”. In: CoRR abs/1710.05941 (2017). arXiv: 1710.05941. [8] Stefan Steidl, Michael Levit, Anton Batliner, et al. “Of All Things the Measure is Man: Automatic Classification of Emotions and Inter-labeler Consistency”. In: Proc. of ICASSP. IEEE – Institute of Electrical and Electronics Engineers, Mar. 2005.

---

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Common Practices – Part 2

## Hyperparameters and Ensembling

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome everybody to deep learning! So, today we want to look into further common practices and in particular, in this video, we want to discuss architecture selection and hyperparameter optimization.

Remember the test data is still in the vault. We are not touching it. However, we need to set our hyperparameters somehow and you’ve already seen that there’s an enormous amount of hyperparameters.

You have to select an architecture, a number of layers, number of nodes per layer, activation functions, and then you have all the parameters in the optimization: the initialization, the loss function, and many more. The optimizers still have options like the type of gradient descent, momentum, learning rate decay, and batch size. In regularization, you have different regularizes L2 and L1 loss, batch normalization, dropout, and so on. You want to somehow figure out all the parameters for those different kinds of procedures.

Now, let’s choose an architecture and loss function. The first step would be to think about the problem and the data. How could features look like? What kind of spatial correlation do you expect? What data augmentation makes sense? How will the classes be distributed? What is important regarding the target application? Then you start with simple architectures and loss functions and of course, you do your research. Try well-known models first and foremost. They are being published and there are so many papers out there. Hence, there is no need to do everything yourself. One day in the library can save hours, weeks, and months of experimentation. Do the research. It will really save you time. Often they just don’t publish the paper, but in very good papers, it’s not just the scientific result, but they also share source code, Sometimes even data. Try to find those papers. This can help you a lot with your own experimentation. So, then you may want to change and adapt the architecture to your problem that you found in the literature. If you change something, find good reasons why this is an appropriate change. There are quite a few papers out there that seem to introduce random changes into the architecture. Later, it turns out that the observations that they made were essentially random and they were just lucky or experimented enough on their own data in order to get the improvements. Typically, there’s also a reasonable argument of why the specific change should give an improvement in performance.

Next, you want to do your hyperparameter search. So you remember learning rate decay, regularization, dropout, and so on. These have to be tuned. Still, the networks can take days or weeks to train and you have to search for these hyperparameters. Hence, we recommend using a log scale. So, for example for η here you go for 0.1, 0.01, and 0.001. You may want to consider a grid search or random search. So in a grid search, you would have equal distance steps and if you look here at reference [2], they have shown that random search has advantages over the grid search. First of all, it’s easier to implement, and second, it has a better exploration of the parameters that have a strong influence. So, you may want to look into that and then adjust your strategy accordingly. So hyperparameters are highly interdependent. You may want to use a coarse to fine search. You optimize on a very coarse scale in the beginning and then make it finer. You may only train the network for a few epochs and then bring all the hyperparameters in sensible ranges. Then, you can refine using random and grid search.

A very common technique that can give you a little bit of boost of performance is ensembling. This is also something that can really help you to get this additional little bit of performance that you still need. So far, we have only considered a single classifier. Ensembling has the idea to use many of those classifiers. If we assume N classifiers that are independent, performing a correct prediction will be at a probability of 1 – p. Now, the probability of seeing k errors is N choose k times p to the power of k times (1 – p ) to the power of (N – k ). This is a binomial distribution. So, the probability of a majority meaning k > N /2 to be wrong is the sum over N choose k times p to the power of k times (1 – p ) to the power of ( N – k ).

So, we visualize this in the following plot here. In this graph, you can see that if I take more of those weak classifiers, we get stronger. Let’s set for example their probability of being wrong to 0.42. Now, we can compute this binomial distribution. Here, you can see that if I choose 20, I get a probability of approximately 14% that the majority is wrong. If I choose N=32, I get less than 10% probability that the majority is wrong. If I choose 70, more than 95% of the cases the majority will be correct. So, you see that this probability is monotonically decreasing for large N. If n approaches infinity, then the accuracy will go towards 1. The big problem here is, of course, independence. So typically, we have problems generating from the same data, independent classifiers. So, if we had of course enough data then we can train many of those independent weak classifiers. So, how do we then implement this as a concept? We somehow have to produce N independent classifiers or regressors and then we combine the predictions by majority or averaging. How can we actually produce such components?

Well, you can choose different models. So in the example here, we have seen that we have a non-convex function. Obviously, they have different local minima. So, the different local minima would result in different models and then we can combine them. Also, what you can try is a cyclic learning rate where you then go up and down with the learning rate to escape certain local minima. This way, you can then try to find different local minima and store them for ensembling.

This way, you could also take different model checkpoints that you extract at different points in the optimization. Later, you can reuse them for ensembling. Also, a moving average of weights w can generate new models. You could even go this far and combine different methods. So, we still have the entire catalog of traditional machine learning approaches that you could also train and then combine them with your new deep learning model. Typically, this is an easy boost of performance if you need just a little bit more. By the way, this was also the idea that finally helped people to break the Netflix challenge. The first two teams that almost broke the challenge, they teamed up and trained an ensemble. This way they broke the challenge together.

So next time in deep learning, we will talk about class imbalance, a very frequent problem, and how to deal with that in your training procedure. Thank you very much for listening and goodbye!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] M. Aubreville, M. Krappmann, C. Bertram, et al. “A Guided Spatial Transformer Network for Histology Cell Differentiation”. In: ArXiv e-prints (July 2017). arXiv: 1707.08525 [cs.CV]. [2] James Bergstra and Yoshua Bengio. “Random Search for Hyper-parameter Optimization”. In: J. Mach. Learn. Res. 13 (Feb. 2012), pp. 281–305. [3] Jean Dickinson Gibbons and Subhabrata Chakraborti. “Nonparametric statistical inference”. In: International encyclopedia of statistical science. Springer, 2011, pp. 977–979. [4] Yoshua Bengio. “Practical recommendations for gradient-based training of deep architectures”. In: Neural networks: Tricks of the trade. Springer, 2012, pp. 437–478. [5] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016). [6] Boris T Polyak and Anatoli B Juditsky. “Acceleration of stochastic approximation by averaging”. In: SIAM Journal on Control and Optimization 30.4 (1992), pp. 838–855. [7] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. “Searching for Activation Functions”. In: CoRR abs/1710.05941 (2017). arXiv: 1710.05941. [8] Stefan Steidl, Michael Levit, Anton Batliner, et al. “Of All Things the Measure is Man: Automatic Classification of Emotions and Inter-labeler Consistency”. In: Proc. of ICASSP. IEEE – Institute of Electrical and Electronics Engineers, Mar. 2005.

---

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Common Practices – Part 3

## Class Imbalance

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome back to deep learning! Today, we want to continue talking about our common practices. The methods that we are interested in today are about class imbalance. So, a very typical problem is that one class – in particular the very interesting one – is not very frequent. So, this is a challenge for all the machine learning algorithms.

Let’s take the example of fraud detection. Out of 10,000 transactions, 9,999 are genuine and only one is fraudulent. So, if you classify everything as genuine, you get 99.99% accuracy. Obviously, even in less severe situations, we are if you had a model and that would misclassify one out of a hundred transactions, then you would end up only in a model with 99% accuracy. This is of course a very hard problem. In particular, in screening applications, you have to be very careful because just classifying everything to the most common class would still get you very very good accuracy.

It doesn’t have to be credit cards, for example, here detecting mitotic cells is a very similar problem. A mitotic cell is a cell undergoing cell division. These cells are very important as we already heard in the introduction. If you count the cells under mitosis, you know how aggressively the associated cancer is growing. So this is a very important feature but you have to detect them correctly. They make up only a very small portion of the cells in tissues. So, the data of this class has been seen much less during the training, and measures like the accuracy, L2 norm, and cross-entropy don’t show this imbalance. So, they are not very responsive to this.

One thing that you can do is for example resampling. The idea is that you balance the class frequencies by sampling classes differently, So, you can understand this means that you have to throw away a lot of the training data of the most frequent classes. This way you get to train a classifier that will be balanced towards both of these classes. Now they are seen approximately as frequently as the other class. The disadvantage of this approach is that you’re not using all the data that is being seen and of course, you don’t want to throw away data.

So another technique is oversampling. You can just sample more often from the underrepresented classes. In this case, you can use all of the data. The disadvantage is of course that it can lead to heavy overfitting towards the less frequently seen examples. Also possible are combinations of under and oversampling.

This then leads to advanced resampling techniques that try to avoid the shortcomings of undersampling by a synthetic minority oversampling. It’s rather uncommon in deep learning. Underfitting caused by undersampling can be reduced by taking a different subset after each epoch. This is quite common and also you can use data augmentation to help reducing overfitting for underrepresented classes. So, you essentially augment more of the samples that you have seen less frequently.

Instead of fixing the data, of course, you can also try to adapt the loss function to be stable with respect to class imbalance. Here, you then choose a loss with the inverse class frequency. You can then create the weighted cross entropy where you introduce an additional weight w which is simply determined as the inverse class frequency. More common in segmentation problems are then things like the Dice loss based on the Dice coefficient . Here, you evaluate the loss on the dice coefficient that measures area overlap. It is a very typical measure for evaluating segmentations instead of class frequency. Weights can also be adapted with regards to other considerations but we are not discussing them here in this current lecture.

This already brings us to the end of this part and in the final section of common practices, we will now discuss measures of evaluation and how to evaluate our models appropriately. So, thank you very much for listening and goodbye!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] M. Aubreville, M. Krappmann, C. Bertram, et al. “A Guided Spatial Transformer Network for Histology Cell Differentiation”. In: ArXiv e-prints (July 2017). arXiv: 1707.08525 [cs.CV]. [2] James Bergstra and Yoshua Bengio. “Random Search for Hyper-parameter Optimization”. In: J. Mach. Learn. Res. 13 (Feb. 2012), pp. 281–305. [3] Jean Dickinson Gibbons and Subhabrata Chakraborti. “Nonparametric statistical inference”. In: International encyclopedia of statistical science. Springer, 2011, pp. 977–979. [4] Yoshua Bengio. “Practical recommendations for gradient-based training of deep architectures”. In: Neural networks: Tricks of the trade. Springer, 2012, pp. 437–478. [5] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016). [6] Boris T Polyak and Anatoli B Juditsky. “Acceleration of stochastic approximation by averaging”. In: SIAM Journal on Control and Optimization 30.4 (1992), pp. 838–855. [7] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. “Searching for Activation Functions”. In: CoRR abs/1710.05941 (2017). arXiv: 1710.05941. [8] Stefan Steidl, Michael Levit, Anton Batliner, et al. “Of All Things the Measure is Man: Automatic Classification of Emotions and Inter-labeler Consistency”. In: Proc. of ICASSP. IEEE – Institute of Electrical and Electronics Engineers, Mar. 2005.

---

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Common Practices – Part 4

## Performance Evaluation

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome everybody to the next part of deep learning! Today, we want to finish talking about common practices, and in particular, we want to have a look at the evaluation. Of course, we need to evaluate the performance of the models that we’ve trained so far. Now, we have set up the training, set hyperparameters, and configured all of this. Now, we want to evaluate the generalization performance on previously unseen data. This means the test data and it’s time to open the vault.

Remember “Of all things the measure is man”. So, data is annotated and labeled by humans and during training, all labels are assumed to be correct. But of course, to err is human. This means we may have ambiguous data. The ideal situation that you actually want to have for your data is that it has been annotated by multiple human voters. Then you can take the mean or a majority vote. There’s also a very nice paper by Stefan Steidl from 2005. It introduces an entropy-based measure that takes into account the confusions of human reference labelers. This is very useful in situations where you have unclear labels. In particular, in emotion recognition, this is a problem as also humans confuse sometimes classes like angry versus annoyed while they are not very likely to confuse “angry” versus “happy” as this is a very clear distinction. There are different degrees of happiness. Sometimes you’re just a little bit happy. In these cases, it is really difficult to differentiate happy from neutral. This is also hard for humans. In prototypes, if you have for example actors playing, you get emotion recognition rates way over 90%. If you have real data emotion and if you have emotions as they occur in daily life, it’s much harder to predict. This can then also be seen in the labels and in the distribution of the labels. If you have a prototype, all of the raters will agree that the observation is clearly this particular class. If you have nuances and not so clear emotions, you will see that also our raters will have a less peaked or even a uniform distribution over the labels because they also can’t assess the specific sample. So, mistakes by the classifier are obviously less severe if the same class is also confused by humans. Exactly this is considered in Steidl’s entropy-based measure.

Now, if we look into performance measures, you want to take into account the typical classification measures. They are typically built around the false negatives, the true negatives, the true positives, and the false positives. From that for binary classification problems, you can then compute true and false positive rates. This typically then leads to numbers like the accuracy that is the number of true positives plus true negatives over the number of positives and negatives. Then, there is the precision or positive predictive value that is computed as the number of true positives over the number of true positives plus false positives. There’s the so-called recall that is defined as the true positives over the true positives plus the false negatives. Specificity or true negative value is given as the true negatives over the true negatives plus the false positives. The F1 score is an intermediate way of mixing these measures. You have the true positive value times the true negative value divided over the sum of two positive and true negative value. I typically recommend the receiver operating characteristic (ROC) curves because all of the measures that you’ve seen above, they are dependent on thresholds. If you have the ROC curves, you essentially evaluate your classifier for all different thresholds. This then gives you an analysis of how well it performs in different scenarios.

Furthermore, there are performance measures in multi-class classification. These are adapted versions of the measures above. The top- K error is the probability of the true class label not being in the K estimates with the highest prediction score. Common realizations are the top-1 and top-5 error. ImageNet, for example, usually uses the top-5 error. If you really want to understand what’s going on in multi-class classification, I recommend looking at confusion matrices. Confusion matrices are useful for let’s say 10 to 15 classes. If you have a thousand classes, confusion matrices don’t make any sense anymore. Still, you can gain a lot of understanding of what’s happening if you look at confusion matrices in cases with fewer classes.

Now, sometimes you have very few data. So, in these cases, you may want to choose cross-validation. In k-fold cross-validation, you split your data into k folds, and then you use k – 1 folds as training data and test on Fold k. Then, you repeat it k times. This way, you have seen in the evaluation data all of the data but you trained on independent data because you held it out at the time of training. It’s rather uncommon in deep learning because it implies very long training times. You have to repeat the entire training K times which is really hard if you train for like seven days. If you have sevenfold cross-validation, you know you can do the math, it will take really long. If you use it for hyperparameter estimation, you have to nest it. Don’t perform cross-validation just on all of your data, select the hyperparameters, and then go ahead with the same data in testing. This will give you optimistic results. You should always make sure that if you select parameters you hold out the test data where you want to test on so there’s techniques for nesting cross-validation into cross-validation but then it will also become computationally very expensive so that’s even worse if you want to nest the cross-validation one thing that you have to keep in mind is that the variance of the results is typically underestimated because the training grants are not independent also pay attention that you may introduce additional bias by incorporating the architecture selection and hyperparameter selection so this should be done on different data and it’s very difficult if you’re working with cross-validation even without cross-validation training is a highly stochastic process therefore you may want to retrain your network multiple times with different initializations if you pick random initializations for example and then report the standard deviation just to make sure how well your training actually performs.

Now, you want to compare different classifiers. The question is: “Is my new method with 91.5% accuracy better than the state-of-the-art with 90.9%?” Of course, training a system is a stochastic process. So, just comparing those two numbers will yield biased results. The actual question that you have to ask is: “Is there a significant difference between the classifiers?” This means that you need to run the training for each method multiple times. Only then you can, for example, use a t-test to see whether the distribution of the results is significantly different. The t-test compares two normally distributed data sets with equal variance. Then, you can determine that the means are significantly different with respect to a significance level α which is the level of randomness. Quite frequently you find in literature like 5% or 1% significance level. So you have a significant difference if the chance of this observation being random is less than 5% or 1%.

Now be careful if you train multiple models on the same data. If you ask the same data a couple of times, you actually have to correct your significance computation. This is called the Bonferroni correction. If we compare multiple classifiers, this will introduce multiple comparisons and then you have to correct for this. If you had n tests with significance level α, then the total risk is n times α. So, to reach the total significance level of α, the adjusted α’ would be α over n for each individual test. So, the more tests you run on the same data, the more you have to divide by. Of course, this assumes independence between the tests and it’s a kind of pessimistic estimation of significance. But you want to be pessimistic in this case just to make sure that you are not reporting something that has been produced by chance. Just because you test often enough and your testing is a random process, there may be a very good result showing up just by chance. More accurate, but incredibly time-consuming would be permutation tests, and believe me, you probably want to go with the Bonferroni correction instead. Permuting everything will take even longer than the cross-validation approach that we’ve seen previously.

Okay so let’s summarize what we’ve seen before: You check your implementation before training, the gradient initialization, monitor the training process continuously, the training, the validation losses, the weights, and the activations. Stick to established architectures before reinventing the wheel. Experiment with little data and keep your evaluation data safe until the evaluation. Decay the learning rate over time. Do a random search, not a grid search for hyperparameters. Perform model ensembling for better performance, and when you check your comparison, of course, you want to go for significance tests to make sure that you are not reporting a random observation.

So next time on deep learning, we actually want to look at the evolution of neural network architectures. So from deep networks to even deeper networks. We want to have a look at sparse and dense connections and we’ll introduce a lot of common names, things you hear all over the place, LeNet, GoogLeNet, ResNet, and so on. So we will learn about many interesting state-of-the-art approaches in this next series of lecture videos. So, thank you very much for listening and see you in the next video!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

### Links

- Online t-test

- Online test for comparing recognition rates

- Online test to compare correlations

## References

[1] M. Aubreville, M. Krappmann, C. Bertram, et al. “A Guided Spatial Transformer Network for Histology Cell Differentiation”. In: ArXiv e-prints (July 2017). arXiv: 1707.08525 [cs.CV]. [2] James Bergstra and Yoshua Bengio. “Random Search for Hyper-parameter Optimization”. In: J. Mach. Learn. Res. 13 (Feb. 2012), pp. 281–305. [3] Jean Dickinson Gibbons and Subhabrata Chakraborti. “Nonparametric statistical inference”. In: International encyclopedia of statistical science. Springer, 2011, pp. 977–979. [4] Yoshua Bengio. “Practical recommendations for gradient-based training of deep architectures”. In: Neural networks: Tricks of the trade. Springer, 2012, pp. 437–478. [5] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016). [6] Boris T Polyak and Anatoli B Juditsky. “Acceleration of stochastic approximation by averaging”. In: SIAM Journal on Control and Optimization 30.4 (1992), pp. 838–855. [7] Prajit Ramachandran, Barret Zoph, and Quoc V. Le. “Searching for Activation Functions”. In: CoRR abs/1710.05941 (2017). arXiv: 1710.05941. [8] Stefan Steidl, Michael Levit, Anton Batliner, et al. “Of All Things the Measure is Man: Automatic Classification of Emotions and Inter-labeler Consistency”. In: Proc. of ICASSP. IEEE – Institute of Electrical and Electronics Engineers, Mar. 2005.

## Lecture Notes Sources

These integrated lecture notes were transcribed from voice recordings of the lecture (FAU LME). Follow the links for the original blog posts:

- [Common Practices Part 1](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-common-practices-part-1/)
- [Common Practices Part 2](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-common-practices-part-2/)
- [Common Practices Part 3](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-common-practices-part-3/)
- [Common Practices Part 4](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-common-practices-part-4/)
