---
course: deep-learning
title: Loss functions and Optimization
source: deep-learning/slides/2_LossAndOptimization/2_LossAndOptimization.tex
---

# Loss functions and Optimization

## Loss Functions

### Regression vs. classification  

In supervised learning two fundamental problem settings are distinguished. **Classification** aims to estimate a discrete variable (a class label) for every input sample. **Regression** aims to estimate a continuous variable for every input sample.  

The accompanying figure shows a two‑dimensional scatter plot with features $x_{1}$ (vertical axis) and $x_{2}$ (horizontal axis). Red dots belong to one class, blue dots to the other. A solid black line bisects the plot and separates the two classes. This line is a decision boundary learned by a binary classifier (e.g., a support‑vector machine or logistic regression) and visualises how the feature space is partitioned to assign new points to one of the two categories.  

A second figure depicts a similar scatter plot with a dashed black line that approximates a linear decision boundary. Some points lie exactly on the line and a few are mis‑classified. This illustration emphasizes the bias‑variance trade‑off when fitting a linear classifier to data that may not be perfectly linearly separable.

---

### Loss function vs. last activation function in a network  

The **last activation function** of a neural network is applied independently to each sample $x_{m}$ in a mini‑batch, both during training and testing. Its purpose is to map the network’s raw output to a prediction (often a probability vector) and it typically yields a vector per sample.  

In contrast, the **loss function** aggregates information over **all** $M$ samples and their associated labels. It is used only during training, produces a single scalar value (the loss), and therefore serves as the objective that the optimisation algorithm seeks to minimise.

---

### Maximum Likelihood Estimation Reminder  

Consider a training set consisting of observations $\mathbf{X}= \{\mathbf{x}_{1},\dots ,\mathbf{x}_{M}\}$ and corresponding labels $\mathbf{Y}= \{\mathbf{y}_{1},\dots ,\mathbf{y}_{M}\}$. Suppose we model the conditional probability density $p(\mathbf{y}\mid \mathbf{x})$.  

If the pairs $(\mathbf{x}_{m},\mathbf{y}_{m})$ are **independent and identically distributed (i.i.d.)**, the joint probability of the entire dataset factorises as  

\[
p(\mathbf{Y}\mid \mathbf{X})=\prod_{m=1}^{M} p(\mathbf{y}_{m}\mid \mathbf{x}_{m}).
\]

The likelihood function is the joint probability viewed as a function of the model parameters $\mathbf{w}$:  

\[
\underset{\mathbf{w}}{\text{maximize}}\;\Bigl\{\prod_{m=1}^{M} p(\mathbf{y}_{m}\mid \mathbf{x}_{m},\mathbf{w})\Bigr\}.
\]

Because a monotonic transformation does not change the maximiser, we usually work with the **negative log‑likelihood (NLL)**, turning the maximisation problem into a minimisation problem.

---

### Regression  

Assume a univariate Gaussian observation model  

\[
p(y\mid \mathbf{x},\mathbf{w},\beta)=\mathcal{N}\bigl(\hat{y}(\mathbf{x},\mathbf{w}),\,\tfrac{1}{\beta}\bigr),
\]

where $\hat{y}(\mathbf{x},\mathbf{w})$ denotes the network’s deterministic prediction (the mean $\mu$) and $\sigma^{2}=1/\beta$ is the variance. Expanding the Gaussian density yields  

\[
p(y\mid \mathbf{x},\mathbf{w},\beta)=\frac{\sqrt{\beta}}{\sqrt{2\pi}}\;
\exp\!\Bigl[-\frac{\beta}{2}\bigl(y-\hat{y}(\mathbf{x},\mathbf{w})\bigr)^{2}\Bigr].
\]

---

### Log‑likelihood function for regression  

The negative log‑likelihood for the whole training set is  

\[
\begin{aligned}
-\ln L(\mathbf{w}) 
&= \sum_{m=1}^{M}\!-\ln\!\Bigl(\frac{\sqrt{\beta}}{\sqrt{2\pi}}\;
e^{-\frac{\beta}{2}(y_{m}-\hat{y}(\mathbf{x}_{m},\mathbf{w}))^{2}}\Bigr)\\[4pt]
&= \sum_{m=1}^{M}\!\Bigl[-\ln\!\Bigl(\frac{\sqrt{\beta}}{\sqrt{2\pi}}\Bigr)
      +\frac{\beta}{2}(y_{m}-\hat{y}(\mathbf{x}_{m},\mathbf{w}))^{2}\Bigr] \\[4pt]
&= \frac{M}{2}\ln(2\pi)-\frac{M}{2}\ln\beta
   +\frac{\beta}{2}\sum_{m=1}^{M}\bigl(y_{m}-\hat{y}(\mathbf{x}_{m},\mathbf{w})\bigr)^{2}.
\end{aligned}
\]

Only the last term depends on the parameters $\mathbf{w}$; the first two terms are constant with respect to $\mathbf{w}$ and can be omitted during optimisation.

---

### $L^{2}$‑loss  

Discarding constants, minimising the NLL is equivalent to minimising the **$L^{2}$‑loss** (also called mean‑squared error):  

\[
\frac{\beta}{2}\sum_{m=1}^{M}\bigl(y_{m}-\hat{y}(\mathbf{x}_{m},\mathbf{w})\bigr)^{2}.
\]

When the output is a vector $\mathbf{y}_{m}$ and the prediction $\hat{\mathbf{y}}(\mathbf{x}_{m},\mathbf{w})$, the loss generalises to  

\[
\frac{\beta}{2}\sum_{m=1}^{M}\bigl\|\mathbf{y}_{m}-\hat{\mathbf{y}}(\mathbf{x}_{m},\mathbf{w})\bigr\|_{2}^{2}.
\]

---

### Classification using an $L$‑norm  

Although $L^{2}$‑ and $L^{1}$‑losses are traditionally associated with regression, they can also be applied to classification. In that context they correspond to minimising an **expected misclassification probability**. However, because they penalise errors only linearly, they lead to **slow convergence** and do not heavily punish strongly mis‑classified samples. They may still be useful when the training data contain **extreme label noise**.

---

### Classification  

When a network outputs a probability $p$ for a binary label, the appropriate likelihood model is the **Bernoulli distribution**  

\[
\mathfrak{B}(y\mid p)=
\begin{cases}
p^{y}(1-p)^{1-y}, & y\in\{0,1\},\\[2pt]
0, & \text{otherwise}.
\end{cases}
\]

For $K\!+\!1$ mutually exclusive classes we use the **multinoulli (categorical) distribution**  

\[
\mathfrak{C}(\mathbf{y}\mid \mathbf{p})=
\begin{cases}
\displaystyle\prod_{k=0}^{K} p_{k}^{\,y_{k}}, & y_{k}\in\{0,1\},\\[6pt]
0, & \text{otherwise},
\end{cases}
\]

where $\mathbf{y}$ is a one‑hot encoded label vector and $\mathbf{p}$ contains the predicted class probabilities.

---

### Example for $\mathfrak{C}$  

Consider a biased coin encoded as  

\[
\text{head} \; \rightarrow\; \begin{bmatrix}1 \\ 0\end{bmatrix},
\qquad
\text{tail} \; \rightarrow\; \begin{bmatrix}0 \\ 1\end{bmatrix}.
\]

If the coin is unfair with $\mathbf{p}= \begin{bmatrix}0.3 \\ 0.7\end{bmatrix}$ and we observe a tail, i.e. $\mathbf{y}= \begin{bmatrix}0 \\ 1\end{bmatrix}$, the multinoulli likelihood evaluates to  

\[
\mathfrak{C}(\mathbf{y}\mid \mathbf{p}) = p_{0}^{0}\,p_{1}^{1}=1\cdot 0.7 = 0.7,
\]

so the probability of obtaining a tail is $70\%$.

---

### Maximum Likelihood Estimation for classification  

Predicted scores $\hat{\mathbf{y}}(\mathbf{x},\mathbf{w})$ are transformed into probabilities with the softmax function. Assuming categorical labels, the likelihood for a single sample is  

\[
p(\mathbf{y}\mid \hat{\mathbf{y}}(\mathbf{x},\mathbf{w})) = \mathcal{C}\bigl(\mathbf{y},\hat{\mathbf{y}}(\mathbf{x},\mathbf{w})\bigr).
\]

The negative log‑likelihood over the whole dataset becomes  

\[
\begin{aligned}
L(\mathbf{w}) 
&= -\sum_{m=1}^{M}\ln p(\mathbf{y}_{m}\mid \hat{\mathbf{y}}(\mathbf{x}_{m},\mathbf{w}))\\
&= -\sum_{m=1}^{M}\sum_{k=0}^{K} y_{k,m}\,\ln \hat{y}_{k,m}\\
&\equiv \text{Cross‑entropy}(\mathbf{Y},\hat{\mathbf{Y}}).
\end{aligned}
\]

Thus the cross‑entropy loss is precisely the NLL for a categorical model.

---

### Relation to the Kullback–Leibler divergence  

The KL‑divergence between two distributions $p$ and $q$ is  

\[
\text{KL}(p,q)=\int p(x)\ln\frac{p(x)}{q(x)}\,dx
               = \underbrace{\int p(x)\ln p(x)\,dx}_{-\,\text{Entropy }H(p)}
                 -\underbrace{\int p(x)\ln q(x)\,dx}_{\text{Cross‑entropy }H(p,q)}.
\]

Since the cross‑entropy term $-\sum_{k} y_{k}\ln\hat{y}_{k}$ appears in the NLL, minimising the cross‑entropy loss is equivalent to minimising the KL‑divergence between the empirical label distribution and the model’s predicted distribution.

---

### Can we also use cross‑entropy for regression?  

Yes. If the model’s outputs are constrained to the interval $[0,1]$ (e.g., via a sigmoid activation), the target vector need not be one‑hot encoded. The same cross‑entropy formulation then measures the divergence between a target probability vector and the network’s prediction, providing a valid loss for regression‑style problems under a probabilistic interpretation.

---

### Summary (loss functions)  

* $L_{2}$‑loss (mean‑squared error) is the Maximum‑Likelihood estimator for a Gaussian regression model.  
* Cross‑entropy loss is the Maximum‑Likelihood estimator for a categorical (or Bernoulli) classification model and is equivalent to minimising the KL‑divergence.  
* In the absence of stronger priors, these two losses are the natural first choices.  
* Both losses are intrinsically multivariate, handling vector‑valued predictions and targets.

---

### Back to the Perceptron – again!  

The classic perceptron criterion can be written as  

\[
\underset{\mathbf{w}}{\text{minimise}}\;
L(\mathbf{w}) = -\sum_{\mathbf{x}_{m}\in\mathcal{M}} y_{m}\,(\mathbf{w}^{\!\top}\mathbf{x}_{m}),
\]

where $y_{m}\in\{-1,1\}$ are binary class labels. Note that the sign function does **not** appear in the objective; if it were inserted, the loss would count only the number of mis‑classifications, yielding a gradient that is zero almost everywhere—hence the need for a surrogate loss such as the hinge loss introduced next.

---

### Hinge loss  

For a binary classifier with labels $y_{m}\in\{-1,1\}$, the **hinge loss** is defined as  

\[
L(\mathbf{w}) = \sum_{m=1}^{M}\max\!\bigl(0,\,1 - y_{m}\,\hat{y}(\mathbf{x}_{m},\mathbf{w})\bigr),
\]

where $\hat{y}(\mathbf{x}_{m},\mathbf{w})$ denotes the raw (pre‑sign) network output. The loss penalises points that lie on the wrong side of the margin or within the margin (i.e., $y_{m}\hat{y}<1$). Because the hinge loss is a convex upper bound on the 0‑1 misclassification loss, it yields a tractable optimisation problem while preserving the essential property that correct classification corresponds to a non‑positive loss.

---

### Subgradients  

For a convex, differentiable function $f$, the first‑order condition states  

\[
f(\mathbf{x}) \ge f(\mathbf{x}_{0}) + \nabla f(\mathbf{x}_{0})^{\!\top}(\mathbf{x}-\mathbf{x}_{0})
\qquad\forall\,\mathbf{x}\in\mathcal{X}.
\]

If $f$ is **convex but not differentiable**, we replace the gradient by a **subgradient**. A vector $\mathbf{g}$ is a subgradient of $f$ at $\mathbf{x}_{0}$ if  

\[
f(\mathbf{x}) \ge f(\mathbf{x}_{0}) + \mathbf{g}^{\!\top}(\mathbf{x}-\mathbf{x}_{0})
\qquad\forall\,\mathbf{x}\in\mathcal{X}.
\]

The set of all subgradients at $\mathbf{x}_{0}$ is the **subdifferential**  

\[
\partial f(\mathbf{x}_{0}) \coloneqq \{\mathbf{g}\mid \mathbf{g}\text{ satisfies the inequality above}\}.
\]

If $f$ happens to be differentiable at $\mathbf{x}_{0}$, the subdifferential collapses to the singleton $\{\nabla f(\mathbf{x}_{0})\}$.

---

#### Subgradients for the ReLU  

Consider the scalar ReLU function $f(x)=\max(0,x)$. Applying the subgradient definition yields  

\[
\partial f(x_{0}) =
\begin{cases}
\{1\}, & x_{0}>0,\\[2pt]
\{0\}, & x_{0}<0,\\[2pt]
[0,1], & x_{0}=0.
\end{cases}
\]

During optimisation we may pick any value in $[0,1]$ when $x_{0}=0$; in practice deep‑learning libraries typically use $0$ or $1$ for computational convenience. The existence of subgradients justifies the use of (sub)gradient‑based optimisation methods for piecewise‑linear, non‑smooth activation functions such as ReLU and for hinge loss.

---

### Isn’t an SVM far more desirable?  

Support‑Vector Machines (SVMs) seek a **maximum‑margin** separating hyperplane. In a two‑dimensional binary setting the figure shows red and blue points, a solid line that correctly separates most points, and two parallel dashed lines that define the margin. Some points lie inside the margin, corresponding to *soft‑margin* violations.

The optimisation problem for a soft‑margin SVM is  

\[
\begin{aligned}
\min_{\mathbf{w},\boldsymbol{\xi}} \quad &
\frac{1}{2}\|\mathbf{w}\|_{2}^{2} + \gamma \sum_{m}\xi_{m} \\[4pt]
\text{s.t.}\quad &
y_{m}\bigl(\mathbf{w}^{\!\top}\mathbf{x}_{m}\bigr) \ge 1 - \xi_{m},\\
&\xi_{m} \ge 0,\qquad \forall m,
\end{aligned}
\]

where $\xi_{m}$ are slack variables allowing misclassifications and $\gamma>0$ controls the trade‑off between margin maximisation and penalty for violations.

---

#### Lagrangian dual and connection to hinge loss  

Introducing Lagrange multipliers $\lambda_{m}\ge 0$ (for the margin constraints) and $\nu_{m}\ge 0$ (for $\xi_{m}\ge 0$) yields the Lagrangian  

\[
\begin{aligned}
L(\mathbf{w},\boldsymbol{\xi},\boldsymbol{\lambda},\boldsymbol{\nu}) 
=&\;\frac{1}{2}\|\mathbf{w}\|_{2}^{2}
 + \gamma\sum_{m}\xi_{m}
 +\sum_{m}\lambda_{m}\bigl(1-y_{m}\,\mathbf{w}^{\!\top}\mathbf{x}_{m}-\xi_{m}\bigr)
 -\sum_{m}\nu_{m}\xi_{m}.
\end{aligned}
\]

Eliminating $\boldsymbol{\xi}$ and rearranging (up to an overall constant) gives the familiar **hinge‑loss formulation**  

\[
\frac{1}{2}\|\mathbf{w}\|_{2}^{2}
 + \gamma\sum_{m}\max\!\bigl(0, 1- y_{m}\,\mathbf{w}^{\!\top}\mathbf{x}_{m}\bigr).
\]

Thus an SVM can be viewed as a neural‑network model whose loss consists of an $L_{2}$ regulariser plus a hinge loss term.

---

### Open points  

* **Robust hinge variants.** Outliers are penalised linearly by the standard hinge loss. A squared hinge loss penalises large violations more severely:  

  \[
  L(\mathbf{w})

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Loss and Optimization – Part 1

## Classification and Regression Losses

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome everybody to deep learning! So, today we want to continue talking about the different losses and optimization. We want to go ahead and talk a bit about more details of these interesting problems. Let’s talk first about the loss functions first. Loss functions are generally used for different tasks and for different tasks you have different loss functions.

The two most important tasks that we are facing are regression and classification. So in classification, you want to estimate a discrete variable for every input. This means that you want to essentially decide in this two-class problem here on the left whether it’s blue or red dots. So, you need to model a decision boundary.

In regression, the idea is that you want to model a function that explains your data. So, you have some input function let’s say x₂ and you want to predict x₁ from it. To do so, you compute a function that will produce the appropriate value of x₁ for any given x₂. Here in this example, you can see this is a line fit.

We talked about activation functions, last activation as softmax, and cross-entropy loss. Somehow, we combined them and obviously there’s a difference between the last activation function in our network and the loss function. The last activation function is applied to the individual samples x each of the batch. It will also be present at training and testing time. So, the last activation function will become part of the network and will remain there to produce the output / the prediction. It generally produces a vector.

Now, the loss function combines all M samples and labels. In their combination, they produce a loss that describes how good the fit is. So, it’s only present during training time and the loss is generally a scalar value that describes how good the fit is. So, you only need it during training time.

Interestingly, many of those loss functions can be put in a probabilistic framework. This leads us to maximum likelihood estimation. In maximum likelihood estimation – just as a reminder – we consider everything to be probabilistic. So, we have a set of observations X that consists of individual observations. Then, we have associated labels. They also stem from some distribution and the observations are denoted as Y . Of course, we need a conditional probability density function that describes us somehow how y and x are related. In particular, we can compute the probability for y given some observation x . This will be very useful for example if we want to decide on a specific class. Now, we have to somehow model this data set. They are drawn from some distribution and the joint probability for the given data set can then be computed as a product over the individual conditional probabilities. Of course, if they’re independent and identically distributed, you can simply write this up as a large product over the entire training data set. So, you end up with this product over all M samples, where it’s just a product of the conditionals. This is useful because we can determine the best parameters by maximizing the joint probability over the entire training data set. We have to do it by evaluating this large product.

Now, this large product has a couple of problems. In particular, if we have high and low values, they may cancel out very quickly. So, it may be interesting to transform the entire problem into the logarithmic domain. Because the logarithm is a monotonous transformation, it doesn’t change the position of the maximum. Hence, we can use the log function and a negative sign to flip the maximization into a minimization. Instead of looking at the likelihood function, we can look at the negative log-likelihood function. Then, our large product is suddenly a sum over all the observations times the negative logarithm of the conditional probabilities.

Now, we can look at a univariate gaussian model. So, now we are one dimensional again and we can model this with a normal distribution where we would then choose the output of our network as the expected value and 1/β as the standard deviation. If we do so, we can find the following formulation: Square root of beta over square root of 2 pi times the exponential function of minus beta times the label minus the prediction to the power of 2 divided by 2.

Okay so let’s go ahead and put this into our log-likelihood function. Remember this is really something, you should know in the oral exam. Everybody needs to know the normal distribution and everybody needs to be able to convert this kind of universe Gaussian distribution into a loss function. If we do so, you will see that we can use the logarithm. It comes in very handy because it allows us to split the product here. Then, we also see that the logarithm cancels out with the exponential function. We simply get this beta over 2 times y subscript m minus y hat subscript m to the power of 2. We can simplify the first term further by applying the logarithm and pulling out the square root 2 pi. Then, we see that the sum over the first two terms is not depending on m, so we can simply multiply by M in order to get rid of the sum and move the sum only to the last term.

Now, you can see that only the last part here actually depends on w. Everything else doesn’t even contain w. So, if we seek to optimize towards w, we can simply neglect the first two parts. Then, we end up only with the part here on the right-hand side. You see that if we now assume β to be 1, we end up exactly with 1/2 and the sum over the square root of the differences. This is nothing else than the L2 norm. If you would write it in vector notation, you end up with this here. Of course, this is equivalent to a multi-dimensional Gaussian distribution with uniform variance.

Okay, so well there’s not just L2-losses. There’s also L1 losses. So, we can also replace those, and we will look at some properties of different L norms in a couple of videos as well. It’s generally a very nice approach and it corresponds to minimizing the expected misclassification probability. It may cause slow convergence, because they don’t penalize heavy misclassified probabilities, but they may be advantageous in extreme label noise.

Now, let’s assume now let’s assume that we want to classify. Then, our network would provide us with some probabilistic output p . Let’s say, we classify only into two classes. Then, we can model this as a Bernoulli distribution where we have classes zero and one. Of course, the probability of the other class is simply one minus p . This then gives us the probability distribution p ʸ times (1 – p)¹⁻ʸ.  Typically, we don’t have only two classes. This means we need to generalize to the multinulli or categorical distribution. Then y is typically modeled again one-hot encoded vector. We can then write down the categorical distribution as the product over all the classes of the probability for each class to the power of the ground truth label which is either zero or one.

Let’s look at an example of a categorical distribution. The example that we want to take here is a Bernoulli trial a coin flip. We encode head as (1 0)ᵀ and tail as (0 1)ᵀ. Then, we have an unfair coin and this unfair coin prefers tails with a probability of 0.7. Its likelihood for heads is 0.3. Then, we observe the true label y as tails. Now, we can use the above equation and plug those observations in. This means we get 0.3 to the power of 0 and 0.7 to the power of 1. Something to the power of 0 always equals to 1. Then 0.7 to the power of 1 is of course 0.7. This gives us 0.7 and this then means that the probability to observe tails for our unfair coin is 70%.

We can always use the softmax function within the network to convert everything into probabilities. Now, we can look at how this behaves with our categorically distributed system. Here, we simply replace our conditional with the categorical distribution. This then gives us a negative log-likelihood function. Again what we’re doing here is of high relevance for the oral exam. So everybody should be able to explain how to come from a probabilistic assumption to the respective loss function using the categorical distribution. So here, we again apply the negative log-likelihood. We plug in the definition of the categorical distribution which is simply the product over all our y subscript k hat to the power of the ground truth label. This can be further simplified because the product can be converted into a sum by moving in the logarithm. If we do so, you can see that the power of the ground truth label can actually be pulled in front of the logarithm. We see that we exactly end up with cross-entropy. Now, if you use the trick with the one-hot encoding again, you can see that we exactly end up with the cross-entropy loss where we have the sum over the entire set of observations times the logarithm of the output at exactly the position where our ground truth label was 1. Hence, we neglect all the other terms in the sum of the classes.

Interestingly, this can also be put in relation to the Kullback Leibler (KL) Divergence. KL divergence is a very common construct that you find in many machine learning papers. Here, you can see the definition. We essentially have an integral over the entire domain of x. It’s integrating the probability of p(x) times the logarithm of p(x) divided by q(x). q(x) is the reference distribution that you want to compare to. Now, you can see that you can split the two into two parts using the property of the logarithm. So, we get the minus part on the right-hand-side which is the cross-entropy. The left-hand side is simply the entropy. So, we can see that this training process is essentially identical to minimizing the cross-entropy. So, in order to minimize the KL divergence, we can minimize the cross-entropy.  You should keep that in mind this kind of relationship appears very often in machine learning papers. So you will find them easier to understand if you have these things in the back of your mind.

Now, can we use cross-entropy for regression? Well, yes we can do that of course. But you have to make sure that your predictions are going to be in the domain of [0, 1] for all of your classes. You can for example do this with a sigmoid activation function. Then you have to be careful because in regression typically you’re no longer one-hot encoded. So, this is something that you have to deal with appropriately. As seen before, this loss is equivalent to minimizing the KL divergence.

Let’s summarize what we’ve seen so far. So L2 loss is typically used for regression. Cross-entropy loss is typically used for classification typically in combination with one-hot encoding. Of course, you can derive them from ML estimators from strict probabilistic assumptions. So what we’re doing here is completely in line with probability theory. In the absence of more domain knowledge, these are our first choices. If you have additional domain knowledge then, of course, it’s a good idea to use it to build a better estimator. The cross-entropy loss is intrinsically multivariate. So, we are not just stuck with two-class problems. We can go to multi-dimensional regression and classification problems as well.

Next time in deep learning, we want to go into some more details about loss functions and in particular, we want to highlight the hinge loss. It is a very important loss function because it allows you to embed constraints. We will see that there are also some relations to classical machine learning and pattern recognition, in particular, the support vector machine. So I hope you enjoyed this video and I am looking forward to seeing you in the next one”!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a clap or a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Secaucus, NJ, USA: Springer-Verlag New York, Inc., 2006. [2] Anna Choromanska, Mikael Henaff, Michael Mathieu, et al. “The Loss Surfaces of Multilayer Networks.” In: AISTATS. 2015. [3] Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, et al. “Identifying and attacking the saddle point problem in high-dimensional non-convex optimization”. In: Advances in neural information processing systems. 2014, pp. 2933–2941. [4] Yichuan Tang. “Deep learning using linear support vector machines”. In: arXiv preprint arXiv:1306.0239 (2013). [5] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. “On the Convergence of Adam and Beyond”. In: International Conference on Learning Representations. 2018. [6] Katarzyna Janocha and Wojciech Marian Czarnecki. “On Loss Functions for Deep Neural Networks in Classification”. In: arXiv preprint arXiv:1702.05659 (2017). [7] Jeffrey Dean, Greg Corrado, Rajat Monga, et al. “Large scale distributed deep networks”. In: Advances in neural information processing systems. 2012, pp. 1223–1231. [8] Maren Mahsereci and Philipp Hennig. “Probabilistic line searches for stochastic optimization”. In: Advances In Neural Information Processing Systems. 2015, pp. 181–189. [9] Jason Weston, Chris Watkins, et al. “Support vector machines for multi-class pattern recognition.” In: ESANN. Vol. 99. 1999, pp. 219–224. [10] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016).

---

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Loss and Optimization – Part 2

## Do SVMs beat Deep Learning?

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome back to deep learning! So, let’s continue with our lecture. We want to talk about loss and optimization. Today, we want to talk a bit about loss functions and optimization. I want to look into a couple of more optimization problems and one of those optimization problems we’ve actually already seen in the perceptron case.

You remember that we were minimizing a sum over all of the misclassified samples. We were choosing this because we could somehow get rid of the sign function and only look into the samples that were relevant for misclassification. Also, note that here we don’t have a 0/1 category but a -1/1 because this allows us to multiply with the class label. This will then always result in a negative number in misclassified samples. Then we add this negative sign in the very beginning such that we always end up with a positive value. The smaller this positive value, the smaller our loss will be. So, we seek to minimize this function. We don’t have the sign function in this criterion because we found an elegant way to formulate this loss function without it. Now, if it were in we would run into problems because this would only count the number of misclassifications and we would not differentiate whether it’s far away from decision boundary or close to the decision boundary. We would simply end up with a count. If we look at the gradient, it would essentially vanish everywhere. So it’s not an easy optimization problem. We don’t know in which direction to go so we can’t find a good optimization. What did we do about this last time? Well, we somehow need to relax this and there are also ways how to fix this.

One way to go ahead is to include the so-called hinge loss. Now with the hinge loss, we can relax this 0/1 function into something that behaves linearly on a large domain. The idea is that we essentially use a line that hits the x-axis at 1 and the y-axis also at 1. If we do it this way, we can simply rewrite this using the max function. So, the hinge loss is then a sum over all the samples that essentially receive 0 if our value is larger than 1. So, we have to rewrite the right-hand part to reformulate this a little. We take 1 – y subscript m times y hat. Here, you can see that we will have the same constraint. If we have opposite sides of the boundary, this term will be negative and by the sign, it will of course be flipped such that we end up having large values for a high number of misclassifications. We got rid of the problem of having to find the set of misclassifications M. Now, we can take the full set of samples by using this max function because everything that will fulfill this constraint will automatically be clamped to 0. So, it will not influence this loss function. Thus, that’s a very interesting way of formulating the same problem. We get implicitly the situation that we only consider the misclassified samples in this loss function. It can be shown that the hinge loss is a convex approximation of the misclassification loss that we considered earlier. One big thing about this kind of optimization problem is of course the gradient. This loss function here has a kink. Thus, the derivative is not continuous in the point x = 1. So there it’s unclear what the derivative is and now you could say: “Okay, I can’t compute the derivative of this function. So, I’m doomed!”

Luckily subgradients save the day. So let’s introduce this concept and in order to do so, we have a look at convex differentiable functions. On those, we can say that at any point f( x ), we can essentially find a lower bound of f( x ) that is indicated by some f( x₀ ) plus the gradient at f( x₀ ) multiplied with the difference from x to x₀ . So, let’s look at a graph to show this concept. If you look at this function here, you can see that I can take any point x₀ and compute the gradient, or in this case, it’s simply the tangent that is constructed. By doing so, you will see that at any point of the tangent will be a lower bound to the entire function. It doesn’t matter where I take this point if I follow the tangent in the tangential direction, I’m always constructing a lower bound. Now, this kind of definition is much more suited for us.

So, let’s expand now on the gradient and go into the direction of subgradients. In subgradients, we define something which keeps this property but is not necessarily a gradient. So a vector g is a subgradient of a convex function f( x₀ ) if we have the same property. So, if we follow the subgradient direction, multiply it with the difference between x and x₀ , then we always have a lower bound. The nice thing with this is that we essentially can relax the requirement of being able to compute a gradient. There could be multiple of those g ‘s that fulfill this property. So, g is not required to be unique. The set of all of these subgradients is then called the subdifferential. So, the subdifferential is then a set of subgradients that all fulfill the property above. If f( x ) is differentiable at x₀ , we can simply say that the set containing all sub differentials is simply the set containing the gradient.

Now let’s look at a case where this is not true. In this example here, we have the rectified linear unit (ReLU) which also has exactly the same problem. Again it’s a convex function which means here at the point where we have the kink, we can find quite a few subgradients. Actually, you see the green line and the red line. Both of them are feasible subgradients and they fulfill this property that they are lower bounds to that respective function. So this means that we can now define a subdifferential and our subdifferential is essentially 1 where we have x₀ greater than 0. We have 0, where it’s smaller than 0. We have exactly g – and g can now be any number between 0 and 1 at the position x₀ =0. This is nice because we can now follow essentially this subgradient direction. It’s just that the gradient is defined differently in different parts of the curve. In particular, at the kink position, we have this situation where we would have multiple possible solutions. But for our optimization, it’s sufficient to just know one of the subgradients. We don’t have to compute the entire set. So, we can now simply extend our gradient descent algorithm to generalize to subgradients. There are proofs that for convex problems, you will still find the global minimum using subgradient theory. So, we can now say: “Well, the functions that we are looking at, they are locally convex.” This then allows us to find the local minima even with ReLUs and with the hinge loss.

So, let’s summarize this a bit: Subgradients are a generalization of gradients for convex non-smooth functions. The gradient descent algorithm is replaced by the respective subgradient algorithm for these functions. Still, this allows us to continue essentially how we did before for piecewise continuous functions. You just choose a particular subgradient and you probably don’t even notice the difference. The nice thing is that we’re not just doing this as an “engineering solution”, but there is also a solid mathematical theory that this actually works. Hence, we can use this for our ReLU and our hinge loss. This is mathematically sound and we can go ahead and not worry too much.

What if people say “Oh, Support Vector Machines (SVMs) are much better than what you’re doing, because they always achieve the global minimum.” So, isn’t it much better to use the SVM? So, let’s have a small look at what an SVM actually does. SVMs compute the optimally separating hyperplane. It’s also computing some plane that separates two classes with the idea is that it wants to maximize the margin between the two sets. So, you try to find the plane or here in this simple example, the line that produces the maximum margin. The hyperplane or the decision boundary is the black line and the dashed lines indicate the margin here. So, the SVM tries to find the margin that is maximally large while separating those classes. What is done typically is that you find this minimization problem where w is the normal vector of our hyperplane. We then minimize the magnitude of the normal vector. Note that this normal vector is not scaled which means that if you increase the magnitude of w , your normal vector gets longer. If you want to compute signed distances from that you typically divide by the magnitude of the normal vector. So, this means that if you increase the length of this normal vector your distances get smaller. If you want to maximize the distances, you minimize the length of the normal vector. Obviously, you could just collapse it to zero and then you would have essentially infinite distances. Just minimizing w would lead to the trivial solution w = 0 . To prevent this, you put this in a constraint optimization where you require for all observations m for all your samples that they are projected onto the right side of the decision boundary. This is introduced by this constraint minimization here. So you want to have the signed distance multiplied with the true label minus one to be smaller than 0.

Now, we can expand this also to cases where the two classes are not linearly separable in the so-called soft margin SVM. The trick here that we introduce a slack variable ξ that allows a misclassification. ξ is added towards the distance to the decision boundary. This means that I can take individual points and move them back to the decision boundary if they were incorrectly classified. To limit excessive use of ξ, I postulate that the ξ are all smaller or equal to 0. Furthermore, I postulate that the sum over all the ξ needs to be minimized as I want to have as few misclassifications as possible.

This then leads us to the complete formulation of the soft margin SVM and if I want to do this in a joint optimization, you convert this to the Lagrangian dual function. In order to do so, you introduce Lagrange multipliers λ and ν. So, you can see that the constraints that we had in the previous slide, they now receive the multipliers λ and ν. Of course, there’s an additional new m that is introduced for the individual constraints. Now, you see that this forms a rather complex optimization problem. Still, we have a single Lagrangian function that can be sought to be minimized for all w , ξ, λ, and ν. So, there’s a lot of parameters introduced here. We can rearrange it a little bit and pull many of the parameters into one sum. If we drop the support vector interpretation, we can regard this sum as a constant. Because we know that all of the lambdas are larger or equal to 0, it means that everything that will be misclassified or is closer to the decision boundary than 1 will create a positive loss. If you replace the lambda term with the maximum function, you will get the same result. The optimization will always produce something that will be zero or greater in the case of misclassification or if you are inside the area of the margin. In this trick, we approximate this by introducing the max function to suppress everything that is below 0, i.e., on the correct side and outside of the margin. Now, you see that we can very elegantly express this as a hinge loss. So, you can show that the support vector machine and the hinge loss formulation with those constraints are equivalent up to an overall multiplicative constant as shown reference [1]. If people say: “Oh, you can’t do deep learning. Take an SVM instead!”. Well, if you choose the right loss function, you can also incorporate a support vector machine into your deep learning framework. That’s actually a quite nice observation.

Okay some open points: Outliers are punished linearly. There’s a variant of the hinge loss which penalizes the outliers more strongly. You can do that for example by introducing squares. So also a very common choice see reference [4]. So, we can also apply this hinge loss to multi-class problems and what we are introducing here is simply an additional sum where we then do one versus many. So, we are not just classifying towards one class but we are classifying one versus the rest. This introduces the new classifiers shown here. In the very end, this leads to a multi-class hinge loss.

So, let’s summarize this a little bit: We have seen that we can incorporate SVMs into a neural network. We can do this with the hinge loss which is a  loss function that you can put in all kinds of constraints in. You can even incorporate forced-choice experiments as a loss function. This has then been called user-loss. So it’s a very flexible function that allows you to formulate also all kinds of constrained optimization problems in the framework of deep learning. You can put all kinds of constrained optimization also into deep learning frameworks. Also very cool that you learned about subgradients and why we can deal with non-smooth objective functions. So this is also a very useful part and if you run into a mathematician and they tell you “Oh, there’s this kink and you can’t compute the gradient!”. So from now on, you say: “Hey, subgradients save the day!”. This way, we just need to find one possible gradient and then it still works. This is really nice. Check our references if somebody ever approaches you that ReLUs are not okay in your gradient descent procedure. There are proofs for that.

Next time in deep learning, we want to go ahead and look a bit into the optimization. So far, all the optimization programs that we considered, they just had this η and this was somehow doing the same for all variables. Now, we’ve seen that different layers parameters can be very different and maybe, then they should not be just treated all equally. Actually, this will lead to big trouble. But if you look into more advanced optimization programs, they have some cool solutions how to treat the individual weights differently automatically. So stay tuned! I hope you like this lecture and would love to welcome you again in the next video!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a clap or a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Secaucus, NJ, USA: Springer-Verlag New York, Inc., 2006. [2] Anna Choromanska, Mikael Henaff, Michael Mathieu, et al. “The Loss Surfaces of Multilayer Networks.” In: AISTATS. 2015. [3] Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, et al. “Identifying and attacking the saddle point problem in high-dimensional non-convex optimization”. In: Advances in neural information processing systems. 2014, pp. 2933–2941. [4] Yichuan Tang. “Deep learning using linear support vector machines”. In: arXiv preprint arXiv:1306.0239 (2013). [5] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. “On the Convergence of Adam and Beyond”. In: International Conference on Learning Representations. 2018. [6] Katarzyna Janocha and Wojciech Marian Czarnecki. “On Loss Functions for Deep Neural Networks in Classification”. In: arXiv preprint arXiv:1702.05659 (2017). [7] Jeffrey Dean, Greg Corrado, Rajat Monga, et al. “Large scale distributed deep networks”. In: Advances in neural information processing systems. 2012, pp. 1223–1231. [8] Maren Mahsereci and Philipp Hennig. “Probabilistic line searches for stochastic optimization”. In: Advances In Neural Information Processing Systems. 2015, pp. 181–189. [9] Jason Weston, Chris Watkins, et al. “Support vector machines for multi-class pattern recognition.” In: ESANN. Vol. 99. 1999, pp. 219–224. [10] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016).

---

- Lecture Notes in Deep Learning

# Lecture Notes in Deep Learning: Loss and Optimization – Part 3

## Optimization with ADAM and beyond…

These are the lecture notes for FAU’s YouTube Lecture “ Deep Learning “. This is a full transcript of the lecture video & matching slides. We hope, you enjoy this as much as the videos. Of course, this transcript was created with deep learning techniques largely automatically and only minor manual modifications were performed. If you spot mistakes, please let us know!

Welcome everybody to our next lecture on deep learning! Today, we want to talk about optimization. Let’s have a look at the gradient descent methods in a little bit more detail. So, we’ve seen that the gradient is essentially optimizing the empirical risk. Here in this figure, you see that we do one step each towards this local minimum. We have this predefined learning rate η. So, the gradient is of course computed with respect to every sample and this is then guaranteed to converge to a local minimum.

Now, of course, this means that for every iteration, we have to use all samples and this is called batch gradient descent. So you have to look in every iteration, for every update, at all samples. It may be really many samples, in particular if you look at big data a computer vision problems. This is of course a preferred option for convex optimization problems because we have a guarantee here that we find the global minimum. Every update is guaranteed to decrease the error. Of course, for non-convex problems, we have a problem anyway. Also, we may have memory limitations. This is why people like to prefer other approaches like the stochastic gradient descent SGD. Here, they use just one sample and then immediately update. So, this is no longer necessarily decreasing the empirical risk in every iteration and it may also be very inefficient because of the latency in transfers to the graphical processing unit. However, if you use just one sample you can do many things in parallel so it’s highly parallelizable. A compromise between the two is that you can use Mini-Batch stochastic gradient descent. Here you use B and B may be a number much smaller than the entire training data set of random samples that you essentially choose them randomly from the entire training data set. Then, you evaluate the gradient on the subset B. This is then called a mini-batch. Now, this mini-batch can be evaluated really quickly and you may also use parallelization approaches and because you can do several mini-batch steps in parallel. Then, you just do the weighted sum and update. So, small batches are useful because they offer a kind of regularization effect. This then typically results in smaller η. So, you have if you use a mini-batch in gradient descent, typically smaller values of η are sufficient and it also regains efficiency. Typically, this is the standard case in deep learning.

So, a lot of people work with this, meaning that the gradient descent is effective. But the question is: “How can this even work?” Our optimization problem is non-convex. There’s an exponential number of local minima and there’s an interesting paper from 2015 and where they show that the metrics that we are typically working with are high dimensional functions. There are many local minima in this environment. The interesting thing is that those local minima are close to the global minimum and actually many of those are equivalent. What is probably more of a problem are saddle points. Also, the local minima might even be better than the global minimum because the global minimum is attained on your training set, but in the end, you want to apply your network to a test data set that may be different. Actually, a global minimum on your training data set may be related to an overfit. Maybe, this is even worse for the generalization of the trained network.

One more possible answer to this is a paper from 2016. The authors are suggesting over-provisioning as there are many different ways of how a network can approximate the desired relationship. You essentially just need to find one. You don’t need to find all of them. A single one is sufficient. Liang at al. verified this experimentally by experiments with random labels. Here, the idea is that you essentially randomize the labels you don’t use the original. You just randomly assign any classes and if you then show that your experiment still solves the problem, then you are creating an overfit.

Let’s have a look at the choice of η. What we’ve already seen that if you have a small learning rate, we may stop even before we reach convergence. If you have a too large learning rate, we might be ending jumping back and forth and not even finding the local minimum. Only with an appropriate learning rate, you will be able to find the minimum. Actually, when you’re far away from the minimum, you want to be able to make big steps, and the closer you get to the minimum smaller steps. If you want to do so in practice, you work with the decay of the learning rate. So, you adapt your η gradually. You start with let’s say 0.01 and then you divide by ten every x epochs. This helps that you don’t miss the local minimum that you’re actually looking for. It’s a typical practical engineering parameter.

Now, you may ask can’t we get rid of this magic η? So what is typically done? Quite a few people suggest doing a line search in similar cases. So, line search of course needs you to estimate the optimal η at every step. So, you need multiple evaluations in order to find the correct η in the direction that the gradient points. It is extremely noisy anyway. So, people have presented methods but they are not the state of the art right now in deep learning. Then, people have suggested second-order methods. If you look into second-order methods, you need to compute the Hessian matrix and this is typically very expensive to calculate. So far, we have not seen that too often. There are L-BFGS methods but they typically don’t perform very well if you are operating outside of the batch setting. So, if you work with mini-batches, they are not that great. There’s a report on that by Google that you can find in reference [7].

What else can we do? Well, of course, we could accelerate the directions of persistent gradients. So, the idea here would be that you somehow keep track of the average that is indicated here with new v . This is essentially a weighted sum over the last couple of gradient steps. You take the current gradient direction indicated in red and average it with the previous steps. This then gives you an updated direction and this is typically called a momentum.

So, we introduce this momentum term. There you add with a new weight some momentum that is indicated with v superscript k-1. This momentum term is essentially computed in an iterative fashion where you iteratively update over the past gradient directions. So you can essentially say by iteratively computing this weighted mean, you keep a history of the previous gradient directions and you gradually update them with the new gradient direction. Then you pick the momentum term in order to perform the update instead of just the gradient direction. So, typical choices for μ are 0.9, 0.95, or 0.99. You can also adapt them from small to large if you want them to pay more emphasis on the previous gradient directions. This overcomes poor Hessians and variance in the stochastic gradient descent. It will dampen oscillations and it accelerates typically the optimization procedures. Still, we need the learning rate decay. So, this doesn’t solve the automatic adjustment of η.

We can also pick a different way of momentum: the Nesterov Accelerated Gradient or simply Nesterov Momentum. This performs a look-ahead. So here, we also have this momentum term. But instead of evaluating the gradient at the position we’re currently at, we add the momentum term before computing the gradient. So, we are essentially trying to approximate the next set of parameters. We use the look-ahead it here and then we perform the gradient update. So you can rewrite this to use the conventional gradient. The idea here is to put the Nesterov Acceleration directly into the gradient update. This term will then be used in the next gradient update. So, this is an equivalent formulation.

Let’s visualize this a bit. Here, you can see momentum and the Nesterov Momentum. Of course, they both use a kind of momentum term. But they use a different direction for calculating the gradient update. That’s the main difference.

Here, you see an example of these momentum terms in comparison. In this situation, you have a strong disagreement in the variance in both directions. So, we have a very high variance in left and right directions and a rather small variance in the top-bottom direction. We’re trying to find the global minimum. This then leads typically to alternating gradient directions very strongly even if you introduce the momentum term. You still get this strong oscillating behavior. If you use the Nesterov Accelerated Gradient, you can see that we compute this look-ahead and this allows us to follow the blue line. So, we are directly moving towards the desired minimum and we are no longer alternating. This is an advantage of Nesterov.

What if our features have different needs? So, suppose some features are activated very infrequently while others are updated very often. Then, we would need individual learning rates for every parameter in the network. We need large learning rates for infrequent parameters and small learning rates for frequent parameters.

In order to accommodate the changes appropriately, this can be done with the so-called AdaGrad method. It is using first the gradient to compute some g superscript k and then it’s computing the product of the gradient with itself to keeps track of its element-wise variance in a variable r . Now, we use our g and our r elementwise as in combination with η to weigh the update of the gradient. So, now we construct updated weights and the variance of the weights in every parameter is incorporated by multiplying with the square root of the respective element, i.e. an approximation of its standard deviation. So, here we note this down as a square root of an entire vector. All the other things here are scalar which means that this also results in a vector again. This vector is then multiplied point-wise with the actual gradient to perform a local scaling. It’s a nice and efficient method and allows individual learning rates for all of the different dimensions for all of the different weights. One problem could be that the learning rate decreases too aggressively.

This is a problem and leads to an improved version and the improved version here is RMSProp. RMSProp is now using this again, but they introduce this ρ and now ρ is being used to essentially introduce a delay such that you don’t have very high increases. Here you can set this ρ in order to dampen the update of the variance of the learning rate. So, Hinton suggests 0.9 and η = 0.001. This turn leads to the aggressive decrease being fixed, but we still have to set the learning rate. If you don’t set the learning rate appropriately, you run into a problem.

Now, Adadelta tries to improve on this further. They use essentially our RMSProp and get rid of η. So, we already have seen r . It is the variance that is computed in a dampened way. Then, in addition, they introduce this Δₓ. It is a weighted combination of some term h and the r that we have seen previously multiplied to the gradient direction. So, this is an additional dampening factor that replaces the η in the original formulation. The factor h is computed again as a sliding average over the Δₓ as an element-wise product. So, this way you don’t have to set a learning rate anymore. Still, you have to choose the parameter ρ. For ρ we suggest going to 0.95.

One of the most popular algorithms that are being used is Adam and Adam is essentially also using this gradient direction g . Then, you have essentially a momentum term v . In addition, you have this r term that is again trying to steer the learning rate for each dimension individually. Furthermore, Adam introduces an additional bias correction where v is scaled by 1 minus μ. r is also scaled by 1 over 1 minus ρ. This then leads to the final update term that involves the learning rate η, our momentum term, and the respective scaling. This algorithm is called Ada ptive M oment estimation. For Adam suggested values are μ = 0.9, ρ = 0.999 and η=0,001. It’s a very robust method and very commonly used. We can combine it with the Nesterov accelerated gradient and you get “Nadam”. But still, you can improve on this.

Adam was empirically observed to fail to converge to optimal/good solutions. In reference [5], you can even see that Adam and similar methods do not guarantee to converge for convex problems. There’s an error in the original convergence proof and therefore, we suggest using AMSGrad that fixes Adam to ensure non increasing step size. So, you can fix it by adding a maximum over the momentum update term. So if you do this, you result in AMSGrad. This is shown to be even more robust. The effect has been shown in large experiments. One lesson that we learn here is that you should keep your eyes open. Even things that go through scientific peer review may have problems that are later identified. Another thing that we learned here is that these gradient descent procedures – as long as you approximately follow the correct gradient direction – you still get quite decent results. Of course, such gradient methods are really hard to debug. So, be sure that you debug your gradients. This really happens as you can see in this example. Even large software frameworks may suffer from such errors. For a long time, people didn’t notice them. They just notice strange behavior, but the problem persists.

Okay, so let’s summarize this a bit. The stochastic gradient descent plus Nesterov momentum plus learning rate decay is a typical choice in many experiments. It converges most reliably and is used in many state-of-the-art papers. Still, it has the problem that this learning rate decay has to be adjusted. Adam has individual learning rates. The learning rates are very well-behaved, but of course, the loss curves are much harder to interpret because you don’t have this typical behavior as you would see with fixed learning rates. What we didn’t discuss here and we only hinted at that is distributed gradient descent. Of course, you can also do this in a parallelized manner and then compute different update steps in different nodes of a distributed network or on different graphic boards. Then you average over them. This has also been shown to be very robust and very fast.

Some practical recommendations: Start by using mini-batch stochastic gradient descent with momentum. Stick to the default momentum. Give Adam a try when you have a feeling for your data. When you see that you need individual learning rates then Adam can help you with getting better or more stable convergence. You can also switch to AMSGrad which is an improvement over Adam. Of course, start adjusting the learning rate first and then keep your eyes open regarding unusual behavior.

Okay, this brings us to a short outlook on the next couple of videos and what we are coming up with. Of course the actual deep learning part, we haven’t discussed this at all so far. So one problem that we still need to talk about is how can we deal with spatial correlation and features. We hear so much about convolution and neural networks. Next time we see why this is a good idea how is it implemented. Of course, one thing that we should think about is how to use variances and how to incorporate them into network architectures.

Some comprehensive questions: “What are our standard loss functions for classification and regression?” So of course, L2 for regression and cross-entropy loss for classification. You should be able to derive those. This is really something that you should know because the statistics and their relations to learning losses are really important. The statistical assumptions, probabilistic theory, and how to modify those to get to our loss functions are highly relevant for the exam. Very important are also subdifferentials. “What’s a subdifferential?” “How can we optimize in situations where our activation functions have a kink?” “What’s the hinge loss?” “How can we incorporate constraints?” and in particular “What do we tell people that claim that all our techniques are not good because an SVM is superior?” Well, you can always show that it’s up to a multiplicative constant the same as using hinge loss. “What is Nesterov Momentum?” These are very typical things you should be able to explain if somebody in the near future is going to ask you questions about the things that you have been learning here. Of course, we have plenty of references again (see below). I hope you enjoyed this lecture as well and I am looking forward to seeing you in the next one!

If you liked this post, you can find more essays here , more educational material on Machine Learning here , or have a look at our Deep Learning Lecture . I would also appreciate a clap or a follow on YouTube , Twitter , Facebook , or LinkedIn in case you want to be informed about more essays, videos, and research in the future. This article is released under the Creative Commons 4.0 Attribution License and can be reprinted and modified if referenced.

## References

[1] Christopher M. Bishop. Pattern Recognition and Machine Learning (Information Science and Statistics). Secaucus, NJ, USA: Springer-Verlag New York, Inc., 2006. [2] Anna Choromanska, Mikael Henaff, Michael Mathieu, et al. “The Loss Surfaces of Multilayer Networks.” In: AISTATS. 2015. [3] Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, et al. “Identifying and attacking the saddle point problem in high-dimensional non-convex optimization”. In: Advances in neural information processing systems. 2014, pp. 2933–2941. [4] Yichuan Tang. “Deep learning using linear support vector machines”. In: arXiv preprint arXiv:1306.0239 (2013). [5] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. “On the Convergence of Adam and Beyond”. In: International Conference on Learning Representations. 2018. [6] Katarzyna Janocha and Wojciech Marian Czarnecki. “On Loss Functions for Deep Neural Networks in Classification”. In: arXiv preprint arXiv:1702.05659 (2017). [7] Jeffrey Dean, Greg Corrado, Rajat Monga, et al. “Large scale distributed deep networks”. In: Advances in neural information processing systems. 2012, pp. 1223–1231. [8] Maren Mahsereci and Philipp Hennig. “Probabilistic line searches for stochastic optimization”. In: Advances In Neural Information Processing Systems. 2015, pp. 181–189. [9] Jason Weston, Chris Watkins, et al. “Support vector machines for multi-class pattern recognition.” In: ESANN. Vol. 99. 1999, pp. 219–224. [10] Chiyuan Zhang, Samy Bengio, Moritz Hardt, et al. “Understanding deep learning requires rethinking generalization”. In: arXiv preprint arXiv:1611.03530 (2016).

## Lecture Notes Sources

These integrated lecture notes were transcribed from voice recordings of the lecture (FAU LME). Follow the links for the original blog posts:

- [Loss And Optimization Part 1](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-loss-and-optimization-part-1/)
- [Loss And Optimization Part 2](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-loss-and-optimization-part-2/)
- [Loss And Optimization Part 3](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-loss-and-optimization-part-3/)
