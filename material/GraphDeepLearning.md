---
course: deep-learning
title: "Graph Deep Learning"
source: deep-learning/slides/GraphDeepLearning/GraphDeepLearningLecture.pptx
---

# Graph Deep Learning

## Graph Deep Learning Part 1

### Introduction to Graph Deep Learning

The research presented in this lecture series originates from the Pattern Recognition Lab at the **Friedrich‑Alexander‑Universität Erlangen‑Nürnberg (FAU)**, specifically within the Faculty of Engineering. The work is a collaborative effort of the following researchers:

- A. Maier  
- V. Christlein  
- K. Breininger  
- S. Vesal  
- F. Meister  
- C. Liu  
- S. Gündel  
- S. Jaganathan  
- N. Maul  
- M. Vornehm  
- L. Reeb  
- F. Thamm  
- C. Bergler  
- F. Denzinger  
- B. Geissler  
- Z. Yang  
- A. Popp  
- M. Nau  

This collaboration provides a solid foundation for the subsequent material, which covers a broad spectrum of topics in graph deep learning, ranging from fundamental concepts to state‑of‑the‑art architectures and applications. The lecture begins with an introductory question — what exactly is meant by "graph deep learning"? Before diving into the technical content the lecturer is careful to disambiguate the term, because in everyday usage the word *graph* can mean very different things, and only one of those meanings is the subject of this lecture.

## Graph Deep Learning

### Graph Deep Learning – Illustrative Figure

The figure combines several visual elements to illustrate what graph deep learning is *not* about as much as what it *is* about. The lecturer deliberately presents a montage of "graphs" in the colloquial sense to clarify the terminology before introducing the technical definition. The unifying message is that none of the following meanings of the word "graph" describe the object of study in this lecture:

- **Mathematical function plots.** A first connotation of "graph" is the plot of a function such as $y=f(x)$, drawn as a smooth curve in the plane. While this is what mathematicians often call the "graph of a function," it is *not* the kind of graph addressed by graph deep learning.

- **Snow Water Equivalent (SWE) Graph.** A central sub‑figure depicts a graph that plots snow accumulation over time, containing multiple series such as a "plot created" line, a "30‑year mean" baseline, and actual SWE observations. This is again a plot in the everyday meaning — a chart of measured values — and is not the technical object of interest.

- **Steffi Graf (1999).** Tongue‑in‑cheek, the lecturer also points out that "Graf" is a famous tennis player. The photograph of the tennis champion serves only as a humorous reminder that the word "graph/Graf" carries many unrelated meanings in everyday speech, none of which are the subject of this lecture.

- **Abstract Geometrical Shapes / node‑edge diagrams.** The surrounding abstract shapes — diagrams of nodes connected by edges — finally illustrate the kind of object that *is* the focus of the lecture: graphs as combinatorial structures in which entities are represented as vertices and pairwise relationships as edges. These shapes also exemplify that graph neural networks (GNNs) operate on non‑Euclidean domains, where the underlying structure is defined by arbitrary connectivity rather than a regular lattice, and so they can handle data defined on meshes, point clouds, or any irregular topology.

The figure is divided into four labeled regions, denoted as **[a]**, **[b]**, **[c]**, and **[d]**. Although the slide does not provide explicit captions for each label, the layout suggests that they correspond to the components described above:

1. **[a]** – The mathematical function plot.
2. **[b]** – The SWE time‑series graph.
3. **[c]** – The portrait of Steffi Graf.
4. **[d]** – Abstract node‑and‑edge diagrams of the kind that this lecture treats.

The lecturer then crystallises the working definition. A computer scientist thinks of a graph as a *set of nodes* connected through *edges*; this is the kind of graph that the lecture will discuss. A mathematician, by contrast, regards a graph as a *manifold* — but a *discrete* one. Both viewpoints will become important: the computer‑science view leads naturally to spatial, message‑passing formulations, while the manifold view leads to spectral formulations through the graph Laplacian.

## Graph Deep Learning

### Graph Definition

From a computer‑science standpoint, a **graph** is defined as a collection of nodes (also called vertices) that are interconnected by edges (also called links). This elementary combinatorial structure serves as a universal framework for representing relationships and interactions among discrete entities such as users in a social network, atoms in a molecule, or routers in a communication network. The formalism is concise: a graph $G$ can be written as the ordered pair $G = (V, E)$, where $V$ is the set of nodes and $E \subseteq V \times V$ is the set of edges. In many applications edges may be directed, weighted, or carry additional attributes, but the core idea remains a set of pairwise connections.

In contrast, a mathematician often regards a graph as a **discrete manifold** — a space that, while composed of isolated points, locally resembles Euclidean space. This viewpoint highlights that a graph can inherit geometric and topological notions (e.g., curvature, Laplace operators, homology) despite its combinatorial nature. By treating a graph as a manifold‑like object, one can bring powerful tools from differential geometry and algebraic topology to bear on graph analysis, enabling concepts such as smooth signal processing on graphs, diffusion processes, and spectral embeddings.

*Figure*: The slide illustrates two visual representations of a graph. On the left, a simple network of connected nodes exemplifies the computer‑scientist's perspective, emphasizing the discrete connectivity pattern. On the right, a stylized rabbit constructed from nodes conveys the manifold interpretation, suggesting that the same underlying structure can be viewed as a shape embedded in a continuous space.

[5]  
[e]

## How would you define a convolution on Euclidean space?

### How would you define a convolution on Euclidean space?

In both computer science and mathematics, a convolution is an operation that quantifies the overlap between two functions as one function is shifted across the other. Before turning to graphs, the lecturer first revisits the definition of convolution on ordinary Euclidean space, where the question is "too easy" for both computer scientists and mathematicians: each community already has its preferred form of the operator.

Formally, let $f$ and $g$ be functions defined on a Euclidean domain. The convolution can be expressed in two equivalent forms, depending on whether the domain is discrete or continuous.

* **Discrete convolution.**  
  When the domain $\mathcal{D}$ consists of integer lattice points (e.g., pixel indices in an image), the convolution at position $n$ is defined as  

  $$
  (f * g)(n) = \sum_{k \in \mathcal{D}} f(k)\,g(n - k)\,.
  $$

  Here the sum iterates over all positions $k$ in the domain, multiplying the value of $f$ at $k$ with the value of $g$ at the shifted location $n-k$. The result captures how well the pattern described by $g$ matches a translated version of $f$. This is exactly the form used to set up the kernels in convolutional deep models — the same discrete sum reappears whenever a CNN slides a small filter over a feature map.

* **Continuous convolution.**  
  When the functions are defined on the continuous space $\mathbb{R}^n$, the convolution becomes an integral computed over the entire space:  

  $$
  (f * g)(n) = \int_{\mathbb{R}^n} f(x - \tau)\,g(\tau)\,d\tau\,.
  $$

  In this formulation, the variable $\tau$ runs over the entire Euclidean space, and the integrand evaluates the product of $f$ shifted by $\tau$ and $g$ at $\tau$. The integral aggregates these products, again measuring the degree of overlap between the two functions.

Both definitions embody the same geometric intuition: as one function slides over the other, the convolution records the accumulated similarity at each shift. This is the "move‑multiply‑sum" picture: one function is shifted across the other, the two are multiplied at every position, and the products are summed (or integrated) to produce the convolution value. This principle underlies many signal‑processing tasks, image filtering, and, most importantly for deep learning, the operation performed by convolutional neural network (CNN) layers.

> **Additional insight.**  
> A classic example frequently mentioned in the lecture is the convolution of two Gaussian functions. If  
> \[
> f(x)=\frac{1}{\sqrt{2\pi\sigma_f^2}}\exp\!\left(-\frac{x^{2}}{2\sigma_f^{2}}\right),\qquad
> g(x)=\frac{1}{\sqrt{2\pi\sigma_g^2}}\exp\!\left(-\frac{x^{2}}{2\sigma_g^{2}}\right),
> \]  
> then their convolution is again a Gaussian:  
> \[
> (f*g)(x)=\frac{1}{\sqrt{2\pi(\sigma_f^{2}+\sigma_g^{2})}}\exp\!\left(-\frac{x^{2}}{2(\sigma_f^{2}+\sigma_g^{2})}\right).
> \]  
> This closure property is the reason why Gaussian kernels are often used as smoothing operators in image processing and why they provide an analytically tractable illustration of the "move‑multiply‑sum" intuition. The lecture explicitly invokes this example: convolving two Gaussians yields a Gaussian again, so the operation is "easy" to visualise as one bell‑curve sliding across another.

> Historically, the term *convolution* (from the Latin *convolvere*, "to roll together") was formalised in the 19th century by mathematicians such as Cauchy and Poisson. It became a cornerstone of linear time‑invariant system theory after Norbert Wiener's work, linking convolution in the time (or spatial) domain to pointwise multiplication in the Fourier domain.

> In the context of CNNs, the discrete definition above is instantiated with a learnable kernel $K$ (often a small matrix of parameters). Because the same kernel is applied at every spatial location, the resulting operation is *shift‑equivariant*: translating the input leads to a translated output, a property inherited directly from the underlying convolution definition.

#### Visual illustration

The accompanying figure (as described on the slide) visualizes several aspects of convolution:

* A mathematical equation of the convolution operation is shown alongside an iconic image of Albert Einstein, emphasizing the broad relevance of the concept.
* A plot depicts a function $f$ being convolved with another function $g$; the notation $f * g$ indicates the resulting combined shape, with a small animation showing the two curves moving over each other and producing the new shape — exactly the Gaussian‑on‑Gaussian example mentioned above.
* Binary matrices portray a kernel $K$ (often a small, learnable filter) and the operation $I * K$, where an input image $I$ is convolved with the kernel. These matrices illustrate how each entry of the kernel interacts with corresponding pixels of the input to produce a new feature map.

Together, these visual elements reinforce the algebraic definitions and demonstrate how convolution aggregates local information in a structured, shift‑invariant manner.

## How would you define a convolution on graphs?

### How would you define a convolution on graphs?

The same question becomes far more delicate once the underlying domain is no longer Euclidean. As the lecture notes put it, the computer scientist "thinks really hard but … what the heck!" — the lattice‑based discrete sum has no obvious analogue when nodes have varying numbers of irregular neighbours. The mathematician, however, knows that **Laplace transforms** can be used to describe convolutions, and so naturally turns to the **Laplacian operator**, which on a smooth domain is given by the divergence of the gradient,
\[
\Delta f = -\operatorname{div}(\nabla f),
\]
where $\nabla$ denotes the gradient and $\operatorname{div}$ the divergence. This observation provides the bridge: in mathematics the convolution can be characterised by how it interacts with the Laplacian, and so once we know how to discretise the Laplacian on a graph, we know how to define a graph convolution.

The bridge in fact goes via the **manifold idea**. As recalled in the lecture, a graph can be viewed as a (discrete) manifold; we know how to convolve functions on a manifold; we know how to discretize that convolution; and therefore we know how to convolve graphs. This logical chain — manifold convolution, discretisation, graph convolution — is the conceptual backbone of the entire spectral approach.

On a graph $G = (V, E)$ with adjacency matrix $\mathbf{A}$ and degree matrix $\mathbf{D}$, the (unnormalized) graph Laplacian is
\[
\mathbf{L} = \mathbf{D} - \mathbf{A}.
\]
Applying $\mathbf{L}$ to a feature vector $\mathbf{f} \in \mathbb{R}^{|V|}$ yields
\[
\Delta \mathbf{f} = \mathbf{L}\mathbf{f},
\]
which can be interpreted as a discrete analogue of the continuous Laplacian $\Delta f$. This operator captures the notion of "difference to neighbours" and thus serves as a natural definition of convolution on graphs: a graph convolution can be expressed as a (possibly weighted) function of the Laplacian acting on the signal $\mathbf{f}$.

In practice, many graph neural network (GNN) architectures replace the exact Laplacian with a *renormalized* version for numerical stability,
\[
\tilde{\mathbf{L}} = \mathbf{I} - \tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}},
\]
where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ adds self‑loops and $\tilde{\mathbf{D}}$ is the corresponding degree matrix. Convolutional layers then take the form
\[
\mathbf{H}^{(k+1)} = \sigma\!\left(\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(k)} \mathbf{W}^{(k)}\right),
\]
where $\sigma$ is a non‑linear activation and $\mathbf{W}^{(k)}$ are learnable weight matrices. This formulation directly follows from the Laplacian‑based definition and demonstrates how convolution can be performed efficiently on irregular graph structures.

> **Illustration**  
> The original slide juxtaposes a cartoon of a computer scientist furrowing his brow while trying to derive a graph convolution, against a mathematician who simply states the definition $\Delta f = - \operatorname{div}(\nabla f)$ "like a boss." The visual metaphor highlights the contrast between a laborious, algorithm‑centric approach and a concise, mathematically elegant formulation.

*Reference:* A. Maier, V. Christlein, K. Breininger, S. Vesal, *Graph Deep Learning*, July 3 2020.

#### Convolution on a smooth manifold and its discretisation

To make the manifold step concrete, consider a smooth manifold $\mathcal{M}$ with a scalar (or vector‑valued) function $f:\mathcal{M}\rightarrow\mathbb{R}$ and a filter (or kernel) $g:\mathcal{M}\rightarrow\mathbb{R}$. The continuous convolution of $f$ with $g$ at a point $x\in\mathcal{M}$ is defined as  

\[
(f * g)(x) \;=\; \int_{\mathcal{M}} f(y)\,g\bigl(\operatorname{dist}_{\mathcal{M}}(x,y)\bigr)\, \mathrm{d}y,
\]

where $\operatorname{dist}_{\mathcal{M}}(x,y)$ denotes the geodesic distance on the manifold. This expression captures the idea that the response at $x$ is obtained by aggregating the values of $f$ over the whole manifold, weighted by a kernel that depends only on the distance to $x$.

A graph $G=(V,E)$ can be regarded as a *discrete sampling* of an underlying manifold. Each vertex $v\in V$ corresponds to a sample point on $\mathcal{M}$, and the edges $E$ encode local neighborhood relationships. To transfer the continuous convolution to the graph, we replace the integral by a finite sum over the neighboring vertices:

\[
(f * g)(v) \;=\; \sum_{u\in\mathcal{N}(v)} f(u)\, g\bigl(d(v,u)\bigr),
\]

where $\mathcal{N}(v)$ denotes the set of vertices that are considered neighbors of $v$ (often the 1‑hop or multi‑hop neighborhood) and $d(v,u)$ is a discrete notion of distance, for example the length of the shortest path between $v$ and $u$ or the weight associated with the edge $(v,u)$. This **discretized convolution** preserves the essential property of the continuous case: the filter depends only on the relative position (distance) of the surrounding nodes, not on their absolute identities.

The discretization step yields a concrete recipe for a **graph convolution**:

1. **Define a kernel function** $g$ that maps a scalar distance to a weight (e.g., a Gaussian $g(r)=\exp\!\bigl(-r^{2}/(2\sigma^{2})\bigr)$ or a learned parametric function).  
2. **Compute distances** $d(v,u)$ for all pairs $(v,u)$ within the chosen neighborhood of each vertex.  
3. **Aggregate neighbor features** by weighting each neighbor's signal $f(u)$ with the kernel value $g\bigl(d(v,u)\bigr)$ and summing the contributions.

Mathematically, for a feature matrix $\mathbf{X}\in\mathbb{R}^{|V|\times C}$ (where each row $\mathbf{x}_v$ stores the $C$ channels of node $v$), a single graph‑convolutional layer can be written as  

\[
\mathbf{X}'_{v} \;=\; \sum_{u\in\mathcal{N}(v)} g\!\bigl(d(v,u)\bigr)\,\mathbf{x}_{u}\, \mathbf{W},
\]

with a learnable weight matrix $\mathbf{W}\in\mathbb{R}^{C\times C'}$ that mixes the input channels into output channels. This formulation mirrors the classic Euclidean convolution while respecting the irregular connectivity of the graph.

> **Figure (description).** The illustration depicts the transition from a smooth, color‑gradient function defined on a continuous manifold to its discrete approximation on a graph. The left side shows the continuous function varying smoothly over the manifold surface. The right side shows a set of sampled points (graph nodes) with discrete values, connected by edges that encode the locality of the original manifold. This visual emphasizes how a continuous convolution can be approximated by aggregating information over the graph's neighborhoods.

The above reasoning follows the approach presented in the literature on graph signal processing and geometric deep learning, notably the works cited in [5].

*Andreas Maier, Viktoria Christlein, Kathrin Breininger, and S. Vesal – "Graph Deep Learning", July 3 2020.*

## Graph Deep Learning

### Heat Diffusion with Newton's Law of Cooling

Having identified the Laplacian as the key operator, the lecture now motivates it physically: "let's diffuse some heat." The diffusion of heat on a continuous domain can be modeled by Newton's Law of Cooling, which in its differential form reads  

\[
f_t(x,t) = -\Delta f(x,t), \qquad f(x,0)=f_0(x).
\]

Here the function $f(x,t)$ denotes the temperature (or more generally, the amount of heat) at spatial point $x$ and time $t$. The development of the system over time is described by the Laplacian: the rate of change at every point equals (minus) the Laplacian of the field at that point. The initial condition $f_0(x)$ specifies the temperature distribution at the initial time $t=0$ — to evolve the system one needs to know how the heat is distributed in the initial state.

The operator $\Delta$ is the **Laplacian**, defined as the negative divergence of the gradient:

\[
\Delta f(x)= -\operatorname{div}\bigl(\nabla f(x)\bigr).
\]

Intuitively, the Laplacian measures how the value of $f$ at a point deviates from the average of $f$ over an infinitesimally small sphere centered at that point. Consequently, the term $-\Delta f$ drives the temperature at each location toward the local average, i.e., heat flows from hotter regions to colder regions.

On a curved surface (a **manifold**) the same diffusion process can be visualized as the smoothing of a scalar field defined on the manifold. The slide illustrates this phenomenon with a 3‑D rendering of a manifold surface. A color map ranging from blue (low temperature) to red (high temperature) displays the evolving heat distribution. A small sphere placed on the surface, together with an arrow labeled "$f$", indicates the direction of the temperature gradient at that location — visualising precisely the "infinitesimal small sphere" around $x$ over which the average is taken.

This diffusion equation underlies many graph‑based learning methods. When a graph is interpreted as a discrete analogue of a manifold, the graph Laplacian plays the role of $\Delta$, and diffusion processes on the graph can be used to propagate information, smooth signals, or define convolution‑like operations.

> *Figure description*: A three‑dimensional visualization of heat diffusion on a curved manifold. The surface is colored from blue (cold) to red (hot). A small sphere is shown on the surface, and an arrow labeled "$f$" points in the direction of the gradient of the heat distribution.

*References*: Newton's Law of Cooling formulation is discussed in [6]. The slide is taken from A. Maier, V. Christlein, K. Breininger, S. Vesal, "Graph Deep Learning", July 3 2020 [5].

## Graph Deep Learning

### Discrete Laplacian Operator on Graphs

Having defined the Laplacian on a smooth domain, the next step is to ask how it should be expressed in *discrete* form on a graph. The lecture's answer is a direct translation of the continuous interpretation: the Laplacian is again the difference between $f(x)$ and the average of $f$ on an infinitesimal sphere around $x$. On a graph, the smallest "infinitesimal" step we can take from a node is to its immediate neighbours, so the discrete Laplacian is built precisely from those neighbour differences.

In continuous calculus the Laplacian, denoted by $\Delta f(x)$, can be written as the negative divergence of the gradient,
\[
\Delta f(x) = -\operatorname{div}\bigl(\nabla f(x)\bigr).
\]
When the domain is a graph, we replace the continuous differential operators by discrete counterparts that respect the graph's connectivity.

For an undirected graph $G=(V,E)$ with adjacency weights $a_{ij}$ and node degrees $d_i=\sum_{j:(i,j)\in E} a_{ij}$, the **discrete Laplacian** applied to a signal $f\colon V\to\mathbb{R}$ is defined by
\[
(\Delta f)_i = \frac{1}{d_i}\sum_{j:(i,j)\in E} a_{ij}\,\bigl(f_i - f_j\bigr).
\]
In this expression the summation runs over all neighbors $j$ of node $i$. The term $f_i - f_j$ is the difference between the value of $f$ at node $i$ and the value at a neighboring node $j$, weighted by the edge strength $a_{ij}$. Dividing by the degree $d_i$ — the number of connections incoming into node $i$ — normalises the contribution of each neighbor, so the discrete Laplacian can be interpreted as **the difference between the value at a node and the average of its values on an infinitesimal "sphere" (i.e., the set of adjacent nodes) around that node**. This is the direct discrete counterpart of the continuous "deviation from the local average" reading of $\Delta f$.

> **Figure (illustration of the discrete Laplacian calculation on a graph).**  
> The picture shows a graph where a scalar function $f$ is defined on each node (and optionally on edges). For a selected node, the Laplacian computes the weighted differences $f_i - f_j$ to each neighbor $j$, sums them, and normalises by the node degree $d_i$. The result quantifies how much $f_i$ deviates from the average of its neighbours, effectively measuring a discrete notion of curvature on the graph.

*Reference: A. Maier, V. Christlein, K. Breininger, S. Vesal | Graph Deep Learning*

## Graph Deep Learning

### Graph Laplacian

A central object in graph deep learning is the **graph Laplacian**, which captures the combinatorial structure of a graph and is used in many spectral methods, graph convolutional networks, and regularization schemes. Beyond the per‑node difference formula written above, the lecture asks: "is there another way of expressing this?" and answers in the affirmative — namely as a **matrix** built from the degree and adjacency matrices.

One way to view the (unnormalized) Laplacian is through its action on a signal (or feature vector) $f$ defined on the nodes of the graph. For a node $i$, the Laplacian applied to $f$ can be written as  

\[
(Af)_i = \sum_{j:(i,j) \in E} a_{ij}\,(f_i - f_j),
\]

where $a_{ij}$ denotes the weight of edge $(i,j)$ (often $a_{ij}=1$ for an unweighted graph) and the sum runs over all neighbours $j$ of node $i$. This expression makes explicit that the Laplacian measures the **difference** between a node's feature value $f_i$ and the feature values of its neighbours, weighted by the adjacency entries. No degree‑normalization $d_i$ appears in this form, which is why the slide asks, "Is there another way of expressing this? (Below without the normalization $d_i$)".

In the **normalized** setting, the Laplacian is often divided by the degree of the source node, leading to the random‑walk Laplacian $L_{rw}=D^{-1}L$ or, more commonly for undirected graphs, the symmetric normalized Laplacian  

\[
L_{\text{sym}} = I - D^{-1/2} A D^{-1/2},
\]

where $I$ is the identity matrix. This symmetric form guarantees a positive‑semi‑definite matrix even for directed graphs after the symmetrization step described later, and it is the basis of the popular Graph Convolutional Network (GCN) formulation introduced by Kipf & Welling [1].

#### Worked example: degree, adjacency and Laplacian matrices

The lecture introduces these matrices with a concrete six‑node example. Suppose the graph has nodes labelled 1 to 6, and we count the incoming connections at each node. Node 1 has 2 incoming connections, Node 2 has 3, Node 3 has 2, Node 4 has 3, Node 5 has 3, and Node 6 has only 1.

##### Degree matrix $D$

The degree matrix encodes the summed edge weights incident to each node. In matrix form it is diagonal, with the $i$‑th diagonal entry equal to the degree $d_i = \sum_j a_{ij}$. For the six‑node example the degree matrix is

```text
D = [[2, 0, 0, 0, 0, 0],
     [0, 3, 0, 0, 0, 0],
     [0, 0, 2, 0, 0, 0],
     [0, 0, 0, 3, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]]
```

Here each diagonal entry counts how many edges are incident to the corresponding node (e.g., node 2 has degree 3). The degree matrix also plays a crucial role in the **symmetrization** of the Laplacian for directed graphs. By constructing $D^{-1/2}$, we can left‑ and right‑multiply the unnormalized Laplacian to obtain $L_{\text{sym}}$, which is guaranteed to be symmetric positive‑definite and thus admits an eigen‑decomposition.

##### Adjacency matrix $A$

The adjacency matrix records the connectivity of the graph: entry $a_{ij}=1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise (for an unweighted graph). The same six‑node graph has the adjacency matrix

```text
A = [[0, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 1],
     [1, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0]]
```

Rows and columns correspond to nodes $1$ through $6$. For instance, node 1 is connected to nodes 2 and 5, as indicated by the 1's in the first row.

##### Laplacian matrix $\Delta$

With the degree and adjacency matrices defined, the (unnormalized) graph Laplacian is obtained by the simple matrix subtraction  

\[
\Delta = D - A.
\]

This is a purely element‑wise difference of the two matrices, and it matches the node‑wise expression above: multiplying $\Delta$ by a feature vector $f$ yields exactly the sum of differences between each node's value and those of its neighbours, weighted by the adjacency entries.

In physics the Laplacian governs diffusion processes. If $f_i(t)$ denotes the temperature (or any scalar quantity) at node $i$ at time $t$, the **discrete heat equation**
\[
\frac{d}{dt}f(t) = -L_{\text{norm}}\,f(t)
\]
describes how heat spreads over the graph. The solution shows that each node's temperature relaxes towards the average of its neighbours, which matches the intuition behind the definition of $(\Delta f)_i$ as a deviation from a local average and confirms the heat‑diffusion analogy used in the previous section.

Beyond the raw difference form, the Laplacian admits a **spectral decomposition**  

\[
\Delta = U \Lambda U^{\top},
\]

where $U$ collects the orthonormal eigenvectors (the **graph Fourier modes**) and $\Lambda$ is a diagonal matrix of eigenvalues (the **spectral frequencies**). This decomposition underlies **spectral graph convolution**: a signal $x$ is transformed into the graph Fourier domain via $\hat{x}=U^{\top}x$, multiplied element‑wise by a filter $g(\Lambda)$, and transformed back with $U$. In practice, the filter is often modeled as a low‑order polynomial of the Laplacian, e.g.  

\[
g_{\theta}(\Delta) = \sum_{k=0}^{K} \theta_k \Delta^{k},
\]

which allows the convolution to be expressed without explicitly computing $U$. The special case $K=1$ with $\theta_0 = 2\theta$, $\theta_1 = -\theta$ yields the first‑order approximation used in the GCN layer  

\[
H = \sigma\!\left( \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} X W \right),
\]

where $\tilde{A}=A+I$ and $\tilde{D}$ is its degree matrix. This derivation shows how the **symmetrized normalized Laplacian** naturally appears when we eliminate the costly eigenbasis $U$ [1].

> **Figure** – The slide depicts the Laplacian matrix formulation within a graph deep learning context. It provides concrete examples of the adjacency matrix $A$ and the degree matrix $D$, which together define the Laplacian $\Delta = D - A$. The underlying graph structure is implicitly encoded by these matrices, illustrating how graph topology is transformed into algebraic objects used by neural models.

*Reference*: A. Maier, V. Christlein, K. Breininger, S. Vesal, "Graph Deep Learning", 3 July 2020.  
Additional references: Kipf & Welling [1]; Hamilton, Ying & Leskovec [2] (for spatial aggregators).

## Graph Deep Learning

### Laplacian Matrix and Symmetric Normalization

In a graph (or sub‑graph) that contains $N$ nodes, the **graph Laplacian** is a central algebraic object that captures the structure of node connections. We denote the Laplacian matrix by $\Delta \in \mathbb{R}^{N \times N}$, and observe that it is an $N\times N$ matrix describing a graph or subgraph consisting of $N$ nodes. Its definition relies on two auxiliary matrices, both also $N\times N$:

1. **Degree matrix** $D \in \mathbb{R}^{N \times N}$ – a diagonal matrix whose $i$‑th diagonal entry $D_{ii}$ equals the number of edges connected to (i.e., incident to) node $i$. In other words, $D_{ii} = \sum_{j} A_{ij}$, where $A$ is the adjacency matrix introduced below.

2. **Adjacency matrix** $A \in \mathbb{R}^{N \times N}$ – a matrix that encodes the connectivity of the graph. For an undirected graph, $A$ is symmetric and $A_{ij}=1$ if there is an edge between nodes $i$ and $j$, and $0$ otherwise. For a directed graph, $A_{ij}=1$ indicates a directed edge from node $i$ to node $j$.

When the graph is **directed**, the Laplacian $\Delta$ is generally **not symmetric positive definite**. Many algorithms (e.g., spectral methods) require a symmetric operator, so we normalise the Laplacian to obtain a symmetric version, denoted $\Delta_{\text{sym}}$. The normalisation proceeds as follows: we start with the original Laplacian matrix and exploit the fact that $D$ is a diagonal matrix, so its inverse square root $D^{-1/2}$ can be computed element‑wise (each diagonal entry is replaced by the reciprocal of its square root, $(D^{-1/2})_{ii}=1/\sqrt{d_i}$). We then multiply the Laplacian from the left and from the right by $D^{-1/2}$, and rearrange. The resulting symmetrised matrix can be written in three equivalent ways:

$$
\Delta_{\text{sym}} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}} .
$$

This expression can be reformulated by adding and subtracting the identity matrix $I$, which yields an equivalent representation that makes the relationship to the classic *unnormalised* Laplacian $L = D - A$ apparent:

$$
\Delta_{\text{sym}} = I - D^{-\frac{1}{2}} A D^{\frac{1}{2}} .
$$

A further simplification, obtained by moving the exponent on the right‑hand $D$ back to $-\frac{1}{2}$, gives the most common form used in graph convolutional networks and related spectral methods:

$$
\Delta_{\text{sym}} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} .
$$

These three equations are mathematically equivalent; they merely highlight different aspects of the normalisation process. The final form emphasizes that the symmetric normalised Laplacian is the identity matrix minus a doubly‑scaled adjacency matrix, guaranteeing that $\Delta_{\text{sym}}$ is symmetric and, under mild conditions, positive semidefinite. The crucial point — explicitly highlighted in the lecture — is that this construction works **even for directed graphs**: regardless of whether the original adjacency matrix is symmetric, the sandwiched form always yields a symmetric operator.

The symmetric normalised Laplacian is also the cornerstone of spectral graph theory. Because $\Delta_{\text{sym}}$ is symmetric positive semidefinite, it admits an eigendecomposition $\Delta_{\text{sym}} = U \Lambda U^{\top}$, where the columns of $U$ are the **graph Fourier modes** and the diagonal entries of $\Lambda$ are the **spectral frequencies**. This decomposition enables a graph Fourier transform $\hat{x}=U^{\top}x$ and its inverse $x = U\hat{x}$, providing the foundation for spectral convolutional filters. The lecture notes stress that once the matrix is symmetric, these eigenvectors can be used as a basis for defining convolutions in the frequency domain, and that the eigenvalues serve as filter coefficients.

Historically, the graph Laplacian dates back to Kirchhoff's circuit laws (1847), where it was used to analyse electrical networks. Its modern incarnation as a tool for spectral clustering and graph neural networks was popularised by works such as Kipf & Welling (2016) [1], which introduced the simplified graph convolution that directly employs the symmetric normalised Laplacian $I - D^{-1/2} A D^{-1/2}$ to avoid costly eigendecompositions.

In practice, the normalisation also improves numerical stability when node degrees vary widely, since each edge contribution is scaled by the geometric mean of the degrees of its incident nodes. This scaling ensures that high‑degree nodes do not dominate the aggregation, a point emphasized in the lecture when discussing the need for symmetric positive definite operators before applying spectral methods.

*Reference*: A. Maier, V. Christlein, K. Breininger, S. Vesal, "Graph Deep Learning," July 3 2020.

## Graph Deep Learning

### Let's do some magic!

In many graph‑signal processing pipelines we start from a **symmetric graph Laplacian** (or a symmetrized version of any graph shift operator) that we denote by $\Delta_{sym}$. The lecture introduces the eigendecomposition step with a flourish — "we can do some magic" — because it is precisely this decomposition that unlocks Fourier analysis on graphs. The "magic" works because, as soon as the matrix is symmetric positive definite, it can be written in eigenvector–eigenvalue form, and we can use the eigenvectors and eigenvalues to Fourier‑transform a graph and to look at its spectral properties.

Because the matrix is symmetric, it admits an **eigendecomposition** of the form  

\[
\Delta_{sym}=U\Lambda U^{\mathsf T}.
\]

Here  

* $U=[u_0,\dots,u_{N-1}]\in\mathbb{R}^{N\times N}$ is an orthogonal matrix whose columns $u_i$ are the eigenvectors of $\Delta_{sym}$. Orthogonality means $U^{\mathsf T}U=UU^{\mathsf T}=I_N$, where $I_N$ is the $N\times N$ identity. These eigenvectors are known as the **graph Fourier modes**.

* $\Lambda\in\mathbb{R}^{N\times N}$ is a diagonal matrix that collects the eigenvalues (also called **spectral frequencies**). Explicitly,  

  \[
  \Lambda=\operatorname{diag}\bigl([\lambda_0,\dots,\lambda_{N-1}]\bigr),
  \qquad
  \lambda_i\in\mathbb{R}.
  \]

The eigenvalues $\lambda_0,\dots,\lambda_{N-1}$ encode the frequency content of the graph: small eigenvalues correspond to smooth (low‑frequency) variation over the graph, whereas large eigenvalues correspond to high‑frequency oscillations. In the lecture's words, $U$ and $U^{\mathsf T}$ play the roles of the forward and inverse Fourier transforms on the graph, and the diagonal $\Lambda$ supplies the **spectral filter coefficients** that can be used to design graph filters in the frequency domain.

> **Graph Fourier Transform (GFT).**  
> For a graph signal $\mathbf{x}\in\mathbb{R}^N$ (i.e., a scalar value assigned to each node), the GFT is defined by projecting $\mathbf{x}$ onto the eigenvector basis:
> \[
> \hat{\mathbf{x}} = U^{\mathsf T}\mathbf{x}.
> \]
> The inverse transform reconstructs the signal from its spectral coefficients:
> \[
> \mathbf{x}=U\hat{\mathbf{x}}.
> \]
> Consequently, the pair $(U,U^{\mathsf T})$ plays the same role as the forward and inverse discrete Fourier transform matrices in classical signal processing, while $\Lambda$ supplies the **spectral filter coefficients** that can be used to design graph filters in the frequency domain.

Thus, by diagonalizing $\Delta_{sym}$ we obtain the tools needed to transfer familiar Fourier‑analytic concepts to the irregular domain of graphs.

For directed or weighted graphs the raw combinatorial Laplacian $L = D - A$ is generally **not** symmetric positive‑definite. To obtain a symmetric operator suitable for the eigendecomposition above, we first form the degree matrix $D$ (a diagonal matrix containing the sum of incident edge weights for each node) and the adjacency matrix $A$. The **normalized symmetric Laplacian** is then constructed as  

\[
L_{\text{sym}} = I_N - D^{-\frac12} A D^{-\frac12},
\]

which is exactly the matrix $\Delta_{sym}$ used in the presentation. This normalization guarantees $L_{\text{sym}}$ is symmetric and has eigenvalues in $[0,2]$, making the spectral interpretation well‑behaved.

A *spectral graph convolution* with a filter $g_\theta$ can be written in the eigenbasis as  

\[
g_\theta \star \mathbf{x} \;=\; U\, g_\theta(\Lambda)\, U^{\mathsf T}\mathbf{x},
\]

where $g_\theta(\Lambda)$ is a diagonal matrix applying a scalar function $g_\theta$ to each eigenvalue. In practice, one chooses $g_\theta$ to be a low‑order polynomial  

\[
g_\theta(\Lambda)=\sum_{k=0}^{K}\theta_k \Lambda^{k},
\]

so that the convolution becomes a linear combination of powers of the Laplacian. This formulation avoids the need to compute $U$ explicitly because $\Lambda^{k}$ can be replaced by $L_{\text{sym}}^{k}$, and matrix‑vector products with $L_{\text{sym}}$ are cheap (they involve only sparse neighbor aggregations).

The lecture highlights a particularly important special case: setting $K=1$ and choosing coefficients $\theta_0=2\theta$, $\theta_1=-\theta$ yields the first‑order approximation  

\[
g_\theta \star \mathbf{x}\;=\;\theta\bigl(I_N + D^{-\frac12} A D^{-\frac12}\bigr)\mathbf{x},
\]

which, after re‑parameterisation, is exactly the propagation rule introduced by Kipf & Welling (2016) for Graph Convolutional Networks (GCNs). This connection explains why many modern GNN architectures operate directly on the normalized adjacency matrix without ever forming $U$ or $\Lambda$.

Finally, the heat‑diffusion analogy mentioned in the lecture provides an intuitive picture: applying a spectral filter corresponds to letting heat (or information) diffuse over the graph for a certain amount of "time", with low‑frequency components spreading slowly (preserving smooth structure) and high‑frequency components dampening quickly (removing noise). Polynomial filters therefore implement controlled diffusion steps that can be learned end‑to‑end in a deep network.

#### Figure
*Figure: A blue swirl is shown on the left side of the slide. This likely represents a visual cue for the concept of patterns or transformations being discussed in the lecture.*

## Graph Deep Learning

### Spectral Graph Signal Processing and Convolution

In graph‑based learning we often associate a scalar value with each node of a graph. Such an assignment is called a **graph signal** and can be represented as a vector  

\[
x \in \mathbb{R}^{N},
\]

where $N$ is the number of nodes and the $i$‑th entry $x_i$ is the signal value at node $i$. More generally we can also assign a *set of coefficients* describing properties of each node — for any such assignment the procedure described below yields a corresponding spectral representation.

#### Graph Fourier Transform

The combinatorial graph Laplacian $L$ (or its normalized counterpart) is a symmetric positive‑semi‑definite matrix. Let  

\[
L = U \Lambda U^{\top}
\]

be its eigendecomposition, where $U = [u_1, \dots, u_N]$ collects the orthonormal eigenvectors and $\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_N)$ holds the corresponding eigenvalues. Because $U$ forms an orthonormal basis, it can be used to define a **graph Fourier transform** (GFT) exactly as the classical Fourier transform uses the complex exponentials.

The forward GFT of a signal $x$ is obtained by projecting $x$ onto the eigenvectors:

\[
\hat{x} = U^{\top} x .
\]

The inverse transform reconstructs the signal from its spectral coefficients:

\[
x = U \hat{x}.
\]

Thus the eigenvectors of the Laplacian play the role of Fourier modes, with the eigenvalues $\lambda_i$ interpreted as frequencies.

> **Heat‑diffusion intuition.** In the continuous setting the Laplace operator governs the diffusion of heat according to Newton's law of cooling. Discretizing this process on a graph leads to the same Laplacian matrix: the value at a node tends to move towards the average of its neighbours. Consequently, a low‑frequency eigenvector corresponds to a smooth temperature distribution, whereas high‑frequency modes exhibit rapid sign changes across edges. This physical analogy underlies the interpretation of spectral filtering as applying a smoothing (or sharpening) operation to graph signals.

#### Convolution in the Spectral Domain

Convolution on a graph can be defined analogously to the Euclidean case: a filter $g$ is applied to a signal $x$ by multiplying their spectral representations element‑wise and then transforming back to the vertex domain. Formally,

\[
g * x \;=\; U \bigl((U^{\top} g) \odot (U^{\top} x)\bigr),
\]

where $\odot$ denotes the element‑wise (Hadamard) product. This expression shows that filtering consists of three steps:

1. **Fourier transform** the filter and the signal (multiply by $U^{\top}$).  
2. **Pointwise multiplication** of the resulting spectral coefficients.  
3. **Inverse Fourier transform** to obtain the filtered signal (multiply by $U$).

These are exactly the same three steps that one would perform when filtering a traditional time‑domain signal via the discrete Fourier transform — the lecture stresses that this is a direct port of classical signal processing to the graph domain.

> **Computational note.** Computing the full eigenbasis $U$ costs $\mathcal{O}(N^{3})$ and the dense matrix multiplications $U^{\top}x$ and $U(\cdot)$ require $\mathcal{O}(N^{2})$ operations, which quickly become prohibitive for large graphs. Crucially, unlike the regular‑grid Fourier transform, $U$ here does **not** admit a fast Fourier transform trick — it is always a full matrix multiplication, which is heavy and motivates the move to **spectral‑filter parameterisations that avoid the explicit eigendecomposition**, as described next.

#### Polynomial Parameterisation of Spectral Filters

Directly learning an arbitrary spectral filter requires storing a distinct coefficient for each eigenvalue, which is infeasible for large graphs. A common remedy is to restrict the filter to be a **polynomial of the Laplacian eigenvalues** — a $k$‑th order polynomial of Laplacians whose coefficients $\theta_i$ are simply real numbers. The resulting polynomial is a polynomial with respect to the spectral coefficients and is linear in the parameters $\theta$. Letting $\Lambda$ denote the diagonal matrix of eigenvalues, a $k$‑th order polynomial filter can be written as

\[
\hat{G} \;=\; \sum_{i=0}^{k} \theta_i \Lambda^{i}
   \;=\; \theta_k \Lambda^{k} + \dots + \theta_1 \Lambda^{1} + \theta_0,
\]

where the coefficients $\theta_i \in \mathbb{R}$ are learnable parameters. Because $\Lambda^{i} = (U^{\top} L U)^{i}$ and $U$ is orthogonal, this polynomial can be transferred back to the vertex domain without explicitly computing the eigenvectors:

\[
\hat{G} = U \left( \sum_{i=0}^{k} \theta_i \Lambda^{i} \right) U^{\top}
         = \sum_{i=0}^{k} \theta_i L^{i}.
\]

Consequently, the filtering operation becomes a **linear combination of powers of the Laplacian**, each of which can be applied efficiently using sparse matrix multiplication. To apply this filter to a new signal we therefore: take the signal $x$, apply the Fourier transform $U^{\top}$, multiply by the polynomial $\hat{G}(\Lambda)$, and apply the inverse Fourier transform $U$. As the lecture remarks, this is "exactly the same as before" — the only choice we make is the polynomial, and we can then learn the filter coefficients $\theta$ end to end. This parameterisation yields three important benefits:

* **Locality** – a $k$‑th order polynomial involves information from at most $k$ hops away in the graph.  
* **Scalability** – the number of learnable parameters grows only with the polynomial order $k$, not with the number of nodes.  
* **Stability** – the filter is a smooth function of the spectrum, which mitigates sensitivity to eigenvalue perturbations.

The remaining problem is that, to multiply by the filter, we still need $U$ — and as already emphasised, $U$ is heavy to compute and there is no fast Fourier transform shortcut on a general graph. The lecture closes this section with a deliberately suspenseful question: "what if I told you that a clever choice of polynomials cancels out $U$ entirely?" That cancellation is exactly the topic of Part 2.

> **Normalized Laplacian and symmetric formulation.** For directed or irregular graphs the combinatorial Laplacian $L = D - A$ is not symmetric positive‑definite. A common remedy is the symmetric normalisation  

\[
\tilde{L}= I - D^{-1/2} A D^{-1/2},
\]

where $D$ is the degree matrix and $A$ the adjacency matrix. The eigenvectors of $\tilde{L}$ are orthogonal and the corresponding eigenvalues lie in $[0,2]$, which simplifies polynomial design and improves numerical stability. In the transcript a concrete six‑node example illustrated how $D$ and $A$ are assembled and how $\tilde{L}=D^{-1/2}(D-A)D^{-1/2}=I-D^{-1/2} A D^{-1/2}$ is obtained.

> **Chebyshev approximation.** Defferrard et al. (2016) showed that restricting the polynomial to a truncated Chebyshev expansion $T_{i}(\tilde{L})$ yields filters that can be computed via a recursion requiring only two sparse matrix–vector multiplications per order. This leads to the *Chebyshev network*, which retains exact locality while avoiding the eigenbasis entirely.

> **First‑order (GCN) approximation.** Kipf & Welling (2017) observed that choosing $k=1$ and setting $\theta_{0}=2\theta$, $\theta_{1}=-\theta$ collapses the filter to  

\[
\hat{G}= \theta \bigl( I + D^{-1/2} A D^{-1/2} \bigr),
\]

eliminating any dependence on $U$. This compact expression forms the core of the widely used Graph Convolutional Network (GCN):  

\[
H^{(l+1)} = \sigma\!\bigl( \hat{A}\, H^{(l)} W^{(l)} \bigr), \qquad
\hat{A}= I + D^{-1/2} A D^{-1/2},
\]

where $\sigma$ is a non‑linearity, $W^{(l)}$ are learnable weight matrices, and $H^{(l)}$ the node representations at layer $l$.

These ideas form the foundation of many spectral graph convolutional networks, such as the Chebyshev network and the Graph Convolutional Network (GCN) introduced by Kipf and Welling.

*Reference: A. Maier, V. Christlein, K. Breininger, S. Vesal – "Graph Deep Learning", July 3 2020.*

## Graph Deep Learning

### Graph Deep Learning

Graph deep learning provides a mathematical framework for building rich, learned representations of complex data structures that are naturally modeled as graphs. By treating entities as nodes and their relationships as edges, we can apply neural‑network‑based techniques to extract latent features that encode both local connectivity and global topology. These learned representations have proven useful across a diverse set of domains, including:

* **Social network analysis** – where node embeddings can reveal community structure, influence patterns, and recommend new connections.  
* **Drug discovery** – where molecules are naturally represented as graphs of atoms (nodes) and chemical bonds (edges); embeddings enable prediction of molecular properties and identification of promising compounds.  
* **Recommendation systems** – where users and items form bipartite graphs; learned embeddings capture user preferences and item characteristics, improving recommendation accuracy.  

#### Core Idea: Learning Node Embeddings

Regardless of the application domain or the size and complexity of the underlying graph, the fundamental objective remains the same: **learn a mapping from each node to a low‑dimensional vector (an embedding) that preserves the structural and attribute information of the graph**. Once node embeddings are obtained, they can be combined with edge information to construct higher‑level representations of whole subgraphs or entire graphs, and these representations can be fed into downstream tasks such as classification, regression, or clustering.

The learning process typically follows three conceptual steps:

1. **Node‑to‑vector conversion** – a neural architecture (e.g., Graph Convolutional Network, GraphSAGE, GAT) aggregates information from a node's local neighbourhood and transforms it into a fixed‑size embedding vector.  
2. **Graph‑level aggregation** – node embeddings are pooled (via sum, mean, max, attention‑based pooling, etc.) together with edge attributes to produce a single representation for a subgraph or the full graph.  
3. **Task‑specific filtering** – the resulting graph‑level representation is passed through task‑specific layers (e.g., fully‑connected classifiers, similarity metrics) to extract the relevant information for the target problem, effectively filtering out irrelevant or noisy parts of the data.

Through these steps, graph deep learning turns an arbitrary, irregular data structure into a set of dense, informative vectors that can be processed by standard machine‑learning pipelines.

> **Figure** – The slide includes a humorous illustration of a person accompanied by meme‑style text, emphasizing the contrast between the apparent complexity of real‑world data and the elegance of well‑chosen algorithms that can tame that complexity.

*Reference*: A. Maier, V. Christlein, K. Breininger, S. Vesal, "Graph Deep Learning," July 3 2020.

## Graph Deep Learning Part 2

The first part of the lecture established that, by viewing a graph as a discrete manifold and discretising the Laplacian, we can define convolutions in the spectral domain through the eigenvectors $U$ of the symmetric normalised Laplacian and the eigenvalues $\Lambda$. In the spectral domain a convolution is a pointwise multiplication, and a polynomial filter gives a tractable parameterisation. The catch was that, in general, computing $U$ requires a full eigenvalue decomposition of an $N\times N$ symmetric matrix and, unlike on regular grids, no fast Fourier transform shortcut is available. Part 2 picks up from this point and asks two questions: (i) can we choose the polynomial cleverly enough to make $U$ disappear entirely, and (ii) do we even need to motivate graph convolutions from the spectral domain in the first place — or can we go back to a purely spatial picture?

## Graph Deep Learning

### Polynomial Approximation of the Graph Convolution Operator

In order to eliminate the matrix $\hat{U}$ from the expression $\hat{U}G\hat{U}^\top x$, we select the order $k$ of the polynomial and the coefficient vector $\theta$ in a particular way. Choosing a first‑order polynomial ($k=1$) and setting the coefficients to
\[
\theta_0 = 2\theta,\qquad \theta_1 = -\theta,
\]
the transformed graph signal can be written as

\[
\hat{U}G\hat{U}^\top x
= U\bigl(2\theta\,\Lambda^{0} - \theta\,\Lambda^{1}\bigr)U^\top x .
\]

Here $U$ is the matrix of eigenvectors of the (symmetric) graph Laplacian $\Delta_{\mathrm{sym}}$, and $\Lambda$ is the diagonal matrix of its eigenvalues. The intermediate filter matrix is therefore $\widehat{G}=2\theta\,\Lambda^{0}-\theta\,\Lambda$. Recalling that $\Lambda$ is diagonal, $\Lambda^{0}$ amounts to taking every diagonal element to the zero‑th power, which yields the identity matrix; the term $\Lambda^{1}$ is just $\Lambda$ itself. So $\widehat G$ already simplifies to $2\theta I-\theta\Lambda$ before we even bring back the $U$ factors.

Expanding the expression step by step yields:

1. **Separate the two terms inside the parentheses**  
   \[
   = \bigl(U\,2\theta\,\Lambda^{0} U^\top - U\,\theta\,\Lambda\,U^\top\bigr) x .
   \]

2. **Recognize that $\Lambda^{0}=I$ (the identity matrix)**  
   \[
   = \bigl(2\theta\,U I U^\top - \theta\,U\Lambda U^\top\bigr) x .
   \]

3. **Use the orthogonality of $U$ ($U U^\top = I$)**  
   \[
   = \bigl(2\theta\,I - \theta\,U\Lambda U^\top\bigr) x .
   \]

4. **Factor out the common scalar $\theta$**  
   \[
   = \theta\bigl(2I - U\Lambda U^\top\bigr) x .
   \]

5. **Replace $U\Lambda U^\top$ by the symmetric normalized Laplacian $\Delta_{\mathrm{sym}}$**  
   \[
   = \theta\bigl(2I - \Delta_{\mathrm{sym}}\bigr) x .
   \]

6. **Express $\Delta_{\mathrm{sym}}$ in terms of the degree matrix $D$ and adjacency matrix $A$**  
   The symmetric normalized Laplacian is defined as
   \[
   \Delta_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}.
   \]
   Substituting this definition gives
   \[
   = \theta\bigl(2I - I + D^{-1/2} A D^{-1/2}\bigr) x .
   \]

7. **Simplify the constant terms**  
   \[
   = \theta\bigl(I + D^{-1/2} A D^{-1/2}\bigr) x .
   \]

Thus, after the chosen parameterization, the original operation reduces to a simple linear combination of the input signal $x$ and the **symmetrically normalized adjacency matrix** $D^{-1/2} A D^{-1/2}$, scaled by the factor $\theta$:
\[
\boxed{\; \hat{U}G\hat{U}^\top x \;=\; \theta\bigl(I + D^{-1/2} A D^{-1/2}\bigr) x \;}
\]

Crucially, $U$ has now disappeared from the expression entirely — we have expressed the entire graph convolution using only the graph Laplacian's constituent matrices $D$ and $A$. Because $D$ is diagonal, both $D^{-1/2}$ and the inverse can be computed element‑wise, so there is no longer any cubic eigendecomposition cost. This is the key payoff that motivates the whole derivation: the only piece left to learn is the scalar $\theta$, and the heavy spectral machinery has been folded into the cheap algebraic identity above.

**Interpretation.**  
The term $I + D^{-1/2} A D^{-1/2}$ can be viewed as an *augmentation* of the identity matrix with a normalized adjacency component. This augmentation blends each node's own feature (the identity part) with a weighted aggregation of its neighbors' features (the $D^{-1/2} A D^{-1/2}$ part). Consequently, a single‑layer linear graph filter with this form already captures a form of first‑order neighbourhood smoothing, which forms the basis of many contemporary graph convolutional networks.

*Figure description:* The original slide presented a cascade of algebraic manipulations that start from a matrix expression involving the graph Laplacian components and a transformation matrix $\hat{U}$. Each step systematically simplifies the expression, ultimately revealing a term that combines the identity matrix with the symmetrically normalized adjacency matrix. The derivation illustrates how a careful choice of polynomial order $k$ and coefficients $\theta$ removes the dependence on $\hat{U}$.

The derivation above mirrors the one presented in the lecture notes for **Part 2** of the course, where the same first‑order polynomial ($k=1$, $\theta_0=2\theta$, $\theta_1=-\theta$) is used to obtain a *U‑free* expression. In those notes the intermediate matrix $\widehat{G}=2\theta I-\theta\Lambda$ is introduced explicitly, and the cancellation of $U$ and $U^\top$ is highlighted as the key step that enables a purely spatial implementation of the convolution. This observation was first formalised in the seminal paper by Kipf and Welling [1], which introduced the **Graph Convolutional Network (GCN)** as a first‑order approximation of spectral graph convolutions. The paper also coined the "renormalization trick", i.e. adding self‑loops to the adjacency matrix ($\tilde{A}=A+I$) and normalising with $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, precisely the form that appears in the boxed equation. The trick stabilises training by keeping the eigenvalues of the filter bounded and prevents the feature magnitudes from exploding across layers.

From a spatial viewpoint, the matrix $I + D^{-1/2} A D^{-1/2}$ implements a **mean‑aggregation** of each node's own representation with the representations of its immediate neighbours. This can be interpreted as a single diffusion step of a heat‑like process on the graph, where the heat at a node is updated by averaging with the heat of its neighbours — an intuition that aligns with the heat‑diffusion analogy discussed earlier in the lecture (see the "diffuse some heat" segment of Part 1). Because the operation is linear and depends only on the local neighbourhood, it can be efficiently implemented without constructing the full eigen‑basis, making it scalable to large graphs.

Finally, it is worth noting that while the derivation here is rooted in the spectral domain, the resulting operation is *equivalent* to many spatial aggregation schemes (e.g., GraphSAGE, GAT) when the aggregation function is chosen as a simple average. This duality between spectral and spatial perspectives is a recurring theme in graph deep learning and underlies the wide variety of GNN architectures explored in the literature.

A. Maier, V. Christlein, K. Breininger, S. Vesal, *Graph Deep Learning*, July 3 2020.

## Graph Deep Learning

### Spectral Convolution on Graphs

Convolution of a signal $x$ defined on the nodes of a graph can be performed in the spectral domain by first transforming the signal into the graph Fourier basis, applying a spectral filter, and then transforming back. The generic formulation is  

\[
x *_{\mathcal{G}} g \;=\; U \,\hat{G}\, U^{\top} x,
\]

where  

* $U \in \mathbb{R}^{N \times N}$ contains the eigenvectors of the normalized graph Laplacian $\mathcal{L}=I - D^{-1/2} A D^{-1/2}$ (the graph Fourier basis),  
* $\hat{G}$ is a diagonal matrix whose entries $\hat{g}(\lambda_i)$ are the spectral filter coefficients evaluated at the eigenvalues $\lambda_i$, and  
* $x \in \mathbb{R}^{N}$ is the input node feature vector.

#### Identity, polynomial choice and U‑free spatial form

The lecture summarises the chain of reasoning leading from the spectral identity to the spatial implementation in a single panel. Three observations are stitched together:

1. We can convolve in the spectral domain via $U\hat G U^\top x$.  
2. We can construct $\hat G$ as a polynomial of Laplacian filters, $\hat G(\Lambda)=\sum_{k=0}^{K}\theta_k\Lambda^k$.  
3. With the particular choice $K=1$, $\theta_0=2\theta$, $\theta_1=-\theta$ the filter depends only on the scalar value $\theta$, and all of the heavy spectral machinery — including the Fourier basis $U^\top$ — drops out.

This is exactly the chain summarised in the earlier derivation, and it leads to the spatial‑domain approximation

\[
U\hat{G}U^{\top} x \;=\; \theta\bigl(I + D^{-1/2} A D^{-1/2}\bigr) x,
\]

where the scalar parameter $\theta$ controls the filter strength. In this formulation the filter coefficients are expressed as  

\[
\theta_{0}=2\theta, \qquad \theta_{1}=-\theta,
\]

corresponding to the zeroth‑ and first‑order terms of the polynomial expansion. Consequently, the convolution reduces to a simple linear combination of the identity matrix $I$ (self‑connections) and the normalized adjacency matrix $D^{-1/2} A D^{-1/2}$ (neighbor aggregation).

This approximation is the foundation of many graph neural network (GNN) layers, such as the Graph Convolutional Network (GCN) layer introduced in [1], because it enables efficient, localized message passing without explicit spectral transforms. As the lecture stresses, this is "the basic graph convolutional operation": you can apply it to scalar values, you simply plug in your degree matrix $D$ and adjacency matrix $A$, and you optimize with respect to $\theta$ to find the weights of your convolutions. There is nothing left to compute beyond cheap sparse matrix–vector products.

> **Figure (described)**  
> The slide depicts a person surrounded by arrows pointing toward and away from the head, each arrow linked to mathematical expressions. The visual metaphor illustrates the transformation pipeline of a node feature vector $x$ as it passes through the spectral convolution process involving the Fourier basis $U$ and the filter $\hat{G}$, ultimately resulting in the spatial expression $\theta(I + D^{-1/2} A D^{-1/2})x$.

By removing the explicit Fourier transform $U^{\top}$ through the polynomial restriction, the operation becomes amenable to scalable, end‑to‑end learning on large graphs while preserving the essential notion of spectral filtering.

The normalized Laplacian $\mathcal{L}=I-D^{-1/2}AD^{-1/2}$ originates from the combinatorial Laplacian $L=D-A$, where $D$ is the degree matrix (a diagonal matrix containing node degrees) and $A$ the adjacency matrix. Normalization symmetrizes $L$, yielding a symmetric positive‑definite matrix even for originally directed graphs. This symmetry guarantees a real eigenbasis $U$ and non‑negative eigenvalues $\lambda_i$, which can be interpreted as **graph frequencies**: low eigenvalues correspond to smooth variations over the graph, while high eigenvalues capture rapid changes, mirroring the classical Fourier analysis on regular grids. The same eigenvectors are often called **graph Fourier modes** and — because they diagonalize $\mathcal{L}$ — they enable a clean definition of convolution in the spectral domain.

A useful physical analogy, highlighted in the lecture, is **heat diffusion**. If $f(x,t)$ denotes the temperature at node $x$ and time $t$, the heat equation on a graph can be written as $\partial_t f = -\mathcal{L}f$. The solution evolves by repeatedly applying the operator $e^{-\tau\mathcal{L}}$, which is precisely a low‑pass spectral filter that attenuates high‑frequency components. This viewpoint clarifies why polynomial filters of $\mathcal{L}$ — especially the first‑order one used in GCNs — perform a form of **local smoothing**: each node aggregates (averages) its own feature with those of its immediate neighbours.

Historically, the first‑order approximation was popularized by **Kipf & Welling (2016)** [1], who showed that restricting the filter to a linear function of $\mathcal{L}$ yields a simple, scalable layer that still captures the essence of spectral graph convolutions. Their "semi‑supervised classification with graph convolutional networks" paper is the canonical reference for the GCN formulation used throughout modern graph deep learning pipelines.

To illustrate the construction, consider the six‑node example discussed in the transcript. The degree matrix $D$ and adjacency matrix $A$ are

\[
D=\operatorname{diag}(2,3,2,3,3,1),\qquad
A=
\begin{pmatrix}
0&1&1&0&0&0\\
1&0&1&1&0&0\\
1&1&0&0&1&0\\
0&1&0&0&1&1\\
0&0&1&1&0&0\\
0&0&0&1&0&0
\end{pmatrix}.
\]

The unnormalized Laplacian is $L=D-A$, and after symmetrization the normalized Laplacian becomes $\mathcal{L}=I-D^{-1/2}AD^{-1/2}$. Plugging these matrices into the GCN update $\theta(I+D^{-1/2}AD^{-1/2})x$ yields the exact operation performed by a single graph convolutional layer on this toy graph.

Finally, while the spectral derivation provides a solid theoretical foundation, many practitioners prefer a **spatial interpretation**: each node updates its representation by aggregating transformed neighbor features, exactly as the term $D^{-1/2}AD^{-1/2}x$ suggests. This dual view bridges the gap to other spatial GNNs such as GraphSAGE [2], where the aggregation function can be mean, max‑pool, or even a learned LSTM, but the core idea of neighbour‑wise message passing remains the same.

## Graph Deep Learning

### Graph Convolutional Network (GCN) Operation

The Graph Convolutional Network (GCN) operation constitutes a fundamental building block in graph deep learning. Its primary purpose is to propagate information across the edges of a graph so that each node can acquire a representation (embedding) that reflects not only its own features but also the features of its neighboring nodes. This propagation is realized through a weighted aggregation of the feature vectors associated with adjacent nodes, followed by a learnable linear transformation.

The mathematical formulation of a single GCN layer is given by the following core equation:

$$
\theta \bigl(I + D^{-1/2} A D^{-1/2}\bigr) x
$$

where each symbol has a precise meaning:

- **$x$** denotes the matrix of input node features. If the graph contains $N$ nodes and each node is described by a $F_{\text{in}}$‑dimensional feature vector, then $x \in \mathbb{R}^{N \times F_{\text{in}}}$.
- **$A$** is the *adjacency matrix* of the graph. The entry $A_{ij}$ equals $1$ if there is an edge between node $i$ and node $j$, and $0$ otherwise. In many implementations a self‑loop is added to $A$ so that each node also attends to its own features.
- **$D$** is the *degree matrix*, a diagonal matrix whose $i$‑th diagonal entry $D_{ii}$ equals the degree of node $i$, i.e. the sum of the $i$‑th row of $A$: $D_{ii} = \sum_j A_{ij}$.
- **$I$** is the identity matrix, which together with $A$ implements the addition of self‑loops (the "+ I" term).
- **$D^{-1/2} A D^{-1/2}$** performs symmetric normalization of the adjacency matrix. This normalization rescales each edge by the inverse square root of the degrees of its endpoint nodes, mitigating the effect of nodes with high degree and ensuring that the aggregation operation is numerically stable.
- **$\theta$** represents the learnable weight matrix of the GCN layer. It maps the aggregated feature vectors from the input dimensionality $F_{\text{in}}$ to an output dimensionality $F_{\text{out}}$, i.e. $\theta \in \mathbb{R}^{F_{\text{in}} \times F_{\text{out}}}$. After the matrix multiplication, a non‑linear activation function (e.g., ReLU) is typically applied.

Interpretation of the equation proceeds as follows:

1. **Self‑loop addition**: The term $I + D^{-1/2} A D^{-1/2}$ constructs a normalized adjacency matrix that includes self‑connections. This guarantees that each node's own feature vector contributes to its new representation.
2. **Neighborhood aggregation**: Multiplying this normalized matrix by $x$ computes, for every node, a weighted sum of the feature vectors of its immediate neighbors (including itself). The weighting scheme accounts for node degrees, preventing high‑degree nodes from dominating the sum.
3. **Linear transformation**: The resulting aggregated feature matrix is then multiplied by the learnable parameter matrix $\theta$, allowing the network to adaptively combine the aggregated information.
4. **Non‑linearity** (typically added after this step) introduces expressive power, enabling the model to capture complex patterns in the graph structure.

This operation can be stacked across multiple layers, enabling information to flow beyond immediate neighbors and to capture higher‑order structural dependencies.

> **Figure description** – The original slide displayed a diagram of a simple graph consisting of nodes connected by edges. Alongside the visual, the adjacency matrix $A$ and the degree matrix $D$ were shown, illustrating how each matrix encodes the graph's connectivity. An arrow labeled "optimization step" indicated that the parameters $\theta$ are learned through gradient‑based optimization (e.g., stochastic gradient descent) by minimizing a task‑specific loss function.

The GCN formulation above was popularized by Kipf and Welling, who demonstrated its effectiveness for semi‑supervised node classification on citation networks and other graph‑structured data sets [1].

*Additional Context and Derivation*  
The GCN equation can be understood as a **first‑order approximation** of a spectral graph convolution. Earlier works (e.g., Bruna et al., 2013; Defferrard et al., 2016) expressed convolutions in the graph spectral domain by applying filters to the eigenvectors $U$ of the (symmetrically normalized) Laplacian $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. By expanding a generic spectral filter as a Chebyshev polynomial of the Laplacian and truncating the expansion to $K=1$, Kipf & Welling showed that the filter reduces to a simple linear combination of the identity and the normalized adjacency matrix, yielding exactly the term $I + D^{-1/2} A D^{-1/2}$. This "renormalization trick" (adding self‑loops before normalization) not only stabilizes training but also guarantees that the resulting propagation matrix is symmetric and row‑stochastic, mirroring a **random‑walk diffusion** process on the graph. Intuitively, each GCN layer performs a single step of a heat‑diffusion or smoothing operation, encouraging nearby nodes to have similar embeddings while preserving discriminative information through the learnable weights $\theta$.

*Historical Note*  
Kipf & Welling's GCN builds on the spectral graph convolution framework but deliberately avoids the costly eigen‑decomposition of $U$ by operating directly in the spatial domain. This design choice was motivated by the observation that the eigenbasis is expensive to compute for large graphs and that many practical tasks only require **local** information, which can be captured by aggregating over immediate neighborhoods. Consequently, the GCN became the de‑facto baseline for graph neural networks, later inspiring a plethora of spatial‑aggregation methods such as GraphSAGE, GAT, and many others.

*Relation to Spectral Normalization*  
Recall that the (symmetrically) normalized Laplacian can be written as $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. The GCN propagation matrix is simply $I - L_{\text{sym}}$, i.e. the identity minus the Laplacian, which highlights that a GCN layer acts as a **low‑pass filter** on the graph signal $x$. This perspective connects graph convolutions to classical signal‑processing notions such as diffusion, smoothing, and filtering on manifolds.

*Andreas Maier, Volker Christlein, Kai Breininger, and Saad Vesal. "Graph Deep Learning." Lecture, 3 July 2020.*

## Graph Deep Learning

### Question: Is it really necessary to motivate the Graph Convolution from Spectral Domain?

Having watched the spectral derivation collapse into a clean spatial expression, it is natural to ask the obvious follow‑up: **"Is it really necessary to motivate the graph convolution from the spectral domain?"** The lecturer's blunt answer is **"No."** The spectral story is mathematically beautiful and historically important, but the same operation can be motivated from scratch in a purely spatial manner — by directly thinking about how to aggregate information from a node's neighbours.

In the study of graph neural networks, it is common to encounter the *spectral* formulation of graph convolutions, which derives the operation from the eigendecomposition of the graph Laplacian. However, this spectral perspective is **not** strictly required in order to understand or develop graph convolutional layers. An alternative, equally valid approach is to motivate graph convolutions **spatially**, i.e., directly in terms of aggregating information from a node's local neighborhood on the graph. The spatial viewpoint emphasizes how each node updates its representation by combining features of its adjacent nodes, mirroring the intuition behind classical convolution on regular grids.

The lecture's setup of the two viewpoints is concise and worth restating in its own words: for a mathematician, a graph is a manifold (a discrete one), which can be discretised and convolved spectrally using the Laplacian — that is the route we have just travelled. But as a computer scientist, one can simply interpret a graph as a set of nodes and vertices connected through edges; we then need to define how to *aggregate the information of one vertex through its neighbours*, and once we do, we obtain the **spatial graph convolution**. The two perspectives describe the same object from opposite sides.

> **Figure (humorous illustration).**  
> An image of Robert Downey Jr. portraying Tony Stark (Iron Man) is shown with the caption "OMG!! REALLY???" superimposed. The illustration serves as a rhetorical device, humorously underscoring the speaker's stance that a purely spectral motivation is unnecessary.

The spatial formulation aligns naturally with the way most modern graph neural network architectures (e.g., GraphSAGE, GAT, GIN) are presented in practice. By focusing on message‑passing — each node receives "messages" from its neighbors, aggregates them (by sum, mean, max, or attention‑weighted combination), and then applies a learnable update function — researchers and practitioners can design and analyze graph convolutions without delving into the complexities of eigen‑bases or Fourier transforms on graphs.

> **Reference**  
> A. Maier, V. Christlein, K. Breininger, S. Vesal. *Graph Deep Learning*. July 3 2020.

In the lectures this point was illustrated by showing how the costly eigen‑decomposition of the symmetric normalized Laplacian $L = I - D^{-1/2} A D^{-1/2}$ can be avoided entirely. By restricting the spectral filter to a first‑order Chebyshev polynomial — i.e. choosing $k=1$, $\theta_0 = 2\theta$ and $\theta_1 = -\theta$ — the Fourier bases $U$ cancel out, yielding the simple propagation rule  

\[
H^{(\ell+1)} \;=\; \sigma\!\left(\,\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}\,H^{(\ell)}\,W^{(\ell)}\right),
\]

where $\tilde{A}=A+I$ adds self‑loops and $\tilde{D}$ is the degree matrix of $\tilde{A}$. This expression, first popularised by Kipf & Welling (2016) [1], is precisely the spatial message‑passing rule used in the original Graph Convolutional Network (GCN). The derivation shows that a spectral motivation leads naturally to the same spatial update once the polynomial filter is chosen to eliminate the Fourier transform, thereby justifying the spatial viewpoint from first principles.

Historically, the spectral approach dates back to early geometric deep learning work (e.g., Bruna et al., 2013), but the computational burden of full eigen‑decomposition limited its scalability. The transition to spatial aggregators was accelerated by Hamilton, Ying & Leskovec's GraphSAGE framework (2017) [2], which explicitly defined a learnable aggregation function (mean, max‑pool, or LSTM) over a node's neighbourhood. Subsequent models such as the Graph Attention Network (GAT) introduced attention‑weighted messages, and the Graph Isomorphism Network (GIN) demonstrated how a suitably powerful aggregating function can match the discriminative power of the Weisfeiler–Lehman graph isomorphism test (Xu et al., 2019). All of these architectures share the same underlying message‑passing paradigm described above, confirming that a purely spatial motivation is sufficient for practical GNN design.

The spatial perspective also clarifies why many modern GNN libraries implement a generic "aggregate‑combine" interface: the aggregation step handles the irregular neighbourhood sizes, while the combine step (often a linear transformation followed by a non‑linearity) updates the node's own representation. This modular view makes it straightforward to incorporate different aggregators (mean, max, attention, LSTM) and to stack multiple layers, leading to expressive hierarchical representations of graph‑structured data without ever referring to the Laplacian eigenbasis.

## Graph Deep Learning

### Two Perspectives: Spatial vs. Spectral

A **graph** is a mathematical structure consisting of a set of *nodes* (also called vertices) together with a set of *edges* that specify which pairs of nodes are connected. From a geometric viewpoint a graph can be regarded as a **discrete manifold**, i.e., a space that locally resembles a finite set of points with adjacency relations, rather than a smooth continuum.

There are two complementary ways of approaching graph deep learning, reflecting the different backgrounds of researchers:

1. **Computer‑science perspective (spatial approach).**  
   In this view the central problem is to design a mechanism that aggregates information from a given vertex and its immediate neighbors. The aggregation rule is applied repeatedly across the graph, yielding what is known as **spatial graph convolution**. Spatial methods are typically defined directly in terms of the graph's adjacency structure and are naturally suited to tasks where the locality of interactions (e.g., message passing) is important.

2. **Mathematical perspective (spectral approach).**  
   Here the graph is treated as a discretization of an underlying manifold. By constructing the **graph Laplacian matrix** $\mathbf{L}$ (often defined as $\mathbf{L} = \mathbf{D} - \mathbf{A}$, where $\mathbf{A}$ is the adjacency matrix and $\mathbf{D}$ the degree matrix), one can perform a **spectral decomposition** of signals on the graph. Convolution is then defined in the spectral domain by modulating the eigenvalues of $\mathbf{L}$, leading to **spectral graph convolution**. This approach leverages tools from harmonic analysis and provides a principled way to define filters that respect the global structure of the graph.

Both perspectives ultimately aim to learn expressive representations of graph‑structured data, but they differ in how they define the convolution operation — either directly in the vertex domain (spatial) or via the eigenbasis of the Laplacian (spectral). As demonstrated in the previous section, however, the two ultimately yield the same operation under the right polynomial parameterisation, so the choice between them is largely one of pedagogy and convenience.

> **Figure:** The slide illustrates a computer scientist and a mathematician facing each other, symbolizing the two distinct methodological approaches to graph deep learning.

*Source: A. Maier, V. Christlein, K. Breininger, S. Vesal, "Graph Deep Learning", July 3 2020.*

## Graph Deep Learning

### Spatial Aggregation: Vertex of Interest and Neighborhood

In graph‑structured data we are often interested in computing a representation for a particular vertex (or node) while taking into account the information stored in its neighboring vertices. The lecture introduces the spatial approach in two stages — first as a conceptual recipe and then as a concrete formula.

**Practical formulation.**  
1. **Define a vertex of interest.** Choose a node $v$ whose representation we wish to update or extract.  
2. **Specify how neighbors contribute.** Determine a rule that aggregates information from the set of neighboring vertices $N(v)$ (the *neighborhood* of $v$) and combines it with the current state of $v$.

**Technical formulation.**  
Let $h_v^{k}$ denote the feature (or hidden) vector associated with node $v$ at layer $k$ of a graph neural network (GNN). Typically, the zeroth layer contains the raw input features,
\[
h_v^{0}=x_v,
\]
where $x_v$ is the initial attribute vector attached to $v$ — that is, the original configuration of the graph.

For each subsequent layer we aggregate the current representation of $v$ with the representations of its neighbors. Formally, for every neighbor $u \in N(v)$ we collect $h_u^{k}$ and apply an aggregation function (e.g., sum, mean, max, or a learned attention mechanism):
\[
\boxed{
\;h_v^{k+1}= \text{AGGREGATE}^{k}\!\big(\{\,h_u^{k}\mid u\in N(v)\,\}\big)\;}
\]
The result $h_v^{k+1}$ is the updated feature vector for $v$ in layer $k+1$. This operation can be repeated across multiple layers, allowing information to propagate further away from the original vertex. In many implementations the aggregation is followed by a transformation (e.g., a linear mapping and a non‑linear activation) that mixes the aggregated neighbor information with $h_v^{k}$ itself.

The neighbourhood $N(v)$ is typically defined to contain every node directly connected to $v$ — i.e. all 1‑hop neighbours — although the framework is flexible enough to admit larger or sampled neighbourhoods. This aggregation step is the elementary building block of spatial GNNs and is directly inspired by the lecturer's statement that, having defined the vertex of interest, "we then need to be able to aggregate in order to compute the next layer," with the aggregation taking the form of a spatial function over the previous layer.

The iterative application of the aggregation step across layers is illustrated in the figure below. The central node $v$ receives messages from its four neighbors $u^{0},u^{1},u^{2},u^{3}$; after each layer the superscript on each node's representation indicates its depth in the network.

> **Figure (description).** The diagram shows a graph with a central node $v$ connected to four neighboring nodes $u^{0}, u^{1}, u^{2}, u^{3}$. Arrows point from each neighbor toward $v$, indicating the direction of information flow during the aggregation step. This process is repeated for successive layers, as reflected by the superscript indices on the node representations.

Graph convolutional networks (GCNs) and related GNN architectures rely on this simple yet powerful message‑passing scheme to learn inductive representations on large graphs [2] and have been applied successfully in domains such as medical image analysis, e.g., for coronary artery segmentation in cardiac CT angiography [3]. In particular, **GraphSAGE** (Hamilton et al. [2]) is the most prominent concrete realisation of this template, defining a learnable aggregation function and combining it with the node's current state through a linear transform and a nonlinearity.

## GraphSAGE - The Algorithm

### GraphSAGE – The Algorithm

GraphSAGE (Graph Sample and Aggregate) is an inductive framework for generating node embeddings on large graphs. The method proceeds by a forward‑propagation scheme that repeatedly aggregates information from a node's local neighbourhood and then transforms the aggregated representation with learnable parameters and a non‑linear activation.

Formally, let  

* a graph $G(V, E)$ with vertex set $V$ and edge set $E$;  
* an input feature vector $x_v$ for each node $v \in V$;  
* a depth (or number of hops) $K \in \mathbb{N}$;  
* a collection of weight matrices $W^{k}$ for each layer $k \in \{1,\dots,K\}$;  
* a non‑linearity $\sigma(\cdot)$ (e.g., ReLU); and  
* a differentiable neighbourhood aggregator $\operatorname{AGGREGATE}_{k}(\cdot)$ for each layer.

The algorithm returns a final embedding vector $z_v$ for every node $v\in V$. The computation can be described step‑by‑step as follows.

1. **Initial node representations.**  
   For every node $v$ the representation at layer 0 is set to its input feature:  

   \[
   h^{v}_{0} \;\leftarrow\; x_{v},\qquad \forall v\in V .
   \]

   This is just the original configuration of the graph — the lecture stresses that "for the zeroth layer this contains the input."

2. **Iterative neighbourhood aggregation (layers $k=1,\dots,K$).**  
   For each layer $k$ we update the representation of every node by (i) aggregating the representations of its neighbours from the previous layer and (ii) applying a linear transformation followed by the non‑linearity. Concretely, for each node $v\in V$:

   * **Neighbour aggregation.**  
     Compute an aggregated message from the set of neighbours $N(v)$ (the set of nodes directly connected to $v$):

     \[
     m^{v}_{k}\; \leftarrow\; \operatorname{AGGREGATE}_{k}\!\bigl(\,\{\,h^{u}_{k-1}\mid u\in N(v)\,\}\bigr).
     \]

     The aggregator reduces the variable‑length set of neighbour feature vectors into a single fixed‑dimension summary vector — exactly the "summary over all of your neighbours" that the lecturer describes.

   * **Combine with the node's own previous representation.**  
     Concatenate the node's own representation $h^{v}_{k-1}$ with the aggregated message $m^{v}_{k}$, apply the learnable weight matrix $W^{k}$, and finally the non‑linearity $\sigma$:

     \[
     h^{v}_{k}\; \leftarrow\; \sigma\!\Bigl( \, W^{k}\,\cdot\,\operatorname{CONCAT}\bigl(h^{v}_{k-1},\, m^{v}_{k}\bigr) \Bigr).
     \]

   This loop is executed for all nodes before proceeding to the next layer, ensuring that each node's representation at depth $k$ incorporates information from its $k$‑hop neighbourhood.

   The choice of the aggregator $\operatorname{AGGREGATE}_{k}$ is a central design decision. In the original GraphSAGE paper the authors describe several families of aggregators that can be plugged into the same framework: a simple **mean** aggregator that computes the average of neighbour embeddings, a **max‑pooling** aggregator that applies a learned linear transformation followed by element‑wise max‑pooling, and a **recurrent** aggregator based on an LSTM that processes neighbour features sequentially. A fourth option, often called the **GCN aggregator**, reproduces the spectral graph convolution of Kipf & Welling by using a normalized sum of neighbour features; this illustrates how GraphSAGE bridges the spatial‑domain view (neighbour aggregation) and the spectral‑domain view (graph Laplacian‑based filters) mentioned in the lecture notes. Because the aggregation operates on a set of neighbour vectors, it naturally handles nodes with varying degrees, and the subsequent concatenation with the node's own representation preserves the "self‑information" while allowing the model to learn how to weight neighbour versus self contributions.

3. **Optional L2‑normalisation.**  
   After the final layer $K$ the obtained representations are often normalised to unit length to stabilise training and facilitate similarity comparisons:

   \[
   h^{v}_{K}\; \leftarrow\; \frac{h^{v}_{K}}{\lVert h^{v}_{K}\rVert_{2}},\qquad \forall v\in V .
   \]

   The lecture summarises this as "scaling by the magnitude of your activations" — it is the final per‑node normalisation step before producing the output embedding.

4. **Output embeddings.**  
   The final embedding for each node is simply the (normalised) representation from the last layer:

   \[
   z_{v}\; \leftarrow\; h^{v}_{K},\qquad \forall v\in V .
   \]

**Figure (description).**  
The above procedure constitutes the GraphSAGE algorithm, a method for generating embeddings on large graphs. By iteratively aggregating features from a node's neighbourhood and transforming them with learnable weights and a non‑linear activation, GraphSAGE produces expressive node embeddings that can be used for downstream tasks such as classification, link prediction, or clustering.

The original exposition of this method appears in Hamilton, Ying, and Leskovec [2].

## Graph Deep Learning: GraphSAGE - Aggregators

### Why aggregators matter

The concept of an **aggregator** is, in the lecturer's words, "key to develop this algorithm because in every node you may have a different number of neighbours." A graph convolutional layer must be able to summarise an arbitrarily‑sized set of neighbour embeddings into a single fixed‑dimension vector while remaining permutation‑invariant (the result should not depend on the arbitrary order in which the neighbours are listed). Different aggregator choices give rise to qualitatively different families of graph deep‑learning approaches; in particular, taking the GCN aggregator brings one back to the spectral representation, which establishes the explicit connection between the spatial and spectral domains. The four aggregators below — mean, GCN, pooling, and LSTM — are the four canonical choices presented in the GraphSAGE paper and discussed in the lecture, and their existence helps explain the **broad variety** of graph deep‑learning methods, which can be subdivided into spectral, spatial, and recurrent families precisely along these aggregator‑based lines.

### Mean Aggregator

The **Mean Aggregator** is the simplest form of neighbor aggregation used in the GraphSAGE framework. For a target node $v$, it computes the arithmetic mean of the hidden representations of all its neighboring nodes $\mathcal{N}(v)$ from the previous layer $k-1$. This average summarizes the local neighborhood information in a permutation‑invariant way.

The update rule for the hidden representation $h_v^k$ at layer $k$ is  

\[
h_v^{k} \;\leftarrow\; \sigma\!\Bigl( W \cdot \operatorname{MEAN}\bigl(\{h_u^{\,k-1}\}_{u \in \mathcal{N}(v)}\bigr) + b \Bigr),
\qquad \forall v \in \mathcal{V},
\]

where  

* $W$ and $b$ are learnable weight matrix and bias vector,  
* $\sigma(\cdot)$ denotes a point‑wise nonlinearity (e.g., ReLU), and  
* $\operatorname{MEAN}(\cdot)$ denotes the element‑wise average over the set of neighbor embeddings.  

Intuitively, each node gathers a "message" that is the mean of its neighbors' features, linearly transforms it, adds a bias, and finally applies a non‑linear activation. This operation is analogous to the convolution step in classical convolutional neural networks, but it is defined over an irregular graph topology.

> The mean aggregator was introduced in the original GraphSAGE paper as a baseline that is trivially permutation‑invariant and computationally cheap, making it suitable for the inductive setting where new nodes appear at test time. It can be seen as a normalized **sum** aggregator, i.e. $\operatorname{MEAN}(\cdot)=\frac{1}{|\mathcal{N}(v)|}\sum_{u\in\mathcal{N}(v)}(\cdot)$, which directly addresses the variable‑size neighbourhood problem highlighted in the lecture notes.

### GCN Aggregator

The **GCN Aggregator** can be viewed as a special case of the Mean Aggregator that incorporates a symmetric normalization of the adjacency structure. In the original graph convolutional network (GCN) formulation, the aggregation step multiplies the neighbor feature matrix by the normalized adjacency matrix $\hat{A}= \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, where $\tilde{A}=A+I$ adds self‑loops and $\tilde{D}$ is the degree matrix of $\tilde{A}$.

When we restrict the GCN aggregator to the GraphSAGE setting, the update simplifies to  

\[
h_v^{k} \;\leftarrow\; \sigma\!\Bigl( W \cdot \sum_{u \in \mathcal{N}(v)} \frac{1}{\sqrt{|\mathcal{N}(v)||\mathcal{N}(u)|}}\; h_u^{\,k-1} + b \Bigr).
\]

Thus, the GCN aggregator still performs a weighted average of neighbor embeddings, but the weights depend on the degrees of the source and target nodes, which mitigates the bias toward high‑degree nodes that a plain mean would incur. Because the normalization is fixed, the GCN aggregator has fewer learnable parameters than a generic mean aggregator with an additional attention mechanism.

> This weighting scheme originates from Kipf & Welling's spectral‑to‑spatial derivation (see reference [1] in the lecture notes). In the transcript it is highlighted that choosing a first‑order polynomial ($k=1$) and appropriate coefficients removes the need for an explicit eigen‑decomposition, yielding exactly the symmetric normalized adjacency used here. Consequently, the GCN aggregator bridges the spectral view of graph convolutions with the spatial, message‑passing perspective emphasized in the GraphSAGE algorithm — selecting it as the GraphSAGE aggregator is, as the lecture remarks, what "brings us back to the spectral representation."

### Pooling Aggregator

The **Pooling Aggregator** introduces a non‑linear, order‑sensitive transformation before the reduction step. Each neighbor's hidden state is first passed through a learnable linear map $W_{\text{pool}}$, followed by a nonlinearity $\sigma(\cdot)$. The transformed neighbor representations are then combined by a *max‑pooling* operation, which selects the most salient feature across all neighbors for each dimension.

Formally, the aggregation at layer $k$ for node $v$ is

\[
\text{AGGREGATE}_{k}^{\text{pool}}(v)
   = \max_{u \in \mathcal{N}(v)} \; \sigma\!\bigl( W_{\text{pool}} \, h_u^{\,k} + b \bigr),
\]

where the $\max$ operator is applied element‑wise over the set of transformed neighbor vectors. After pooling, the resulting vector is concatenated with the target node's own representation (or further processed by a linear layer) to form the updated embedding $h_v^{k}$.

Pooling can capture more expressive, localized patterns because the max operation emphasizes the strongest activation across a neighbourhood, akin to the way max‑pooling in CNNs highlights the most prominent visual features.

> In the GraphSAGE paper, the **max‑pooling** aggregator was proposed precisely to increase expressive power beyond simple averaging. The transcript notes that "a broad variety of aggregators… includes maximum pooling," underscoring its role as a spatial, order‑invariant yet non‑linear summarizer. Empirically, max‑pooling has been shown to improve performance on tasks where the presence of a single highly informative neighbour is more important than the overall average (e.g., detecting anomalous nodes).

### LSTM Aggregator

The **LSTM Aggregator** treats the neighbor set $\mathcal{N}(v)$ as a sequence and feeds the neighbor embeddings into a recurrent network — specifically a Long Short‑Term Memory (LSTM) model. The LSTM processes the sequence $\{h_{u_1}^{\,k-1}, h_{u_2}^{\,k-1}, \dots, h_{u_{|\mathcal{N}(v)|}}^{\,k-1}\}$ and produces a final hidden state $h_{\text{LSTM}}^{\,k}(v)$ that serves as the aggregated neighbourhood representation.

The update equation can be written abstractly as  

\[
h_v^{k} \;\leftarrow\; \sigma\bigl( W \, h_{\text{LSTM}}^{\,k}(v) + b \bigr),
\]

where $h_{\text{LSTM}}^{\,k}(v)$ is obtained by iterating the standard LSTM recurrence over the neighbor feature sequence. By leveraging the LSTM's gating mechanisms, this aggregator can, in principle, learn to weight neighbors differently based on their order or content, thereby capturing more complex, possibly directional, dependencies among neighboring nodes.

> The lecture notes remark that "recurrent networks like LSTM aggregators" belong to the family of spatial aggregators. Because graphs are inherently unordered, applying an LSTM requires an imposed ordering (e.g., random, degree‑sorted). While this adds expressive flexibility, the transcript also points out the computational overhead compared to the simpler mean or GCN aggregators. Subsequent works have therefore favored attention‑based aggregators (e.g., Graph Attention Networks) as a more efficient way to learn adaptive weights without sequence ordering.

> **Figure note:** No explicit diagram is provided in the original material. The discussion, however, covers four distinct aggregator functions — Mean, GCN, Pooling, and LSTM — each extending the basic message‑passing paradigm of GraphSAGE in a different way.

## Graph Deep Learning

### Graph Neural Network Architecture Landscape

The field of graph deep learning has rapidly expanded, and by 2019 a substantial variety of Graph Neural Network (GNN) architectures had been proposed. The slide visualises these architectures with a grid‑like diagram and groups them according to their core mechanisms and historical lineage. The spatial arrangement in the diagram reflects similarities in design principles — for example, models that share a message‑passing formulation tend to cluster together, whereas architectures that introduce novel pooling or attention mechanisms occupy distinct regions. This visual taxonomy helps researchers locate a new method within the broader research landscape and understand which prior works it builds upon or diverges from. As the lecturer remarks, the very existence of so many variants — spectral, spatial, recurrent, attention‑based — is a direct consequence of the freedom in choosing the aggregator function and graph representation.

Below is a non‑exhaustive but representative list of the models that appear in the diagram. The models are ordered alphabetically for readability; each entry corresponds to a distinct methodological contribution to graph representation learning.

- **GNN** – The original graph neural network framework that introduced recurrent message passing on graph nodes.
- **MPNN** – Message Passing Neural Network, a unifying formalism that captures many later GNN variants.
- **MoNet** – A framework for defining convolution‑like operators on non‑Euclidean domains via pseudo‑coordinates.
- **GraphSAGE** – An inductive approach that samples and aggregates neighbourhood information.
- **PATCHY‑SAN** – A method that extracts a fixed‑size receptive field from graphs to enable convolution.
- **GAT** – Graph Attention Network, which assigns learned attention coefficients to edges during aggregation.
- **DGCNN** – Deep Graph Convolutional Neural Network, emphasizing hierarchical graph convolution and sorting.
- **FastGCN** – A scalable variant of GCN that treats graph convolutions as integral transforms and uses Monte‑Carlo sampling.
- **SumPooling** – A simple global pooling operation that aggregates node embeddings by summation.
- **AvgPooling** – Global pooling by averaging node embeddings.
- **GGNN** – Gated Graph Neural Network, which incorporates gated recurrent units into message passing.
- **DGI** – Deep Graph Infomax, a self‑supervised method that maximizes mutual information between local and global representations.
- **CGMM** – Conditional Graphical Model Mechanism for learning on graph‑structured data.
- **GraphESN** – Graph Echo State Network, an extension of reservoir computing to graphs.
- **SortPooling** – A pooling scheme that sorts node embeddings before truncation to obtain a fixed‑size graph representation.
- **SSE** – Spectral Sampling Encoder, which leverages spectral graph theory for pooling.
- **DiffPool** – Differentiable pooling that learns hierarchical cluster assignments.
- **GAAN** – Graph Attention Aggregation Network, combining attention with adaptive aggregation.
- **Huang et al.** – The architecture introduced in Huang et al.'s work on graph convolutions (specific details omitted for brevity).
- **PGC‑DGCNN** – A hybrid that combines Pattern‑Guided Convolutions with Deep Graph CNN.
- **SpectralCNN** – Convolutional networks defined in the spectral domain of the graph Laplacian.
- **NN4G** – Neural Network for Graphs, a generic term for early dense‑layer based graph models.
- **AGCN** – Adaptive Graph Convolutional Network, which learns edge weights dynamically.
- **GeniePath** – A model that separates feature propagation and transformation into two learnable stages.
- **Henaff et al.** – The spectral graph convolution approach proposed by Henaff and colleagues.
- **DualGCN** – A dual‑graph convolutional network that processes node and edge features jointly.
- **MaxPooling** – Global pooling that retains the maximum activation across nodes.
- **ClusterGCN** – A mini‑batch training scheme that clusters the graph to reduce computational load.
- **LGCN** – Localized Graph Convolutional Network, emphasizing locality in spectral filters.
- **GCN** – Graph Convolutional Network, the seminal spectral‑spatial hybrid that popularized graph convolutions.
- **StoGCN** – Stochastic Graph Convolutional Network, introducing randomness into the aggregation process.
- **CayleyNet** – A spectral GNN that employs Cayley filters for expressive frequency response.
- **GlobalAttentionPooling** – An attention‑based global pooling mechanism that learns to weight node embeddings.
- **Set2Set** – An order‑invariant readout that models the graph as a set and uses a recurrent decoder.
- **DCNN** – Diffusion Convolutional Neural Network, which leverages diffusion processes on graphs.
- **ChebNet** – Chebyshev Network; approximates spectral graph convolutions using Chebyshev polynomials, avoiding explicit eigen‑decomposition.
- **Spectral / Spatial / Recurrent** – Three meta‑categories that group GNNs by the way information is propagated: via the eigen‑basis of the graph Laplacian, via direct neighbourhood aggregation, or via temporal recurrence over graph snapshots.

These architectures collectively illustrate the breadth of methodological innovations in graph deep learning, ranging from different message‑passing schemes and spectral filter designs to a variety of pooling and attention mechanisms. Understanding the relationships among them is essential for selecting an appropriate model for a given graph‑structured problem.

#### Spatial versus recurrent patterns

Spatial GNNs operate directly on the graph topology: each node aggregates information from its immediate neighbours according to a predefined or learned weighting scheme. This approach mirrors the way classical convolutional neural networks slide filters over a regular grid, but the "slide" follows the irregular connectivity of the graph.

Recurrent GNNs, by contrast, model **temporal dependencies** within a sequence of graph states (e.g., evolving social networks or video frames represented as graphs). They typically employ gated recurrent units (GRUs) or LSTMs to propagate hidden states over time, allowing the network to capture dynamic relationships that change across successive graph instances.

#### Additional technical context

The taxonomy above can be grounded in two complementary perspectives on graph convolutions that were emphasized in the lecture: the **spectral** view and the **spatial** view. The spectral approach originates from treating a graph as a discrete manifold and employing the graph Laplacian $L = D - A$ (or its normalized form $\tilde L = I - D^{-1/2} A D^{-1/2}$) as a shift operator. By diagonalising $\tilde L$ as $\tilde L = U \Lambda U^\top$, the eigenvectors $U$ become graph Fourier modes and the eigenvalues $\Lambda$ act as frequencies. Convolution can then be defined as a pointwise multiplication in this Fourier domain, i.e. $\hat y = g_\theta(\Lambda) \hat x$, with $g_\theta$ a learnable spectral filter. However, evaluating $U$ is costly for large graphs, which motivated the search for **polynomial approximations** of $g_\theta(\Lambda)$. A first‑order Chebyshev polynomial leads directly to the propagation rule of the **Graph Convolutional Network (GCN)** introduced by Kipf & Welling [1], where the filter reduces to $(I + D^{-1/2} A D^{-1/2})$ multiplied by a learnable weight matrix. This derivation explains why GCN appears in the list as a seminal spectral‑spatial hybrid.

The spatial perspective sidesteps the eigen‑decomposition entirely by defining **message‑passing** directly on the graph topology. In this view, a node $v$ updates its hidden state $h_v^{(k)}$ by aggregating transformed features from its neighbours $\mathcal N(v)$, e.g.
\[
h_v^{(k)} = \sigma\!\left( W^{(k)} \cdot \text{AGG}\bigl(\{ h_u^{(k-1)} \mid u \in \mathcal N(v) \}\bigr) \right),
\]
where $\text{AGG}$ may be a mean, sum, max, or attention‑weighted operation. This formulation underlies many of the architectures listed, such as **GraphSAGE** [2], which samples a fixed‑size neighbourhood and concatenates the aggregated neighbour vector with the node's own representation before applying a linear transform and nonlinearity. The lecture detailed the GraphSAGE pipeline: initialise $h^{(0)}$ with raw node features, loop over layers, for each node aggregate neighbour embeddings, concatenate with the node's current embedding, apply a weight matrix, non‑linear activation, and finally normalise by the activation magnitude. Variants of the aggregator (mean, max‑pooling, LSTM, attention) give rise to the multitude of GNN families displayed in the diagram, including **GAT** (attention‑based aggregation) and **AGCN** (adaptive edge‑weight learning).

Historically, the **Message Passing Neural Network (MPNN)** framework was later proposed as a unifying abstraction that captures both spectral‑derived and spatial‑derived GNNs by specifying a message function, an aggregation function, and an update function. Many of the entries — such as **GGNN**, **GeniePath**, and **DualGCN** — can be interpreted as particular instantiations of the MPNN schema, differing mainly in how they treat edge features, incorporate gating mechanisms, or perform hierarchical pooling. Formally, MPNN updates can be written as
\[
m_{v}^{(l)} = \!\!\sum_{u\in \mathcal{N}(v)}\! M\bigl(h_{v}^{(l)},h_{u}^{(l)},e_{vu}\bigr),\qquad
h_{v}^{(l+1)} = U\bigl(h_{v}^{(l)}, m_{v}^{(l)}\bigr),
\]
where the choice of message function $M$ and update function $U$ recovers the various architectures (e.g., attention coefficients in GAT, mean‑pooling in GraphSAGE, gated updates in GGNN).

Recurrent GNNs such as **GGNN** extend the message‑passing framework across time steps $t$, employing gated recurrent units:
\[
h_v^{(t+1)} = \text{GRU}\!\bigl(h_v^{(t)},\,m_v^{(t)}\bigr),
\]
making them suitable for dynamic graphs where edges appear or disappear over successive snapshots.

Finally, the lecture highlighted the evolution from purely spectral designs (e.g., **SpectralCNN**, **CayleyNet**) toward **hierarchical pooling** strategies that enable graph‑level readouts. Methods like **DiffPool**, **SortPooling**, **Set2Set**, and the simple **SumPooling/AvgPooling/MaxPooling** listed above embody this trend, allowing GNNs to produce fixed‑dimensional graph embeddings suitable for classification, regression, or retrieval tasks. The self‑supervised approach **Deep Graph Infomax (DGI)** further illustrates how mutual‑information maximisation can be leveraged to learn expressive node and graph representations without explicit labels.

#### Applications: from meshes to coronary arteries

Beyond the canonical citation‑network benchmarks, the lecture stresses that any of these algorithms can be applied to **meshes**, including very complex ones, and points to the references for examples of what kind of applications are possible. A particularly compelling biomedical example is the use of graph convolutional networks to process information defined on **coronary artery** trees: the vascular tree is naturally modelled as a graph with nodes along the centreline and edges connecting consecutive points, and a GCN can be trained to label vessel segments or perform segmentation directly on this graph [3]. Similar ideas appear in **material science**, **traffic forecasting**, and **protein–protein interaction prediction**, illustrating how the abstract message‑passing machinery generalises across very different domains.

> *Figure:* The image depicts a complex network of interconnected nodes, resembling a brain's neural connections. Distinct regions are highlighted in varying shades of blue, green, and yellow, visually emphasizing the connections and paths. This illustration conveys the intricate relationships that GNN architectures aim to capture and process.

#### Outlook

Looking ahead, the lecture closes by hinting at the next topic — **embedding prior knowledge into deep networks**. This is a natural continuation of the present discussion because, as the lecturer puts it, it allows much of what we know from theory and signal processing to be fused with deep learning approaches. Graph deep learning is itself a prime example of this fusion: the spectral derivation of the GCN layer is, in essence, an injection of graph signal‑processing theory (Laplacians, Fourier modes, polynomial filters) into a learnable neural network. The same spirit will guide the next lecture's treatment of how broader prior knowledge can be incorporated into deep models.

#### References

[4] Wu, Zonghan, et al. "A comprehensive survey on graph neural networks." *arXiv preprint* arXiv:1901.00596 (2019).  
[1] Kipf, Thomas N., and Max Welling. "Semi‑supervised classification with graph convolutional networks." *arXiv preprint* arXiv:1609.02907 (2016).  
[2] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." *Advances in Neural Information Processing Systems* 30 (2017).

*Prepared by A. Maier, V. Christlein, K. Breininger, S. Vesal – Graph Deep Learning, 3 July 2020*

## References

### References

The following bibliography provides a concise yet comprehensive overview of key contributions to graph deep learning. It includes seminal papers on graph convolutional networks, methods for inductive representation learning on large graphs, applications in medical imaging, an exhaustive survey of graph neural network architectures, and a tutorial on geometric deep learning for graphs and manifolds. These works are essential reading for anyone researching or developing models within the graph deep learning domain.

1. **[1]** Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." *arXiv preprint* arXiv:1609.02907 (2016).  

2. **[2]** Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." *Advances in Neural Information Processing Systems* (2017).  

3. **[3]** Wolterink, Jelmer M., Tim Leiner, and Ivana Išgum. "Graph convolutional networks for coronary artery segmentation in cardiac CT angiography." In *International Workshop on Graph Learning in Medical Imaging*, Springer, Cham (2019).  

4. **[4]** Wu, Zonghan, et al. "A comprehensive survey on graph neural networks." *arXiv preprint* arXiv:1901.00596 (2019).  

5. **[5]** Bronstein, Michael et al. Lecture "Geometric deep learning on graphs and manifolds" held at SIAM Tutorial Portland (2018).  

The first reference ([1]) introduced the now‑canonical graph convolutional network (GCN) that operates directly on the symmetrically‑normalised graph Laplacian, bridging the gap between spectral graph theory and practical deep learning. Building on earlier spectral formulations (e.g., Bruna et al., 2013), Kipf and Welling's linear‑approximation of Chebyshev polynomials made GCNs scalable to large sparse graphs and sparked a wave of subsequent works.

Reference [2] presents **GraphSAGE**, an inductive framework that learns node embeddings by aggregating feature information from a node's neighbourhood. Unlike the transductive setting of the original GCN, GraphSAGE can generate representations for previously unseen nodes, which is crucial for dynamic or evolving graphs. The algorithm's flexible aggregators (mean, max‑pooling, LSTM‑based) have become a standard building block in many modern spatial GNN architectures.

The medical‑imaging application in reference [3] showcases how GCNs can be adapted to segment coronary arteries in cardiac CT angiography. By constructing a graph over the vessel centreline and feeding voxel‑wise intensity features into a GCN, the authors achieved state‑of‑the‑art segmentation performance while respecting the underlying vascular topology — illustrating the practical impact of graph‑based deep models in clinical settings.

Reference [4] is a widely‑cited survey that systematically categorises graph neural network designs into spectral, spatial, and hybrid families, discusses issues such as over‑smoothing, scalability, and expressive power, and provides a curated benchmark table. It serves as a one‑stop reference for newcomers seeking a panoramic view of the field up to 2019.

Finally, the tutorial by Michael Bronstein ([5]) laid the conceptual foundation of **geometric deep learning** by unifying deep learning on graphs, manifolds, and other non‑Euclidean domains. The lecture, delivered at the SIAM Tutorial in Portland (2018), is repeatedly acknowledged in the course materials and was explicitly thanked by Prof. Maier in the "Thanks" slide of the lecture notes. Special thanks are also recorded in both transcripts to **Florian Thamm** for preparing this set of slides.

*Figure: This slide contains a list of references pertaining to graph deep learning. The references span a range of topics, including graph convolutional networks, inductive representation learning, and applications in medical imaging. These resources are relevant for individuals researching and working within the field of graph deep learning.*

## Image References

### Image References

The following collection enumerates the URLs of all images that were incorporated throughout the lecture on graph deep learning. These sources include visualizations of mathematical functions, climate data plots, historical photographs, geometric shapes, and examples of biological and medical imaging. They were used to illustrate key concepts, provide real‑world context, and support the narrative of the presentation.

* **[a]** https://de.serlo.org/mathe/funktionen/funktionsbegriff/funktionen-graphen/graph-funktion – Graph of a mathematical function.  
* **[b]** https://www.nwrfc.noaa.gov/snow/plot_SWE.php?id=AFSW1 – Snow water equivalent (SWE) plot from NOAA.  
* **[c]** https://tennisbeiolympia.wordpress.com/meilensteine/steffi-graf/ – Photograph of tennis player Steffi Graf.  
* **[d]** https://www.pinterest.de/pin/624381935818627852/ – Image of a stylized pentagram.  
* **[e]** https://www.uihere.com/free-cliparts/the-pentagon-pentagram-symbol-regular-polygon-golden-five-pointed-star-2282605 – Diagram of a regular pentagon and pentagram.  
* **[f]** http://geometricdeeplearning.com/ – Homepage of the "Geometric Deep Learning on Graphs and Manifolds" project.  
* **[g]** https://i.stack.imgur.com/NU7y2.png – Illustration of a graph‑neural‑network architecture.  
* **[h]** https://de.wikipedia.org/wiki/Datei:Convolution_Animation_(Gaussian).gif – Animated GIF demonstrating Gaussian convolution.  
* **[i]** https://www.researchgate.net/publication/306293638/figure/fig1/AS:396934507450372@1471647969381/Example-of-centerline-extracted-left-and-coronary-artery-tree-mesh-reconstruction.png – Example of a centerline‑extracted coronary artery mesh reconstruction.  
* **[j]** https://www.eurorad.org/sites/default/files/styles/figure_image_teaser_large/public/figure_image/2018-08/0000015888/000006.jpg?itok=hwX1sbCO – Radiological image used as a case study.

*Figure*: The slide simply lists the image references used within the broader presentation on graph deep learning. These URLs point to a variety of images — including graphs, diagrams, and real‑world examples — employed to illustrate the concepts discussed.

The slide's image list was deliberately curated to span several domains that graph‑based methods can address.  

- **Mathematical graph (a)** serves as the canonical example of a node‑edge structure, grounding the abstract definition of a graph in a familiar visual of a function plot.  
- **The NOAA snow‑water‑equivalent map (b)** demonstrates how irregular spatial data can be represented as a graph, a motif that recurs when discussing diffusion processes on non‑Euclidean domains.  
- **The Steffi Graf photograph (c)** appears as a tongue‑in‑cheek reminder that the word "graph" has many everyday meanings; the lecturer uses it to highlight the importance of precise terminology in computer‑science contexts.  
- **Pentagram and pentagon images (d, e)** illustrate regular, highly symmetric graph topologies. Such symmetric structures are often used to explain eigen‑value spectra of the graph Laplacian and to motivate concepts like graph Fourier modes (see the spectral‑convolution discussion).  
- **The Geometric Deep Learning project page (f)** is cited as the community hub where the seminal "Geometric Deep Learning on Graphs and Manifolds" tutorial by Michael Bronstein (SIAM 2018) originated; the lecture repeatedly references this work when introducing the manifold‑view of graphs.  
- **The schematic GNN architecture (g)** visualises the message‑passing paradigm that underlies both spatial and spectral graph convolutions, reinforcing the connection between the adjacency‑based aggregation described in the GraphSAGE algorithm and the Laplacian‑based spectral filter.  
- **The Gaussian‑convolution animation (h)** provides an intuitive analogue of heat diffusion on a continuous domain, which the lecturer later discretises via the graph Laplacian to explain graph‑based diffusion.  
- **The coronary‑artery mesh (i)** exemplifies a biomedical application where vessels are modelled as graph structures; this is precisely the setting used in Wolterink et al. (2019) for graph‑convolutional segmentation of cardiac CT angiography.  
- **The radiological case image (j)** is employed in the final part of the lecture to showcase how graph‑based segmentation can be combined with classical radiology pipelines, illustrating the practical relevance of the theoretical material.

These contextual notes were mentioned explicitly in the transcript (e.g., "There are image references that I'll put into the description of this video") and serve to bridge the visual assets with the mathematical concepts presented throughout the lecture.

*Prepared by A. Maier, V. Christlein, K. Breininger, and S. Vesal; Graph Deep Learning, 3 July 2020.*

## Lecture Notes Sources

These integrated lecture notes were transcribed from voice recordings of the lecture (FAU LME). Follow the links for the original blog posts:

- [Graph Deep Learning Part 1](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-graph-deep-learning-part-1/)
- [Graph Deep Learning Part 2](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-graph-deep-learning-part-2/)
