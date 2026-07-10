---
course: deep-learning
title: Introduction
source: deep-learning/slides/0_Intro/0_Introduction.tex
---
# Introduction

### Who are we? - Lab Members
*Figure (unresolved): `people/Andreas2.jpg`.*

			Andreas  
Maier  

			
			

*Figure (unresolved): `people/soroosh.jpg`.*

   

			Soroosh    
 Tayebi Arasteh   

		
		
			
			

*Figure (unresolved): `people/changliu.jpg`.*

   

			Chang   
 Liu   

			
			
			

*Figure (unresolved): `people/lindaS.jpg`.*

   

			Linda-Sophie   
 Schneider  

		
		
		    
			

*Figure (unresolved): `people/alexander_barnhill.jpg`.*

   

			Alexander   
 Barnhill  

			
			

*Figure (unresolved): `people/zijin.jpg`.*

   

			Zijin   
 Yang  

	
### Deep Learning -- Buzzwords
This is a mind map centered on "Deep Learning," illustrating its key relationships and context. The central node, "Deep Learning," branches to several categories: "Neural Network" (a core component), "Feature" (implying automatic feature extraction), and "Tasks" (which lists "Classification," "Segmentation," "Regression," and "Generation" as primary applications, with an ellipsis suggesting more). It also connects to "Supervised vs. unsupervised," indicating learning paradigms. On the right side, "Deep Learning" links to broader fields: "Artificial Intelligence (AI)," "Machine Learning," "Representation Learning," and "Big Data," highlighting its place within these domains. The diagram conceptually illustrates that deep learning is a subfield of machine learning that uses neural networks to automatically learn representations from data, enabling diverse tasks and being enabled by big data and AI.

{ 
	
	
}

## Motivation

### NVIDIA Stock Market
The figure is a line chart depicting the price of NVIDIA stock from 2018 to 2024. The vertical axis represents the stock price in arbitrary units, ranging from 0 to 1,000, while the horizontal axis shows years. A single green line traces the stock’s price movement, which remains near zero until 2020, then exhibits gradual growth through 2022, followed by a period of volatility and a steep upward trend beginning in late 2023, culminating near 850 in early 2024. The chart visually illustrates the dramatic acceleration in NVIDIA’s stock value over the final two years, highlighting its market performance during this period.

There is a considerable dip around 2019 / the end of 2018. Here, you can see that it’s not only deep learning that is driving the market share value of Nvidia. There’s also another very interesting thing happening at the same time and that is Bitcoin mining. The Bitcoin value really decreased in at this period of time and also the Nvidia stock market value went down. So it’s partially also associated to the Bitcoin, but you can see that the value is going up again because there’s a huge demand for compute power in deep learning.

### The Big Bang of Deep Learning & ILSVRC
This section merges the historical context and dataset details. Four example images from the ImageNet dataset are shown, each labeled with the object it depicts: a red mite on a textured surface, a large container ship sailing on water, a person riding a motor scooter in an urban setting at night, and a leopard resting on a tree branch. These images illustrate the diverse range of visual content included in ImageNet, which contains approximately 14 million images labeled into around 20,000 synonym sets. The dataset was used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which employed approximately 1,000 distinct classes.

Before the image net challenge, classifying into a thousand classes was essentially deemed completely impossible. The images that are used here have been downloaded from the internet and they have a single label per image. So, this is a really huge database that allows us to assign categories large numbers of categories into individual images. In 2012, there was a big step ahead with the win of the Alex Network. AlexNet really halved the error rate. So, if we look at the different error rates that we have obtained over the scope of the image net challenge, you can see that we started off with error rates about 25% regarding the Top-5 error. So in 2011 and the years before, we were approximately in the ballpark of 25 percent and you could see this stalling over the last couple of years. Well in 2012, the first convolutional neural network (CNN) here was introduced and the CNN almost halved the error rate. Now that was quite a big surprise because nobody else could do it at the time. You can see that not only in 2012 there has been progressing, but in 2013 and so on the error rates more and more decreased until we essentially reached a level where they are approximately in the same range as humans.

The neural networks there reported the first results where people have been claiming that there has been superhuman performance. So, is it really superhuman performance? Well, not so many humans have really evaluated the entire test set. So, you could actually say that superhuman performance should be a super Karpathy-an performance because he actually went through the entire test dataset.

There are a couple of problems with ImageNet. The above the top row are probably rather easy cases, but if you look at the bottom row, there’s a couple of really difficult ones as well. So, where you have only parts of the image shown or in particular if you look at the cherry that also shows a dog, it’s very hard to differentiate those images. And of course, this is a problem if you only have a single label per image. maybe a single label is just not enough to describe an entire image.

- $\approx 14$\,mio.\ images, labeled into $\approx 20.000$ **synonym sets**
		- **I**mageNet **L**arge **S**cale **V**isual **R**ecognition **C**hallenge using $\approx 1000$ classes
		- Images downloaded from the Internet, **single** label per image
		- **2012: Breakthrough** by Krizhevsky et al. [Krizhevsky et al. (2012) [@Krizhevsky12]]
	
Krizhevsky et al.\ 2012

### Deep Learning Users
The image displays a collage of logos representing major corporations and technology entities that are prominent users or adopters of deep learning technologies. The logos are arranged in a grid-like pattern without a specific structural hierarchy. Observable entities include Netflix, Apple, Google, Microsoft, IBM, Samsung, Siemens, GE, Daimler, Xerox, Lunit, DeepMind, and others. The visual cue distinguishing the components is their distinct brand logos, each with unique typography, color schemes, and graphical elements. 

You can see that the Netflix challenge has been solved partially with deep learning. This was a 1 million dollar challenge to actually build a recommendation system that will recommend movies that you actually like. And you can see that healthcare is going in there: Siemens and GE. But also car manufacturers such as Daimler and many other carmakers are going there because there’s a huge trend towards autonomous driving.

### Playing Go
- **1997**: Deep Blue beats Garry Kasparov
			- **Go** as a next challenge
			- Large branching factor
			- **2016**: AlphaGo [Silver et al. (2016) [@Silver2016]] beats a professional
			- **2017**: AlphaGoZero [Silver et al. (2017) [@AlphaGoZero]] surpasses every human in Go by self-play
			- **2017**: AlphaZero [Silver et al. (2017) [@AlphaZero]] generalizes to a number of other board games
			- **2019**: AlphaStar beats professional StarCraft players
		

Solving chess is a little easier because it has not so complex moves. The way how they actually solve the game is that they had a dictionary of starting moves. Then, they essentially did a brute-force search over the entire game in the mid part of the game and towards the end of the game again they were using a dictionary. Now Go is a much harder challenge because in every move of the game you can place a stone on every part of the board. This means that if you really want to do an exhaustive search and look for all the different opportunities that you have in the game, only after a couple of moves the actual number of moves is exploding due to the large branching factor. At the present compute-power, we’re not able to brute-force the entire game. However, in 2016 AlphaGo, a system created by a Deep Mind really beat a professional Go player. In 2017, AlphaGo Zero even surpassed every human and only by self-playing. Then shortly later, AlphaGo Zero generalizes to a number of other board games. They even managed to go towards even other games that are not like the typical board games. In AlphaStar , they beat professional Starcraft players. So, it’s really a very interesting technology to look at.

A wooden Go board with a grid of intersecting lines, populated with black and white stones arranged in a mid-game configuration. The board is elevated on four small, rounded feet and rests on a green tatami mat. Two dark, circular stone containers (go bowls) are partially visible in the foreground. The image illustrates the physical setup of the game Go, a two-player strategy game played on a 19x19 grid, where players place stones to capture territory and surround the opponent’s stones. The visible stone placement suggests an ongoing match, with both players having established territories and potential threats. This figure serves as a visual anchor for the lecture’s discussion of Go’s complexity and the AI breakthroughs in mastering it, such as AlphaGo and AlphaZero.

			{
			

The image displays the official logo for the video game *StarCraft II*. The title "STAR CRAFT" is rendered in a stylized, metallic, three-dimensional font with sharp, angular edges and a blue and silver color scheme. Behind the text, a glowing blue lightning bolt effect emanates from the central "II" symbol, which is designed to resemble a futuristic shield or energy core. The background is a cosmic scene featuring a dark nebula with swirling

## Lecture Notes Sources

These integrated lecture notes were transcribed from voice recordings of the lecture (FAU LME). Follow the links for the original blog posts:

- [Introduction Part 1](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-introduction-part-1/)
- [Introduction Part 2](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-introduction-part-2/)
- [Introduction Part 3](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-introduction-part-3/)
- [Introduction Part 4](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-introduction-part-4/)
- [Introduction Part 5](https://lme.tf.fau.de/lecture-notes/lecture-notes-dl/lecture-notes-in-deep-learning-introduction-part-5/)
