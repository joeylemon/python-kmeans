# python-kmeans

An implementation of the [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) unsupervised machine learning algorithm used to reduce the number of colors required to represent an image.

<p align="center">
<img src="https://user-images.githubusercontent.com/8845512/135511666-b6965ffc-4baf-4d83-9c50-718a4b9db037.png" />
</p>

## Motivation

As students in COSC425: Introduction to Machine Learning at the [University of Tennessee](https://utk.edu/), we were tasked with implementing the K-means algorithm from scratch in Python. We were then to use the algorithm to determine the best set of RGB colors to represent a given image. Finally, we had to analyze the performance of our algorithm by observing how quickly the clusters reached a centroid convergence and the distribution of pixels and their corresponding clusters. We performed our algorithms on images with K values of 4, 16, or 32. We set a max iteration count of 24, and we determined convergence with a max RGB value delta of 1.

## Analysis

For the image above of [Smokey](https://en.wikipedia.org/wiki/Smokey_(mascot)), the mascot of the University of Tennessee, we can plot how quickly a set of K values reach a cluster convergence. We can also observe the distribution of pixels and their corresponding clusters:

<p align="center">
<img src="https://user-images.githubusercontent.com/8845512/135512789-297fdbe2-77c9-4eb6-9936-384cc03f074e.png" />
</p>

As a further step to understanding how our K-means algorithm reaches a conclusion, we can plot the image's RGB values in 3D and observe how the cluster centroids converge as the algorithm iterates. The below figure observes a separate run of the algorithm with K=4 on the image of Smokey:

<p align="center">
<img src="https://user-images.githubusercontent.com/8845512/135518453-caeb6851-e6f4-4c35-8485-5cf1700f301c.jpg" />
</p>
