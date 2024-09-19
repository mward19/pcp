# Brief summary
I was looking for ways to work with and extract information from noisy 3-dimensional images of bacteria called tomograms. I read a book about ways of making the most of messy data using linear algebra and optimization<sup>1</sup>. In the process, I learned about Principal Component Pursuit. 

Here I apply Principle Component Pursuit to a video I shot in my lab. Using this method, I am able to extract the background and foreground of a video, using nothing more than linear algebra and a simple convex optimization problem.

![pcp demo](./demo.gif)

 - The top video is the original, which I shot of myself in my lab.
 - The middle video is the "background" component of the video.
 - The bottom video is the "foreground" (moving) component of the video.

I have yet to successfully apply this method to tomograms, but my advisor and I thought it was fascinating, so we feel like we succeeded anyway! If you would like to learn more details, read the longer summary below.

# Longer summary
## What is this?
I work in a biophysics research group, in which we perform analysis on 3-dimensional images of bacteria called tomograms. These images are often very noisy<sup>2</sup>, so my advisor and I started reading a textbook<sup>1</sup> to learn more ways to deal with the dimensionality and messiness of our data. 

One thing we read about was a way to separate an almost-low-rank matrix into a sparse component and a low-rank component using convex optimization. The authors claimed that it could be applied to video, but they did not provide code, so I decided to test it myself. I implemented the method, called Principal Component Pursuit, in Julia.

Although we have yet to find a way to apply Principal Component Analysis to tomograms, we thought the concept was fascinating.

## What am I seeing?
I recorded the top video in my lab. I wanted a video with a mostly static background and something moving in the foreground. 

Using the method, which I will describe next, I separate the video into a static component and a moving component. The static component is the middle video. The moving component is the bottom video. In other words, the bottom video added to the middle video yields the top video.

## How does it work?
Each frame of a grayscale video can be thought of as a matrix of grayscale values. For each frame, I take that matrix and flatten it into a vector. Thus, each frame of the video can be represented as a long vector with as many elements as there are pixels in a frame. I will call this vector a "frame vector".

By representing the frames as vectors, the entire video can itself be represented as a matrix. This is done by stacking all of the frame vectors side by side into a huge matrix, with as many columns as there are frames in the video. I will call this matrix a "video matrix".

In a video with a mostly static background and something moving in the foreground, the video matrix is *almost* low rank, since the frame vectors are mostly the same (since most of the pixels don't change). But it isn't, because of the movement in the foreground. Nevertheless, we can find a video matrix that is nearly equal to the original video matrix but is in fact low rank. In simpler terms, we can extract the background of the video.

Let $\bf{Y}$ be the original video matrix. We want a video matrix $\bf{L}$ that is nearly equal to $\bf{Y}$, differing only by a sparse (meaning most of the elements are 0) matrix $\bf{L}$. In other words, we want to find $\bf{L}$ and $\bf{S}$ such that $\bf{Y} = \bf{L} + \bf{S}$, while minimizing $\text{rank}(\bf{L})$ and $\Vert\bf{S}\Vert_0$ (where $\Vert\cdot\Vert_0$ gives the number of non-zero elements of its input, which technically is not a proper mathematical norm). 

One could set this up as an optimization problem

$$
\begin{aligned}
    \text{minimize}\hspace{5mm}&\text{rank}(\bf{L}) + \lambda \Vert\bf{S}\Vert_0 \\
    \text{subject to}\hspace{5mm}&\bf{Y} = \bf{L} + \bf{S}
\end{aligned}
$$

for some tuning parameter $\lambda \in \mathbb{R}$, but the objective is not convex, making this very difficult to solve.

Rather, we set up the problem using convex surrogate norms:

$$
\begin{aligned}
    \text{minimize}\hspace{5mm}&\Vert\bf{L}\Vert_* + \lambda \Vert\bf{S}\Vert_1 \\
    \text{subject to}\hspace{5mm}&\bf{Y} = \bf{L} + \bf{S},
\end{aligned}
$$

where $\Vert \cdot \Vert_*$ is the *nuclear norm*, meaning, the sum of the singular values of the input matrix, and $\Vert \cdot \Vert_1$ is the standard matrix 1-norm (the maximum column sum). (See "Note: Why these norms?" below.)

This new problem is convex! It is easy to solve with off-the-shelf convex optimizers. I have opted to implement the optimizer myself, but other libraries like CVXPY (in Python) or Convex.jl (for Julia) should work fine.

By solving

$$
\begin{aligned}
    \text{minimize}\hspace{5mm}&\Vert\bf{L}\Vert_* + \lambda \Vert\bf{S}\Vert_1 \\
    \text{subject to}\hspace{5mm}&\bf{Y} = \bf{L} + \bf{S},
\end{aligned}
$$

we find video matrices $\bf{L}$ and $\bf{S}$ that, for all intents and purposes, separate the original video matrix $\bf{Y}$ into "background" and "foreground" components respectively. Problem solved!

### Note: Why these norms?
First, we want to minimize the rank of $\bf{L}$. When the rank of $\bf{L}$ is minimized, we hope that most of its singular values are zero. Perhaps this will shed some intuition on why the nuclear norm makes sense here.

Second, we want to minimize the number of nonzero elements of $\bf{S}$ (what I called $\Vert \cdot \Vert_0$ above). When the number of nonzero elements of $\bf{S}$ is minimized, we would hope that the maximum column sum of $\bf{S}$ is quite small. Hopefully this clarifies why the 1-norm is a reasonable choice.

For more formal justification for these choices of norm, consult Wright and Ma's textbook<sup>1</sup>.

## References

1: [Wright, J., & Ma, Y. (2022). *High-Dimensional Data Analysis with Low-Dimensional Models: Principles, Computation, and Applications*. Cambridge University Press.](https://book-wright-ma.github.io/)

2: [CryoET Data Portal](https://cryoetdataportal.czscience.com/browse-data/datasets)



