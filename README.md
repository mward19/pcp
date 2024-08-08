# Principal Component Pursuit
This code is an implementation and demo of Principal Component Pursuit, as applied to an image background removal problem. The algorithm is based on section 5.2 of John Wright & Yi Ma's *High-Dimensional Data Analysis with Low-Dimensional Models* [Wright & Ma, 2022].

The algorithm represents a video as a matrix in which each column is a flattened frame of the video. We assume that the video mostly contains a static background, and a few things moving or changing in the foreground. Using convex optimization, we separate the video matrix into a low-rank component and a sparse component -- or in other words, a background component, and a foreground component.

Here are some rudimentary results, using the `demo` function in `PCP_demo.jl` (in this repository). They would probably be better if I tuned the hyperparameters a bit more. The top shows the original video. The middle shows what the algorithm decided is the background. The bottom is the foreground. The bottom video added to the middle video yields the top video.

![pcp demo](./demo.gif)

Wright, J., & Ma, Y. (2022). High-dimensional data analysis with low-dimensional models: Principles, computation, and applications. Cambridge University Press.
