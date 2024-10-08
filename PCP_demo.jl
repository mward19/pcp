using Images
using Random
using LinearAlgebra
using TestImages

using VideoIO
using FileIO
using Images

using ProgressMeter

using Plots

include("PCP.jl")
using .PCP_by_ADMM


""" Convert a video file into a vector of grayscale image frame matrices. """
function video_to_frames(filename::String)
    video_frames = load(filename)
    gray_frames = Vector{Matrix}(undef, length(video_frames))

    # Downsample frames to reduce size
    image_size = size(video_frames[begin])
    aspect_ratio = image_size[2] / image_size[1]
    resize_size = (250, floor(Int, aspect_ratio*250))

    @showprogress desc="Processing video..." for i in eachindex(video_frames)
        gray_frames[i] = Gray.(imresize(video_frames[i], resize_size))
    end
    
    return gray_frames
end

""" Convert a vector of image frames into a gif file. """
function gif_from_frames(frames, fps=20)
    # Convert frames into 3D array
    array = cat(frames...; dims=3)
    save("temp/demo.gif", array; fps=fps)
end

""" 
Vectorizes each image in `images`, returning a matrix in which each column is an
image.
"""
function vectorize_images(images::AbstractVector)
    n_images = length(images)
    img_dims = size(images[begin])

    # The target matrix 𝐘 has a full image in each column. Vectorize the images
    # and horizontally concatenate to construct the target matrix 𝐘.
    𝐘 = hcat([Float64.(vec(image)) for image in images]...)
    
    return 𝐘
end

""" The inverse of `vectorize_images`. """
function devectorize_images(image_vectors::AbstractMatrix, original_size)
    return [reshape(image, original_size) for image in eachcol(image_vectors)]
end

""" Rescales an array between 0 and 1. """
function rescale(array)
    min_val = findmin(array)[1]
    max_val = findmax(array)[1]
    return (array .- min_val) ./ (max_val - min_val)
end

""" Forces the number `x` between floor and ceil. """
function force_between(x; floor=0, ceil=1)
    if x < floor
        return floor
    elseif x > ceil
        return ceil
    else
        return x
    end
end

"""
Perform a demo of PCP on a video file, separating the static parts from the
changing parts.
"""
function demo(filename::String)
    frames = video_to_frames(filename)
    image_size = size(frames[begin])
    n_images = length(frames)
    𝐘 = vectorize_images(frames)

    # Hyperparameters for PCP (Principal Component Pursuit). 
    λ = 1/√max(*(image_size...), n_images)
    μ = 1/10
    # Decompose 𝐘 into a low-rank component (𝐋) and a sparse component (𝐒)
    # with PCP.
    𝐋, 𝐒 = PCP(𝐘, λ, μ; maxiter=10, ϵ=1)
    # Recover images from 𝐋 and 𝐒.
    𝐋_images = devectorize_images(force_between.(𝐋), image_size)
    𝐒_images = devectorize_images(rescale(𝐒), image_size)

    # Concatenate result images for easy viewing
    display_frames = [vcat(Y, L, S) for (Y, L, S) in zip(frames, 𝐋_images, 𝐒_images)]
    gif_from_frames(display_frames)

    return 𝐘, 𝐋, 𝐒
end



