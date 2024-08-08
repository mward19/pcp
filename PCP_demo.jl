using Images
using Random
using LinearAlgebra
using TestImages

using VideoIO
using FileIO
using Images

using Plots

include("PCP.jl")
using .PCP_by_ADMM


""" Convert a video file into a vector of grayscale image frame matrices. """
function video_to_frames(filename::String)
    video_frames = load(filename)
    gray_frames = Vector{Matrix}(undef, length(video_frames))
    for i in eachindex(video_frames)
        gray_frames[i] = Gray.(video_frames[i])
    end
    
    return gray_frames
end

""" 
Vectorizes each image in `images`, returning a matrix in which each column is an image.
"""
function vectorize_images(images::AbstractVector)
    n_images = length(images)
    img_dims = size(images[begin])

    # The target matrix 𝐘 has a full image in each column.
    # Vectorize the images and horizontally concatenate to construct the target matrix 𝐘.
    𝐘 = hcat([Float64.(vec(image)) for image in images]...)
    
    return 𝐘
end

""" Rescales an array between 0 and 1. """
function rescale(array)
    min_val = findmin(array)[1]
    max_val = findmax(array)[1]
    return (array .- min_val) ./ (max_val - min_val)
end


""" The inverse of `vectorize_images`. """
function devectorize_images(image_vectors::AbstractMatrix, original_size)
    return [reshape(image, original_size) for image in eachcol(image_vectors)]
end

""" (This can be refined) """
function gif_from_frames(frames, fps=20)
    # Convert frames into 3D array
    array = cat(frames...; dims=3) |> rescale
    save("demo.gif", array; fps=fps)
end


function demo(filename::String)
    frames = video_to_frames(filename)
    image_size = size(frames[begin])
    n_images = length(frames)
    𝐘 = vectorize_images(frames)

    # Hyperparameters for PCP (Principal Component Pursuit). 
    λ = 1/√max(*(image_size...), n_images)
    μ = 1/10
    # Decompose 𝐘 into a low-rank component (𝐋) and a sparse component (𝐒) with PCP.
    𝐋, 𝐒 = PCP(𝐘, λ, μ; maxiter=10)
    # Recover images from 𝐋 and 𝐒.
    𝐋_images = devectorize_images(𝐋, image_size)
    𝐒_images = devectorize_images(𝐒, image_size)

    # Concatenate result images for easy viewing
    display_frames = [vcat(Y, L, S) for (Y, L, S) in zip(frames, 𝐋_images, 𝐒_images)]
    gif_from_frames(display_frames)
end



