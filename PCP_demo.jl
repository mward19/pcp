using Images
using Random
using LinearAlgebra
using TestImages
using VideoIO
using FileIO
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

""" Converts an MP4 video to a GIF using FFmpeg. """
function mp4_to_gif(input_mp4::String, output_gif::String)
    palette="/tmp/palette.png"

    # Generate a color palette from the video
    run(`ffmpeg -y -i $input_mp4 -vf "palettegen" $palette`)

    # Convert video to GIF using the generated palette
    run(`ffmpeg -y -i $input_mp4 -i $palette -filter_complex "paletteuse" $output_gif`)
end

""" Convert a vector of image frames into a gif file with a specified save path. """
function gif_from_frames(frames; save_path="output.gif", fps=20)
    # Convert frames into 3D array
    array = cat(frames...; dims=3)
    save(save_path, array; fps=fps)
end

""" Vectorizes each image in `images`, returning a matrix where each column is an image. """
function vectorize_images(images::AbstractVector)
    n_images = length(images)
    img_dims = size(images[begin])
    𝐘 = hcat([Float64.(vec(image)) for image in images]...)  # Vectorize images and concatenate
    return 𝐘
end

""" The inverse of `vectorize_images`. """
function devectorize_images(image_vectors::AbstractMatrix, original_size)
    return [reshape(image, original_size) for image in eachcol(image_vectors)]
end

""" Rescales an array between 0 and 1. """
function rescale(array)
    min_val, max_val = extrema(array)
    return (array .- min_val) ./ (max_val - min_val)
end

""" Forces the number `x` between floor and ceil. """
function force_between(x; floor=0, ceil=1)
    return clamp(x, floor, ceil)
end

""" Creates an animated plot showing movement over time. """
function animate_movement(𝐒::Matrix; fps=20, save_path="movement.mp4")
    frame_indices = 1:size(𝐒, 2)
    movement_metric = [norm(frame, 0) for frame in eachcol(𝐒)]

    anim = @animate for i in eachindex(frame_indices)
        plot(
            frame_indices, movement_metric;
            title="Nonzero elements in the sparse\n (foreground) component of video",
            label="Nonzero elements",
            lw=2
        )
        scatter!([frame_indices[i]], [movement_metric[i]], color=:red, markersize=6, label="Current position")
    end
    
    if endswith(save_path, ".mp4")
        mp4(anim, save_path, fps=fps)
    else
        gif(anim, save_path, fps=fps)
    end
end


""" Concatenates multiple MP4 videos vertically. """
function concatenate_videos_vertically(video1::String, video2::String, output::String="output.mp4")
    # Get width of video1
    width_cmd = `ffprobe -v error -select_streams v:0 -show_entries stream=width \
                  -of default=noprint_wrappers=1:nokey=1 $video1`
    width = parse(Int, chomp(read(width_cmd, String)))
    
    # Get frame rate of video1 (as a fraction like "30000/1001")
    fps_cmd = `ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
                -of default=noprint_wrappers=1:nokey=1 $video1`
    fps_str = chomp(read(fps_cmd, String))
    num, den = split(fps_str, "/")
    fps = parse(Float64, num) / parse(Float64, den)
    
    # Build filter: scale video2 to same width and adjust its fps, then stack vertically
    filter_str = "[1:v]scale=$width:-1,fps=$fps[v2];[0:v][v2]vstack=inputs=2[v]"
    
    # Run ffmpeg command
    ffmpeg_cmd = `ffmpeg -y -i $video1 -i $video2 -filter_complex $filter_str \
                   -map "[v]" -c:v libx264 -crf 23 -preset medium $output`
    run(ffmpeg_cmd)
end

function get_framerate(filepath::String)
    cmd = `ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 $filepath`
    fps_str = chomp(read(cmd, String))
    num, den = split(fps_str, "/")
    return parse(Float64, num) / parse(Float64, den)
end


"""
A demo of PCP on a video file, separating the static parts from the changing
parts and showing when the algorithm sees the most movement.
"""
function motion_detection_demo(filename::String, title::String)
    frames = video_to_frames(filename)
    framerate = get_framerate(filename)
    image_size = size(frames[begin])
    n_images = length(frames)
    𝐘 = vectorize_images(frames)

    # Hyperparameters for PCP (Principal Component Pursuit). 
    λ = 1/√max(*(image_size...), n_images)
    μ = 1/100
    # Decompose 𝐘 into a low-rank component (𝐋) and a sparse component (𝐒)
    # with PCP.
    𝐋, 𝐒 = PCP(𝐘, λ, μ; maxiter=30, ϵ=1e-2 * max(image_size...))
    
    # Recover images
    𝐋_images = devectorize_images(force_between.(𝐋), image_size)
    𝐒_images = devectorize_images(rescale(𝐒), image_size)

    # Concatenate result images for easy viewing
    display_frames = [vcat(Y, L, S) for (Y, L, S) in zip(frames, 𝐋_images, 𝐒_images)]

    @info "Creating animations..."

    decomposition_path = "$(title) decomposed.gif"
    movement_metric_path = "temp/$(title) movement plot.mp4"

    gif_from_frames(display_frames, save_path=decomposition_path, fps=framerate)
    animate_movement(𝐒, save_path=movement_metric_path, fps=framerate)

    # Concatenate original video with movement metric plot
    concatenate_videos_vertically(filename, movement_metric_path, "$(title) movement demo.mp4")

    return 𝐘, 𝐋, 𝐒
end

𝐘, 𝐋, 𝐒 = motion_detection_demo("walking.mp4", "walking")
