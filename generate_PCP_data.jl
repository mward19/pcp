using Images
using Random
using LinearAlgebra
using TestImages

include("PCP.jl")
using .PCP_by_ADMM

function make_flowers(background_img, poss_vals=LinRange(0, 1, 1000), N_flowers=8)
    img_dims = size(background_img)
    possible_petals = collect(4:2:12)
    petals = [rand(possible_petals) for flower in 1:N_flowers]
    
    centers = [img_dims .* [rand(), rand()] for flower in 1:N_flowers]
    
    possible_radii = LinRange(min(img_dims...) / 100, min(img_dims...) / 20, 1000)
    radii = [rand(possible_radii) for flower in 1:N_flowers]

    possible_puffiness_scales = LinRange(1, 5, 500)
    puffiness = [r * rand(possible_puffiness_scales) for r in radii]

    # Functions defining the radius around the respective centers
    rad_funcs = [θ -> r*sin(p*θ)+puff 
                        for (p, r, puff) in zip(petals, radii, puffiness)]
    
    # θ = arctan(Δy / Δx)
    in_criteria = [(coord) -> 
                     norm(coord - center) <= 
                     rf(atan((coord[2] - center[2]) / (coord[1] - center[1])))
                     for (center, rf) in zip(centers, rad_funcs)]
                     
    image = Float64.(background_img)
    flower_vals = [rand(poss_vals) for flower in 1:N_flowers]
    for x in 1:img_dims[1], y in 1:img_dims[2]
        for (crit, val) in zip(in_criteria, flower_vals)
            if crit([x, y])   image[x, y] = val   end
        end
    end

    return image
end

# source_image is what we hope 𝐘 resembles
source_image = Gray.(imresize(testimage("barbara_gray_512"), (256, 256)))

# Generate noise clumps
poss_num_flowers = collect(3:6)
N_noisy = 10
noisy_images = [make_flowers(
                    source_image, 
                    LinRange(0, 1, 1000),
                    rand(poss_num_flowers)
                ) for i in 1:N_noisy]

# Generate 𝐘
n1 = *(size(source_image)...)
n2 = N_noisy
𝐘 = zeros(n1, n2)
for (index, image) in enumerate(noisy_images)
    𝐘[:, index] = vec(image)
end

λ = 1/√max(n1, n2)
μ = 1/10
𝐋, 𝐒 = PCP(𝐘, λ, μ)

function devec(v⃗)
    rows, cols = size(source_image)
    if rows * cols != n1   
        raise(DimensionMismatch())
    end

    return reshape(v⃗, size(source_image))
end

# Should be close to source_image
Gray.([
    devec(𝐋[:,1]) devec(𝐋[:,2]) devec(𝐋[:,3])
    devec(𝐋[:,4]) devec(𝐋[:,5]) devec(𝐋[:,6])
    devec(𝐋[:,7]) devec(𝐋[:,8]) devec(𝐋[:,9])
]) |> display

# Should be sparse
Gray.([
    devec(𝐒[:,1]) devec(𝐒[:,2]) devec(𝐒[:,3])
    devec(𝐒[:,4]) devec(𝐒[:,5]) devec(𝐒[:,6])
    devec(𝐒[:,7]) devec(𝐒[:,8]) devec(𝐒[:,9])
]) |> display

# Should be close to noisy_images
Gray.([
    devec(𝐋[:,1] + 𝐒[:,1]) devec(𝐋[:,2] + 𝐒[:,2]) devec(𝐋[:,3] + 𝐒[:,3])
    devec(𝐋[:,4] + 𝐒[:,4]) devec(𝐋[:,5] + 𝐒[:,5]) devec(𝐋[:,6] + 𝐒[:,6])
    devec(𝐋[:,7] + 𝐒[:,7]) devec(𝐋[:,8] + 𝐒[:,8]) devec(𝐋[:,9] + 𝐒[:,9])
]) |> display

# noisy_images
Gray.([
    noisy_images[1] noisy_images[2] noisy_images[3]
    noisy_images[4] noisy_images[5] noisy_images[6]
    noisy_images[7] noisy_images[8] noisy_images[9]
]) |> display