module PCP_by_ADMM

using LinearAlgebra
using ProgressMeter

export PCP

""" Implements Algorithm 5.1 from section 5.2.1 of Wright & Ma. """
function PCP(𝐘, λ, μ, maxiter=100, ϵ=1e-2)
    # Define necessary functions for the algorithm.
    relu(x) = max(x, 0)
    𝒮(τ, 𝐌) = sign.(𝐌) .* relu.(abs.(𝐌) .- τ)
    function 𝒟(τ, 𝐌)
        F = svd(𝐌)
        𝐔, 𝚺, 𝐕ᵀ = (F.U, F.S, F.Vt)
        return 𝐔 * 𝒮(τ, diagm(𝚺)) * 𝐕ᵀ
    end
    nuclearnorm(𝐌) = sqrt.(tr(𝐌' * 𝐌))
    objective(𝐋, 𝐒) = nuclearnorm(𝐋) + λ*norm(𝐒, 1)

    # Initialize variables
    𝐒 = zeros(size(𝐘))
    𝚲 = zeros(size(𝐘))

    # Save old variables to check convergence condition, and force scope outside loop
    𝐋 = nothing
    𝐋_old = nothing
    𝐒_old = 𝐒

    # Perform Algorithm 5.1
    @showprogress "Iterating..." for iter in 1:maxiter
        𝐋 = 𝒟(1/μ,  𝐘 - 𝐒 - 1/μ * 𝚲)
        𝐒 = 𝒮(λ/μ,  𝐘 - 𝐋 - 1/μ * 𝚲)
        𝚲 = 𝚲 + μ * (𝐋 + 𝐒 - 𝐘)

        # Check for convergence
        curr_obj = objective(𝐋, 𝐒)
        if iter > 1 && abs(curr_obj - objective(𝐋_old, 𝐒_old)) < ϵ
            return 𝐋, 𝐒
        end
        
        𝐋_old, 𝐒_old = (𝐋, 𝐒)
    end

    @info "Failed to converge after $maxiter iterations."
    return 𝐋, 𝐒
end

end # module


