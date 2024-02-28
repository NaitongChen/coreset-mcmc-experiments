"""
Slice sampler [Neal, 2003](https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full).
"""

@with_kw struct SliceSamplerMD <: AbstractKernel
    w::Float64 = 10.0 # initial slice width
    p::UInt = 20 # maximum number of doublings
    n_passes::UInt = 3 # number of passes through variables per step
end

function step!(kernel::SliceSamplerMD, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    update_estimator!(state, model, cv, nothing, nothing, nothing)
    cached_lp = log_potential(state, model, cv)
    cached_lp = slice_sample!(kernel, state, model, cv, cached_lp, copy(state.θ))
    return state.θ
end

# function slice_sample!(kernel::SliceSamplerMD, state::AbstractState, 
#                         model::AbstractModel, cv::AbstractLogProbEstimator, pointer, cached_lp, ::Bool)
#     if pointer[] == Bool(0)
#         lp0 = cached_lp
#         pointer[] = Bool(1)
#         lp1 = log_potential(state, model, cv) 
#     else
#         lp1 = cached_lp
#         pointer[] = Bool(0)
#         lp0 = log_potential(state, model, cv)
#     end

#     if rand(state.rng) < exp(lp0-lp1)/(1.0 + exp(lp0-lp1))
#         pointer[] = Bool(0)
#         return lp0
#     else
#         pointer[] = Bool(1)
#         return lp1
#     end
# end

function slice_sample!(kernel::SliceSamplerMD, state::AbstractState, 
                        model::AbstractModel, cv::AbstractLogProbEstimator, cached_lp, θ::Any)
    z = cached_lp - rand(state.rng, Exponential(1.0))
    search_dir = randn(state.rng, length(θ))
    search_dir = search_dir / norm(search_dir)
    L, R, lp_L, lp_R = slice_double(kernel, state, model, cv, z, θ, search_dir)
    cached_lp = slice_shrink!(kernel, state, model, cv, z, L, R, lp_L, lp_R, θ, search_dir)
    return cached_lp
end

function slice_double(kernel::SliceSamplerMD, state::AbstractState,
                        model::AbstractModel, cv::AbstractLogProbEstimator, z, θ, search_dir)
    old_position = θ # store old position (trick to avoid memory allocation)
    L, R = initialize_slice_endpoints(kernel.w, θ, state.rng, search_dir) # dispatch on either float or int
    K = kernel.p
    
    state.θ = L
    potent_L = log_potential(state, model, cv) # store the log potential
    state.θ = R
    potent_R = log_potential(state, model, cv)
    while (K > 0) && ((z < potent_L) || (z < potent_R))
        V = rand(state.rng)        
        if V <= 0.5
            L = L - (R - L)
            state.θ = L
            potent_L = log_potential(state, model, cv) # store the new log potential
        else
            R = R + (R - L)
            state.θ = R
            potent_R = log_potential(state,model, cv)
        end
        K = K - 1
    end
    state.θ = old_position # return the state back to where it was before
    return (L, R, potent_L, potent_R)
end

function initialize_slice_endpoints(width, θ, rng::AbstractRNG, search_dir)
    L = θ - width * rand(rng) * search_dir
    R = L + width * search_dir
    return (L, R)
end

# function initialize_slice_endpoints(width, pointer, rng::AbstractRNG, ::Type{T}) where T <: Integer
#     width = convert(T, ceil(width))
#     L = pointer[] - rand(rng, 0:width)
#     R = L + width 
#     return (L, R)
# end

function slice_shrink!(kernel::SliceSamplerMD, state::AbstractState, 
                        model::AbstractModel, cv::AbstractLogProbEstimator, z, L, R, lp_L, lp_R, θ, search_dir)
    old_position = θ
    Lbar = L
    Rbar = R

    while true
        new_position = draw_new_position(Lbar, Rbar, state.rng)
        state.θ = new_position 
        new_lp = log_potential(state, model, cv)
        consider = z < new_lp
        state.θ = old_position
        if consider && slice_accept(kernel, state, model, cv, new_position, z, L, R, lp_L, lp_R, θ, search_dir)
            state.θ = new_position
            return new_lp
        end
        if norm(new_position - Lbar) <= norm(old_position - Lbar)
            Lbar = new_position
        else
            Rbar = new_position
        end
    end
    # code should never get here
    error()
    return 0.0 
end

draw_new_position(L, R, rng::AbstractRNG) = L + rand(rng) * (R-L)
# draw_new_position(L, R, rng::AbstractRNG, ::Type{T}) where T <: Integer = rand(rng, L:R)

function slice_accept(kernel::SliceSamplerMD, state::AbstractState, model::AbstractModel, 
                        cv::AbstractLogProbEstimator, new_position, z, L, R, lp_L, lp_R, θ, search_dir)
    old_position = θ
    Lhat = L
    Rhat = R
    Rstale = false
    Lstale = false
    
    D = false
    while norm(Rhat - Lhat) > 1.1 * kernel.w
        M = (Lhat + Rhat)/2.0
        if (( sum((old_position - M) ./ search_dir) < 0 ) && ( sum((new_position - M) ./ search_dir) >= 0 )) || 
            (( sum((old_position - M) ./ search_dir) >= 0 ) && ( sum((new_position - M) ./ search_dir) < 0 ))
            D = true
        end
        
        if sum((new_position - M) ./ search_dir) < 0
            Rhat = M
            Rstale = true
        else
            Lhat = M
            Lstale = true
        end
        
        if D 
            if Rstale
                state.θ = Rhat
                lp_R = log_potential(state, model, cv)
                Rstale = false
            end
            if Lstale
                state.θ = Lhat
                lp_L = log_potential(state, model, cv)
                Lstale = false
            end
            if ((z >= lp_L) && (z >= lp_R))
                state.θ = old_position 
                return false
            end
        end
    end
    state.θ = old_position
    return true
end