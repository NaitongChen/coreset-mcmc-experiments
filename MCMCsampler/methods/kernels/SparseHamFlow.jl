@with_kw mutable struct SHF <: AbstractKernel
    L::Int64 = 10 # number of leapfrog steps in between refreshment
    R::Int64 = 5 # number of refreshment
    S::Int64 = 100 # warm start and ELBO subsample size
    T::Int64 = 10000 # optimization iteration
    n::Int64 = 1 # sample size for ELBO estimation
    q0::Union{Distribution, Nothing} = nothing
    q0_sample::Function = (rng, dist, n) -> n == 1 ? rand(rng, dist) : [rand(rng, dist) for i=1:n]
    q0_logpdf::Function = (θ, dist) -> logpdf(dist, θ)
    # params to be optimized
    logϵ::Union{AbstractArray, Nothing} = nothing
    logσ2::Union{AbstractArray, Nothing} = nothing
    μ::Union{AbstractArray, Nothing} = nothing
    # ADAM parameters
    α::Float64 = 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    τ::Float64 = 10^-8
    t::Int64 = 0
    m::Union{Nothing, Vector{Float64}} = nothing
    v::Union{Nothing, Vector{Float64}} = nothing
    # loss tracking
    ls::Vector{Float64} = Vector{Float64}(undef, 0)
end

function step!(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator, iter::Int64)
    if iter == 1
        init_estimator!(state, model, cv)
        if isnothing(kernel.q0)
            kernel.q0 = MvNormal(zeros(length(state.θ)), I)
        end
        kernel.logϵ = log.(0.0001 .* ones(length(state.θ)))
        warm_start!(kernel, state, model, cv)
        kernel.m = zeros(cv.N + length(kernel.logϵ) + length(kernel.μ) + length(kernel.logσ2))
        kernel.v = zeros(cv.N + length(kernel.logϵ) + length(kernel.μ) + length(kernel.logσ2))
        @showprogress for i=1:kernel.T
            g = est_gradient(kernel, state, model, cv)
            kernel.t += 1
            @unpack α, β1, β2, τ, t = kernel
            kernel.m .= β1 * kernel.m + (1-β1)*g
            kernel.v .= β2 * kernel.v + (1-β2)*g.^2
            m̂ = kernel.m/(1-β1^t)
            v̂ = kernel.v/(1-β2^t)
            step = α* m̂./(sqrt.(v̂) .+ τ)
            cv.weights = exp.(log.(cv.weights) - step[1:cv.N])
            kernel.logϵ -= step[cv.N+1:cv.N+length(state.θ)]
            kernel.μ -= step[cv.N+length(state.θ)+1:cv.N+length(state.θ)+kernel.R*length(state.θ)]
            kernel.logσ2 -= step[cv.N+length(state.θ)+kernel.R*length(state.θ)+1:end]
            push!(kernel.ls, est_loss(kernel, state, model, cv))
        end
    end
    state.θ, _, _ = flow(kernel, state, model, cv)
end

function flow(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    θ0 = kernel.q0_sample(state.rng, kernel.q0, 1)
    r0 = randn(state.rng, length(state.θ))
    μr = reshape(kernel.μ, (kernel.R, length(state.θ)))
    logσ2r = reshape(kernel.logσ2, (kernel.R, length(state.θ)))
    return flow(kernel, state, model, cv, θ0, r0, μr, logσ2r)
end

function flow(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator,
                    θ0::AbstractArray, r0::AbstractArray, μr::AbstractArray, logσ2r::AbstractArray)
    θ_original = copy(state.θ)
    r, θ = deepcopy(r0), deepcopy(θ0)
    logdet = 0.
    for i=1:(kernel.L*kernel.R)
        state.θ = deepcopy(θ)
        θ, r = leapfrog!(kernel, state, r, model, cv)
        if i % kernel.L == 0
            j = Int(i/kernel.L)
            r = diagm(vec(1. ./ exp.(logσ2r[j,:]))) * (r .- μr[j,:])
            logdet -= sum(logσ2r[j,:])
        end
    end
    θf, rf = copy(state.θ), deepcopy(r)
    state.θ = θ_original # reset θ
    return θf, rf, logdet
end

function warm_start!(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    @info "warm start"
    kernel.logσ2, kernel.μ = zeros(kernel.R, length(state.θ)), zeros(kernel.R, length(state.θ))
    θs = kernel.q0_sample(state.rng, kernel.q0, kernel.S)
    rs = [randn(state.rng, length(state.θ)) for i=1:kernel.S]
    for i=1:(kernel.L*kernel.R)
        batch_leapfrog!(kernel, state, model, cv, θs, rs)
        if i % kernel.L == 0
            j = Int(i/kernel.L)
            kernel.logσ2[j,:] = log.(diag(cov(rs)))
            kernel.μ[j,:] = sum(rs) / length(rs)
            rs = batch_refresh(rs, kernel.logσ2[j,:], kernel.μ[j,:])
        end
    end
    kernel.logσ2 = vec(kernel.logσ2)
    kernel.μ = vec(kernel.μ)
    @info "warm start done"
end

function batch_leapfrog!(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator,
                            θs::AbstractArray, rs::AbstractArray)
    θ0 = copy(state.θ)
    for i=1:kernel.S
        state.θ = θs[i]
        θs[i], rs[i] = leapfrog!(kernel, state, rs[i], model, cv)
    end
    state.θ = θ0
end

function batch_refresh(rs::AbstractArray, logσ2::AbstractArray, μ::AbstractArray)
    λ = x -> diagm(1. ./ exp.(logσ2)) * (x .- μ)
    return λ.(rs)
end

function est_gradient(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    param_vec = vcat(log.(copy(cv.weights)), copy(kernel.logϵ), copy(kernel.μ), copy(kernel.logσ2))
    return Zygote.gradient(p -> est_obj(kernel, state, model, cv, p), param_vec)[1]
end

function est_loss(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    param_vec = vcat(log.(copy(cv.weights)), copy(kernel.logϵ), copy(kernel.μ), copy(kernel.logσ2))
    return est_obj(kernel, state, model, cv, param_vec)
end

function est_obj(kernel::SHF, state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator, p::AbstractArray)
    cv_zero = ZeroLogProbEstimator(N = kernel.S)
    ws0, logϵ0 = copy(cv.weights), copy(kernel.logϵ)
    logws = p[1:cv.N]
    logϵ = p[cv.N+1:cv.N+length(state.θ)]
    μr = reshape(p[cv.N+length(state.θ)+1:cv.N+length(state.θ)+kernel.R*length(state.θ)], (kernel.R, length(state.θ)))
    logσ2r = reshape(p[cv.N+length(state.θ)+kernel.R*length(state.θ)+1:end], (kernel.R, length(state.θ)))
    o = 0.
    for i=1:kernel.n
        update_estimator!(state, model, cv_zero, nothing, nothing, nothing)
        θ0, r0 = kernel.q0_sample(state.rng, kernel.q0, 1), randn(state.rng, length(state.θ))
        cv.weights = exp.(logws)
        kernel.logϵ = logϵ
        θ, r, logdet = flow(kernel, state, model, cv, copy(θ0), copy(r0), μr, logσ2r)
        θ_reset = copy(state.θ)
        state.θ = copy(θ)
        elbo = (log_potential(state, model, cv_zero) - 0.5*norm(r)^2) - 
                    (kernel.q0_logpdf(θ0, kernel.q0) - 0.5*norm(r0)^2 - logdet)
        state.θ = θ_reset
        o -= elbo/kernel.n
    end
    cv.weights = ws0
    kernel.logϵ = logϵ0
    return o
end