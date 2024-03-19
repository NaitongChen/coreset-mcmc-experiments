@with_kw mutable struct CoresetMCMC <: AbstractAlgorithm
    kernel::AbstractKernel
    replicas::Int64 = 2
    delay::Int64 = 1
    train_iter::Int64 = 20000
    α::Function = t -> 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
    t::Int64 = 0
    m::Union{Nothing, Vector{Float64}} = nothing
    v::Union{Nothing, Vector{Float64}} = nothing
    cv_zero::Union{Nothing, ZeroLogProbEstimator} = nothing
    proj_n::Int64 = 1000
end

function step!(alg::CoresetMCMC, metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, iter::Int64)
    if iter == 1
        init_estimator!(metaState.states[1], model, cv)
        alg.m = zeros(cv.N)
        alg.v = zeros(cv.N)
    end

    if isnothing(alg.cv_zero)
        alg.cv_zero = ZeroLogProbEstimator(N = alg.proj_n)
        if alg.proj_n == model.N
            update_estimator!(metaState.states[1], model, alg.cv_zero, nothing, nothing, [1:model.N;])
        end
    end

    Threads.@threads for i=1:alg.replicas
        step!(alg.kernel, metaState.states[i], model, cv, iter)
    end 
    # update the coreset weights via ADAM
    if iter <= alg.train_iter * alg.delay && iter % alg.delay == 0
        alg.t += 1
        @unpack α, β1, β2, ϵ, t = alg
        g = est_gradient(metaState, model, cv, alg.cv_zero)
        alg.m .= β1 * alg.m + (1-β1)*g
        alg.v .= β2 * alg.v + (1-β2)*g.^2
        m̂ = alg.m/(1-β1^t)
        v̂ = alg.v/(1-β2^t) 
        cv.weights -= α(t) * m̂./(sqrt.(v̂) .+ ϵ)
        cv.weights = max.(0., cv.weights)
    end
end

function est_gradient(metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, cv_zero::ZeroLogProbEstimator)
    g = project(metaState, model, cv)
    proj_sum = project_sum(metaState, model, cv_zero, cv)
    h = proj_sum .- (g' * cv.weights)
    grd = -g*h/(length(metaState.states)-1)
    return grd
end
