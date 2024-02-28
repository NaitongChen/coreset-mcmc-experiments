@with_kw mutable struct QuasiNewtonCoreset <: AbstractAlgorithm
    kernel::AbstractKernel
    replicas::Int64 = 1
    S::Int64 = 500
    τ::Float64 = 0.01 # regularization
    k::Float64 = 0.9 # line search condition check
    a::Float64 = 1.5 # line search scaling
    β::Float64 = 0.01 # threshold for stopping criterion
    norm_grd::Float64 = 0. # tracking stopping criterion
    t::Int64 = 50 # line search rounds
    K::Int64 = 100 # optimization iteration
    γ::Float64 = 1. # tuned step size
    ls_iter::Int64 = 10 # line search iter
    optimizing::Bool = true
    cv_zero::Union{Nothing, ZeroLogProbEstimator} = nothing
end

function step!(alg::QuasiNewtonCoreset, metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, iter::Int64)
    if isnothing(alg.cv_zero)
        alg.cv_zero = ZeroLogProbEstimator(N = model.N)
        update_estimator!(metaState.states[1], model, alg.cv_zero, nothing, nothing, [1:model.N;])
    end
    
    if iter <= alg.K && alg.optimizing
        @info "iteration " * string(iter) * " / " * string(alg.K)
        if iter == 1
            init_estimator!(metaState.states[1], model, cv)
        end
        θ0 = copy(metaState.states[1].θ)
        θs, _, _, _, _ = sample!(alg.kernel, metaState.states[1], model, cv, alg.S)
        metaState.states[1].θ = θ0
        weights0 = copy(cv.weights)
        g, d = get_descent_vars(alg, metaState, model, cv, θs, alg.cv_zero)
        if iter <= alg.ls_iter
            alg.γ = 1.0
            fail, upd = 0, 0.0
            while all(cv.weights + alg.γ * g .< 0)
                alg.γ /= alg.a
            end
            @showprogress for i in 1:alg.t
                cv.weights = max.(0., cv.weights + alg.γ * d)
                g′, _ = get_descent_vars(alg, metaState, model, cv, θs, alg.cv_zero)
                cv.weights = weights0
                test = (g′' * d) / (g' * d)
                if test >= alg.k || test <= 0
                    alg.γ /= alg.a
                else
                    fail = 1
                    break
                end
                upd = alg.γ*d
            end
            if fail == 0 @warn("failed to tune step size") end
            cv.weights = max.(0., cv.weights + upd)
        else
            cv.weights = max.(0., cv.weights + alg.γ * d)
            curr_norm_grd = norm(g)
            rel_d = abs(alg.norm_grd - curr_norm_grd) / curr_norm_grd
            alg.norm_grd = curr_norm_grd
            if rel_d <= alg.β
                alg.optimizing = false
            end
        end
    end
    step!(alg.kernel, metaState.states[1], model, cv, iter)
end

function get_descent_vars(alg::QuasiNewtonCoreset, metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, θs::AbstractArray, cv_zero::ZeroLogProbEstimator)
    g = project(metaState, model, cv, θs)
    proj_sum = project_sum(metaState, model, cv_zero, θs)
    h = proj_sum .- (g' * cv.weights)
    G = (g * g') / length(θs)
    grd = -g*h/length(metaState.states)
    direction = -(G + alg.τ * I) \ grd
    return grd, direction
end