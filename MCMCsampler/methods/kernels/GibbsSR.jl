mutable struct GibbsSR <: AbstractKernel end

function step!(kernel::GibbsSR, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    update_estimator!(state, model, cv, nothing, nothing, nothing)
    X = @view(cv.sub_dataset[:,1:end-1])
    y = @view(cv.sub_dataset[:,end])

    if iter == 1
        Xp = @view(cv.sub_xp[1:end-1,:])
        state.XX = Vector{Vector{Float64}}(undef, 0)
        state.Xy = Vector{Vector{Float64}}(undef, 0)

        for i in [1:model.d;]
            push!(state.Xy, Xp[i,:].*y)
            for j in [1:model.d;]
                push!(state.XX, Xp[i,:].*X[:,j])
            end
        end
    end

    # update β
    iszerovec = (state.θ[(model.d+1):(2*model.d)] .== 0)
    diag_entry = (iszerovec .* 1. + (1 .- iszerovec) .* model.c) .* model.τ
    Vβinv = diagm(diag_entry.^(-2))
    binv = cv.weights ./ state.θ[end]

    # Binv = diagm(cv.weights ./ state.θ[end])
    # X = @view(cv.sub_dataset[:,1:end-1])
    # Xp = @view(cv.sub_xp[1:end-1,:])
    # y = @view(cv.sub_dataset[:,end])
    # M0 = Xp * Binv
    # M1 = M0 * X
    # M2 = M0 * y
    # @infiltrate

    M1 = zeros(model.d, model.d)
    M2 = zeros(model.d)
    for i in [1:model.d;]
        M2[i] = state.Xy[i]' * binv
        for j in [1:model.d;]
            M1[i,j] = state.XX[(i-1)*model.d + j]' * binv
            M1[j,i] = M1[i,j]
        end
    end

    Σinv = M1 + Vβinv
    Σinv = (Σinv + Σinv') ./ 2
    Σ = (Σinv \ I)
    Σ = (Σ + Σ') ./ 2
    μ = Σ * M2
    state.θ[1:model.d] = rand(MvNormal(μ, Σ))

    # update σ2
    residual = y - X*state.θ[1:model.d]
    iga = (sum(cv.weights) + model.ν)/2
    igb = (sum((residual.^2) .* cv.weights) + model.ν * model.λ)/2
    state.θ[end] = rand(InverseGamma(iga, igb))

    # update γ
    seq = sample(state.rng, [1:model.d;], model.d, replace = false, ordered = false)
    for i in 1:model.d
        a, b = compute_ab(state, model, cv, seq[i], M1, M2)
        state.θ[model.d+seq[i]] = rand(Bernoulli(a/(a+b)))
    end
end

function compute_ab(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, i::Int64,
                    M1::AbstractArray, M2::AbstractArray)
    θ_copy = copy(state.θ)

    state.θ[model.d+i] = 1
    iszerovec = (state.θ[(model.d+1):(2*model.d)] .== 0)
    diag_entry = (iszerovec .* 1. + (1 .- iszerovec) .* model.c) .* model.τ
    Vβinv = diagm(diag_entry.^(-2))
    Σinv = M1 + Vβinv
    Σinv = (Σinv + Σinv') ./ 2
    Σ = (Σinv \ I)
    Σ = (Σ + Σ') ./ 2
    μ = Σ * M2
    a = exp(log(model.p) + logpdf(MvNormal(μ, Σ), state.θ[1:model.d]))

    state.θ[model.d+i] = 0
    iszerovec = (state.θ[(model.d+1):(2*model.d)] .== 0)
    diag_entry = (iszerovec .* 1. + (1 .- iszerovec) .* model.c) .* model.τ
    Vβinv = diagm(diag_entry.^(-2))
    Σinv = M1 + Vβinv
    Σinv = (Σinv + Σinv') ./ 2
    Σ = (Σinv \ I)
    Σ = (Σ + Σ') ./ 2
    μ = Σ * M2
    b = exp(log1p(-model.p) + logpdf(MvNormal(μ, Σ), state.θ[1:model.d]))

    state.θ = θ_copy
    return a, b
end