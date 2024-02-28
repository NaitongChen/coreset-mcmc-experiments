@with_kw mutable struct State <: AbstractState
    θ::Union{AbstractArray, Number}
    weight::Union{Float64, Nothing} = nothing
    aux::Any = nothing
    rng::AbstractRNG
    lp_evals::UInt = 0
    grad_lp_evals::UInt = 0
    hess_lp_evals::UInt = 0
    XX::Union{AbstractArray, Nothing} = nothing
    Xy::Union{AbstractArray, Nothing} = nothing
end

@with_kw mutable struct MetaState <: AbstractMetaState
    states::AbstractArray{State} = Array{State}[]
end