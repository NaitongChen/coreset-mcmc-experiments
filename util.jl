using LinearAlgebra
using StatsBase

function kl_gaussian(μq, Σq, μp, Σp)
    d = length(μq)
    if det(Σq) < 0
        return 0.5 * (logdet(Σp) - log(1e-20) - d + tr(Σp \ Σq) + transpose(μp - μq) * (Σp \ (μp - μq)))
    else    
        return 0.5 * (logdet(Σp) - logdet(Σq) - d + tr(Σp \ Σq) + transpose(μp - μq) * (Σp \ (μp - μq)))
    end
end

function log_reg_stratified_sampling(model, M, rng)
    ind_seq = [1:model.N ;]
    positives = ind_seq[model.datamat[:,end] .== 1.]
    negatives = ind_seq[model.datamat[:,end] .== 0.]

    count_positives = size(positives, 1)

    # take 50% positive, 50% negative (if possible)
    n_pos = min(Int(ceil(M / 2.)), count_positives)
    n_neg = M - n_pos

    inds_pos = sort(sample(rng, positives, n_pos, replace = false))
    inds_neg = sort(sample(rng, negatives, n_neg, replace = false))

    inds = sort(vcat(inds_pos, inds_neg))

    return inds
end