using CSV
using DataFrames
using Random
using JLD
using Statistics
using StatsBase
using Distributions
include("../MCMCsampler/MCMCsampler.jl")
include("../util.jl")

function main(args)
    data = JLD.load("../data/sparse_regression_10000.jld")["data"]
    
    N = length(data)
    d = 5
    
    @assert length(args) == 3 "Error: script has 3 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1])) 

    # Create the model
    println("Initializing model")
    model = MCMCsampler.SparseRegressionModel(length(data), d, data, Matrix(reduce(hcat, data)'), 0.2, 0.1, 1, 10, 1/10, nothing)

    # parse number of samples
    n_samples = parse(Int, args[2])

    # Create the algorithm
    println("Initializing sampler")
    kernel = MCMCsampler.GibbsSR()
    cv = MCMCsampler.CoresetLogProbEstimator(N = parse(Int, args[3]))
    cv.inds = sample(rng, [1:model.N;], cv.N; replace = false)
    cv.sub_dataset = @view(model.datamat[cv.inds,:])
    cv.sub_xp = Matrix(cv.sub_dataset')
    cv.weights = (model.N / cv.N) * ones(cv.N)

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time, weights = MCMCsampler.sample!(kernel, model, cv, 2*n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/sparse_regression_big.jld")["θs"]
    m_method = vec(mean(ts[:,vcat(1:5, 11)], dims=1))
    v_method = cov(ts[:,vcat(1:5, 11)])
    kl_est = kl_gaussian(m_method, v_method, vec(mean(D_stan[:,vcat(1:5, 11)], dims=1)), cov(D_stan[:,vcat(1:5, 11)]))
    println(c_lp[end])
    println(c_time[end])

    save("sparse_regression_uniform_" * args[3] * "_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time,
                                                    "weights", weights, "inds", cv.inds,
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

println("Running sampler with $(ARGS)")
main(ARGS)

# args = ["1", "10000", "1000"] # seed n_sample subsample_size
# main(args)