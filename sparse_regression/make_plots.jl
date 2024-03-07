using JLD
using LinearAlgebra
using Plots
using Measures
using Statistics
using MCMCDiagnosticTools
using StatsPlots
include("../plotting_util.jl")

########################
# legends
########################
n_run = 10
n_samples = 10000
names = ["CoresetMCMC-S", "QNC", "Uniform", "Austerity", "Confidence", "CoresetMCMC"]
colours = [palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[8],
            palette(:Paired_12)[12], palette(:Paired_10)[9], palette(:Paired_8)[2]]

KLs = zeros(n_run, 6)
mrels = zeros(n_run, 6)
srels = zeros(n_run, 6)
KLDs = zeros(n_run, 6)
esss = zeros(n_run, 6)
essDs = zeros(n_run, 6)
trains = zeros(n_run, 6)

################################
################################
# KL coreset size
################################
################################
Threads.@threads for i in 1:n_run
    println(i)
    KLs[i,1] = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "kl")
    KLs[i,2] = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "kl")
    KLs[i,3] = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "kl")
    KLs[i,4] = load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "kl")
    KLs[i,5] = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "kl")
    KLs[i,6] = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "kl")
end

boxplot([names[1]], KLs[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], KLs[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], KLs[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], KLs[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], KLs[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], KLs[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("Two-Moment KL")
yticks!(10. .^[-2:1:6;])
savefig("plots/kl_all.png")

################################
################################
# mrel
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
stan_mean = vec(mean(D_stan[:,vcat(1:10, 21)], dims=1))
Threads.@threads for i in 1:n_run
    println(i)
    m_method = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,1] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,2] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,3] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "mean")
    mrels[i,4] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "mean")
    mrels[i,5] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,6] = norm(m_method - stan_mean) / norm(stan_mean)
end

boxplot([names[1]], mrels[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], mrels[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], mrels[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], mrels[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], mrels[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], mrels[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("Relative mean error")
yticks!(10. .^[-5:1:-1;])
ylims!((10^-5, 10^-1))
savefig("plots/mrel_all.png")

################################
################################
# srel
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
stan_cov = cov(D_stan[:,vcat(1:10, 21)], dims=1)
Threads.@threads for i in 1:n_run
    println(i)
    m_method = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,1] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,2] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,3] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "cov")
    srels[i,4] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "cov")
    srels[i,5] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,6] = norm(m_method - stan_cov) / norm(stan_cov)
end

boxplot([names[1]], srels[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], srels[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], srels[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], srels[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], srels[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], srels[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("Relative cov error")
yticks!(10. .^[-3:1:1;])
ylims!((10^-3, 10^1))
savefig("plots/srel_all.png")

################################
################################
# KL discrete
################################
################################
for i in 1:n_run
    println(i)
    println("cmcmcs")
    gamma_m = reduce(hcat, load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end
    
    KLDs[i,1] = copy(KKL)

    println("qnc")
    gamma_m = reduce(hcat, load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end
    
    KLDs[i,2] = copy(KKL)

    println("unif")
    gamma_m = reduce(hcat, load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end
    
    KLDs[i,3] = copy(KKL)

    println("austerity")
    gamma_m = reduce(hcat, load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end
    
    KLDs[i,4] = copy(KKL)

    println("confidence")
    gamma_m = reduce(hcat, load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end

    KLDs[i,5] = copy(KKL)

    println("cmcmc")
    gamma_m = reduce(hcat, load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,11:20]
    gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
    mapm = countmap(gamma_m)
    
    f((k,v)) = k => convert(Float64, v)
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    gamma_stan = D_stan[:, 11:20]
    gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
    mapstan = countmap(gamma_stan)
    mapstan = Dict(Iterators.map(f, pairs(mapstan)))
    mapm = Dict(Iterators.map(f, pairs(mapm)))

    mid = merge(mapstan, mapm)
    for (key, value) in mid
        mid[key] = 0
        if haskey(mapm, key)
            mid[key] +=  0.5 * (mapm[key] / 10000)
        end
        if haskey(mapstan, key)
            mid[key] +=  0.5 * (mapstan[key] / 100000)
        end
    end

    KKL = 0
    for (key, value) in mapstan
        KKL += 0.5 * ((mapstan[key]/100000) * (log((mapstan[key]/100000)) - log(mid[key])))
    end
    for (key, value) in mapm
        KKL += 0.5 * ((mapm[key]/10000) * (log((mapm[key]/10000)) - log(mid[key])))
    end
    
    KLDs[i,6] = copy(KKL)
end

boxplot([names[1]], KLDs[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], KLDs[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], KLDs[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], KLDs[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], KLDs[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], KLDs[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("JS divergence")
yticks!(10. .^[-5:1:6;])
ylims!((10^-4, 10^0))
savefig("plots/kl_discrete.png")

################################
################################
# ess coreset size
################################
################################
for i in 1:n_run
    println(i)
    println("cmcmcs")
    m_method = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[25001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[50001:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,1] = minimum(ess(split)) / time
    
    println("qnc")
    m_method = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[end-10000+1]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[end-10000+1:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,2] = minimum(ess(split)) / time
    
    println("unif")
    m_method = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,3] = minimum(ess(split)) / time
    
    println("austerity")
    m_method = load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_austerity_" * "0.001_0.5_" *string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,4] = minimum(ess(split)) / time
    
    println("confidence")
    m_method = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,5] = minimum(ess(split)) / time

    println("cmcmc")
    m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[25001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[50001:end, vcat(1:10, 21)]
    split = rand(10000, 1, 11)
    split[:,1,:] = m_method
    esss[i,6] = minimum(ess(split)) / time
end

boxplot([names[1]], esss[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], esss[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], esss[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], esss[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], esss[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], esss[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("min ESS/s")
yticks!(10. .^[-5:1:6;])
ylims!((10^-1, 10^4))
savefig("plots/ess_all.png")

################################
################################
# ess discrete
################################
################################
for i in 1:n_run
    println(i)
    println("cmcmcs")
    m_method = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_coresetMCMC_" * "2_0.5_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[25001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[50001:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,1] = minimum(ess(split)) / time
    
    println("qnc")
    m_method = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[end-10000+1]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[end-10000+1:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,2] = minimum(ess(split)) / time
    
    println("unif")
    m_method = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,3] = minimum(ess(split)) / time
    
    println("austerity")
    m_method = load("sparse_regression_austerity_" * "0.001_0.5_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_austerity_" * "0.001_0.5_" *string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,4] = minimum(ess(split)) / time
    
    println("confidence")
    m_method = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_confidence_" * "0.001_0.5_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,5] = minimum(ess(split)) / time

    println("cmcmc")
    m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("sparse_regression_coresetMCMC_" * "0.01_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[25001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[50001:end, 11:20]
    prec = zeros(n_samples)
    for i in [1:n_samples;]
        prec[i] = sum(m_method[i,:] .== [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]) / 10
    end
    split = rand(10000, 1, 1)
    split[:,1,:] = prec
    essDs[i,6] = minimum(ess(split)) / time
end

boxplot([names[1]], essDs[:,1], label = names[1], color = colours[1])
boxplot!([names[6]], essDs[:,6], label = names[6], color = colours[6])
boxplot!([names[2]], essDs[:,2], label = names[2], color = colours[2])
boxplot!([names[3]], essDs[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], essDs[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], essDs[:,5], label = names[5], color = colours[5], yscale = :log10, legend=false, xrotation=40, guidefontsize=20, tickfontsize=15, formatter=:plain, margin=10mm)
ylabel!("ESS/s")
yticks!(10. .^[-5:1:6;])
ylims!((10^-1, 10^4))
savefig("plots/ess_discrete.png")

save("results.jld", "KLs", KLs,
                    "KLDs", KLDs,
                    "esss", esss,
                    "essDs", essDs,
                    "mrels", mrels, 
                    "srels", srels)

# KLs = load("results.jld", "KLs")
# KLDs = load("results.jld", "KLDs")
# esss = load("results.jld", "esss")
# essDs = load("results.jld", "essDs")
# mrels = load("results.jld", "mrels")
# srels = load("results.jld", "srels")