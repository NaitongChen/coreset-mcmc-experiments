using JLD2
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
names = ["CoresetMCMC (SG)", "QNC", "SHF", "Uniform", "Austerity", "Confidence", "SGLD", "SGHMC", "CoresetMCMC"]
colours = [palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_8)[4], palette(:Paired_10)[8],
            palette(:Paired_12)[12], palette(:Paired_10)[9], palette(:Paired_8)[3], palette(:Paired_10)[7],
            palette(:Paired_8)[2]]

n_run = 10

KLs = zeros(n_run, 9)
mrels = zeros(n_run, 9)
srels = zeros(n_run, 9)
esss = zeros(n_run, 9)
trains = zeros(n_run, 9)

################################
################################
# KL coreset size
################################
################################
Threads.@threads for i in 1:n_run
    println(i)
    KLs[i,1] = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(500) * "_" * string(i) * ".jld", "kl")
    if isfile("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld")
        KLs[i,2] = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "kl")
    else
        KLs[i,2] = NaN
    end
    KLs[i,3] = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "kl")
    KLs[i,4] = load("poisson_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "kl")
    KLs[i,5] = load("poisson_regression_austerity_" * "0.01_" * string(i) * ".jld", "kl")
    KLs[i,6] = load("poisson_regression_confidence_" * "0.01_" * string(i) * ".jld", "kl")
    KLs[i,7] = load("poisson_regression_sgld_" * string(i) * ".jld", "kl")
    KLs[i,8] = load("poisson_regression_sghmc_" * string(i) * ".jld", "kl")
    KLs[i,9] = load("new_poisson_regression_coresetMCMC_" * "1_" * string(500) * "_" * string(i) * ".jld", "kl")
end

boxplot([names[1]], KLs[:,1], label = names[1], color = colours[1])
boxplot!([names[9]], KLs[:,9], label = names[9], color = colours[9])
boxplot!([names[2]], (KLs[:,2])[KLs[:,2] .!== NaN], label = names[2], color = colours[2])
boxplot!([names[3]], KLs[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], KLs[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], KLs[:,5], label = names[5], color = colours[5])
boxplot!([names[6]], KLs[:,6], label = names[6], color = colours[6])
boxplot!([names[7]], KLs[:,7], label = names[7], color = colours[7])
boxplot!([names[8]], KLs[:,8], label = names[8], color = colours[8], yscale = :log10, legend=false, xrotation=-20)
ylabel!("Two-Moment KL")
yticks!(10. .^[-2:1:6;])
savefig("plots/kl_all.png")

################################
################################
# mrel coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld")["data"]
stan_mean = vec(mean(D_stan, dims=1))
Threads.@threads for i in 1:n_run
    println(i)
    m1 = load("new_poisson_regression_coresetMCMC_" * "0.3_" *string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,1] = norm(m1 - stan_mean) / norm(stan_mean)
    if isfile("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld")
        m_method = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "mean")
        mrels[i,2] = norm(m_method - stan_mean) / norm(stan_mean)
    else
        mrels[i,2] = NaN
    end
    m_method = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,3] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("poisson_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,4] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("poisson_regression_austerity_" * "0.01_" * string(i) * ".jld", "mean")
    mrels[i,5] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("poisson_regression_confidence_" * "0.01_" * string(i) * ".jld", "mean")
    mrels[i,6] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("poisson_regression_sgld_" * string(i) * ".jld", "mean")
    mrels[i,7] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("poisson_regression_sghmc_" * string(i) * ".jld", "mean")
    mrels[i,8] = norm(m_method - stan_mean) / norm(stan_mean)
    m_method = load("new_poisson_regression_coresetMCMC_" * "1_" *string(500) * "_" * string(i) * ".jld", "mean")
    mrels[i,9] = norm(m_method - stan_mean) / norm(stan_mean)
end

boxplot([names[1]], mrels[:,1], label = names[1], color = colours[1])
boxplot!([names[9]], mrels[:,9], label = names[9], color = colours[9])
boxplot!([names[2]], (mrels[:,2])[mrels[:,2] .!== NaN], label = names[2], color = colours[2])
boxplot!([names[3]], mrels[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], mrels[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], mrels[:,5], label = names[5], color = colours[5])
boxplot!([names[6]], mrels[:,6], label = names[6], color = colours[6])
boxplot!([names[7]], mrels[:,7], label = names[7], color = colours[7])
boxplot!([names[8]], mrels[:,8], label = names[8], color = colours[8], yscale = :log10, legend=false, xrotation=-20)
ylabel!("Relative mean error")
yticks!(10. .^[-10:1:6;])
savefig("plots/mrel_all.png")

################################
################################
# srel coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld")["data"]
stan_cov = cov(D_stan, dims=1)
Threads.@threads for i in 1:n_run
    println(i)
    m1 = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,1] = norm(m1 - stan_cov) / norm(stan_cov)
    if isfile("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld")
        m_method = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "cov")
        srels[i,2] = norm(m_method - stan_cov) / norm(stan_cov)
    else
        srels[i,2] = NaN
    end
    m_method = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,3] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("poisson_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,4] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("poisson_regression_austerity_" * "0.001_" * string(i) * ".jld", "cov")
    srels[i,5] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("poisson_regression_confidence_" * "0.001_" * string(i) * ".jld", "cov")
    srels[i,6] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("poisson_regression_sgld_" * string(i) * ".jld", "cov")
    srels[i,7] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("poisson_regression_sghmc_" * string(i) * ".jld", "cov")
    srels[i,8] = norm(m_method - stan_cov) / norm(stan_cov)
    m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(500) * "_" * string(i) * ".jld", "cov")
    srels[i,9] = norm(m_method - stan_cov) / norm(stan_cov)
end

boxplot([names[1]], srels[:,1], label = names[1], color = colours[1])
boxplot!([names[9]], srels[:,9], label = names[9], color = colours[9])
boxplot!([names[2]], (srels[:,2])[srels[:,2] .!== NaN], label = names[2], color = colours[2])
boxplot!([names[3]], srels[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], srels[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], srels[:,5], label = names[5], color = colours[5])
boxplot!([names[6]], srels[:,6], label = names[6], color = colours[6])
boxplot!([names[7]], srels[:,7], label = names[7], color = colours[7])
boxplot!([names[8]], srels[:,8], label = names[8], color = colours[8], yscale = :log10, legend=false, xrotation=-20)
ylabel!("Relative cov error")
yticks!(10. .^[-2:1:6;])
savefig("plots/srel_all.png")

################################
################################
# ess coreset size
################################
################################
for i in 1:n_run
    println(i)
    m_method = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[50001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[100001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,1] = minimum(ess(split)) / time
    
    if isfile("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld")
        m_method = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "θs")
        time = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[end-10000+1]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[end-10000+1:end, :]
        split = rand(10000, 1, size(D_stan, 2))
        split[:,1,:] = m_method
        esss[i,2] = minimum(ess(split)) / time
    else
        esss[i,2] = NaN
    end
    
    m_method = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[1]
    m_method = reduce(hcat, m_method)'
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,3] = minimum(ess(split)) / time
    
    m_method = load("poisson_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_uniform_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,4] = minimum(ess(split)) / time
    
    m_method = load("poisson_regression_austerity_" * "0.01_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_austerity_" * "0.01_" *string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,5] = minimum(ess(split)) / time
    
    m_method = load("poisson_regression_confidence_" * "0.01_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_confidence_" * "0.01_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,6] = minimum(ess(split)) / time
    
    m_method = load("poisson_regression_sgld_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_sgld_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,7] = minimum(ess(split)) / time
    
    m_method = load("poisson_regression_sghmc_" * string(i) * ".jld", "θs")
    time = load("poisson_regression_sghmc_" * string(i) * ".jld", "c_time")
    time = time[end] - time[10001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[10001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,8] = minimum(ess(split)) / time

    m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(500) * "_" * string(i) * ".jld", "θs")
    time = load("new_poisson_regression_coresetMCMC_" * "1_" * string(500) * "_" * string(i) * ".jld", "c_time")
    time = time[end] - time[50001]
    m_method = reduce(hcat, m_method)'
    m_method = m_method[100001:end, :]
    split = rand(10000, 1, size(D_stan, 2))
    split[:,1,:] = m_method
    esss[i,9] = minimum(ess(split)) / time
end

boxplot([names[1]], esss[:,1], label = names[1], color = colours[1])
boxplot!([names[9]], esss[:,9], label = names[9], color = colours[9])
boxplot!([names[2]], (esss[:,2])[esss[:,2] .!== NaN], label = names[2], color = colours[2])
boxplot!([names[3]], esss[:,3], label = names[3], color = colours[3])
boxplot!([names[4]], esss[:,4], label = names[4], color = colours[4])
boxplot!([names[5]], esss[:,5], label = names[5], color = colours[5])
boxplot!([names[6]], esss[:,6], label = names[6], color = colours[6])
boxplot!([names[7]], esss[:,7], label = names[7], color = colours[7])
boxplot!([names[8]], esss[:,8], label = names[8], color = colours[8], yscale = :log10, legend=false, xrotation=-20)
ylabel!("min ESS/s")
yticks!(10. .^[-2:1:6;])
savefig("plots/ess_all.png")

################################
################################
# training time coreset size
################################
################################
for i in 1:n_run
    println(i)
    time = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(500) * "_" * string(i) * ".jld", "c_time")
    trains[i,1] = time[50000]
    
    if isfile("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld")
        time = load("new_poisson_regression_QNC_" * string(500) * "_" * string(i) * ".jld", "c_time")
        trains[i,2] = time[100]
    else
        trains[i,2] = NaN
    end
    
    time = load("poisson_regression_SHF_" * string(500) * "_" * string(i) * ".jld", "c_time")
    trains[i,3] = time[1]

    time = load("new_poisson_regression_coresetMCMC_" * "1_" * string(500) * "_" * string(i) * ".jld", "c_time")
    trains[i,9] = time[50000]
end

boxplot([names[1]], trains[:,1], label = names[1], color = colours[1])
boxplot!([names[9]], trains[:,9], label = names[9], color = colours[9])
boxplot!([names[2]], (trains[:,2])[trains[:,2] .!== NaN], label = names[2], color = colours[2])
boxplot!([names[3]], trains[:,3], label = names[3], color = colours[3], #=yscale = :log10,=# legend=false, xrotation=-20)
ylabel!("Training time (s)")
yticks!([100, 500, 1000, 2000, 3000, 4000, 5000, 6000])
# yticks!(10. .^[2.4:0.2:3.8;])
ylims!((100, 5000))
savefig("plots/train_all.png")

JLD2.save("results.jld", "KLs", KLs,
                        "mrels", mrels,
                        "srels", srels,
                        "esss", esss,
                        "trains", trains)

# KLs = JLD2.load("results.jld", "KLs")
# mrels = JLD2.load("results.jld", "mrels")
# srels = JLD2.load("results.jld", "srels")
# esss = JLD2.load("results.jld", "esss")
# trains = JLD2.load("results.jld", "trains")