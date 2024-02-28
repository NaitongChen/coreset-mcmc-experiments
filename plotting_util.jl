using StatsBase, LinearAlgebra, Interpolations
using LinearAlgebra, Random, Distributions

function get_percentiles(dat; p1=25, p2=75)
    n = size(dat,2)
    median_dat = vec(median(dat, dims=1))

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        dat_remove_inf = (dat[:,i])[iszero.(isinf.(dat[:,i]))]
        dat_remove_nan = (dat_remove_inf)[iszero.(isnan.(dat_remove_inf))]
        median_dat[i] = median(dat_remove_nan)
        plow[i] = median_dat[i] - percentile(vec(dat_remove_nan), p1)
        phigh[i] = percentile(vec(dat_remove_nan), p2) - median_dat[i]
    end

    return plow, phigh
end

function get_medians(dat)
    n = size(dat,2)
    median_dat = vec(median(dat, dims=1))

    for i in 1:n
        dat_remove_inf = (dat[:,i])[iszero.(isinf.(dat[:,i]))]
        dat_remove_nan = (dat_remove_inf)[iszero.(isnan.(dat_remove_inf))]
        median_dat[i] = median(dat_remove_nan)
    end

    return median_dat
end

function time_range(time)
    if size(time,2) == 1
        ts = time
    else
        begin_time = maximum(time[:,1])
        end_time = minimum(time[:,end])
        ts = [begin_time:(end_time - begin_time)/(iter / save_freq ):end_time ;]
    end

    return ts
end

function get_interpolated_data(dat, time)
    m, n = size(dat)
    ts = time_range(time)
    interpolated_data = zeros(m,n)

    for i in 1:m
        interp = LinearInterpolation(time[i,:], dat[i,:])
        interpolated_data[i,:] = interp(ts)
    end

    return interpolated_data
end

function time_median(dat, time)
    if size(dat,2) == 1
       return dat
    else
        dat = get_interpolated_data(dat, time)
        return vec(median(dat, dims=1))
    end
end

function time_percentiles(dat, time; p1=25, p2=75)
    if size(dat,2) == 1
        plow = zeros(n)
        phigh = zeros(n)
    else
        dat = get_interpolated_data(dat, time)
        n = size(dat,2)
        median_dat = vec(median(dat, dims=1))

        plow = zeros(n)
        phigh = zeros(n)

        for i in 1:n
            plow[i] = median_dat[i] - percentile(vec(dat[:,i]), p1)
            phigh[i] = percentile(vec(dat[:,i]), p2) - median_dat[i]
        end
    end

    return plow, phigh
end

function rel_err_mean(D_full_HMC, D)
    μ = vec(mean(D_full_HMC, dims=1))
    μhat = vec(mean(D, dims=1))
    return norm(μ - μhat) / norm(μ)
end

function rel_err_cov(D_full_HMC, D)
    Σ = cov(D_full_HMC, D_full_HMC)
    Σhat = cov(D, D)
    return norm(Σ - Σhat) / norm(Σ)
end


function rel_err_mean_no_hmc(D, μ)
    μhat = vec(mean(D, dims=1))
    return norm(vec(μ) - μhat) / norm(μ)
end

function rel_err_mean_no_dat(μhat, μ)
    return norm(vec(μ) - μhat) / norm(μ)
end

function rel_err_cov_no_hmc(D, Σ)
    Σhat = cov(D, D)
    return norm(Σ - Σhat) / norm(Σ)
end

function rel_err_cov_no_dat(Σhat, Σ)
    return norm(Σ - Σhat) / norm(Σ)
end

function rel_err(D_full_HMC, D)
    e_mean = rel_err_mean(D_full_HMC, D)
    e_cov = rel_err_cov(D_full_HMC, D)
    return e_mean, e_cov
end


function rel_err_log_cov(D_full_HMC, D)
    Σ = cov(D_full_HMC, D_full_HMC)
    Σhat = cov(D, D)

    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_log_cov_no_hmc(D, Σ)
    Σhat = cov(D, D)
    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_log_cov_no_dat(Σhat, Σ)
    log_Diag = log.(diag(Σ))
    log_Diag_hat = log.(diag(Σhat))

    return mean(abs.((log_Diag_hat - log_Diag) ./ log_Diag))
end

function rel_err_no_hmc(post_mean, post_cov, D)
    e_mean = rel_err_mean_no_hmc(D, post_mean)
    e_cov = rel_err_cov_no_hmc(D, post_cov)
    e_log = rel_err_log_cov_no_hmc(D, post_cov)
    return e_mean, e_cov, e_log
end

function rel_err_no_dat(post_mean, post_cov, est_mean, est_cov)
    e_mean = rel_err_mean_no_dat(est_mean, post_mean)
    e_cov = rel_err_cov_no_dat(est_cov, post_cov)
    e_log = rel_err_log_cov_no_dat(est_cov, post_cov)
    return e_mean, e_cov, e_log
end

function energy_dist(post_mean, post_cov, D_z)
    n = size(D_z, 1)
    d = MvNormal(post_mean, post_cov)
    D_z_star = Matrix(rand(d, n)')
    
    n_half = Int(floor(n/2))
    
    D = D_z[1:n_half,:]
    D_prime = D_z[n_half+1:end,:]

    D_true = D_z_star[1:n_half,:]
    D_true_prime = D_z_star[n_half+1:end,:]

    return 2*mean(norm.(eachrow(D - D_true))) - mean(norm.(eachrow(D - D_prime))) - mean(norm.(eachrow(D_true - D_true_prime)))
end