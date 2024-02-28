module MCMCsampler

using Statistics
using StatsBase
using Distributions
using Random
using LinearAlgebra
using Parameters
using ProgressMeter
using Infiltrator
using ForwardDiff
using Zygote
using PDMats
using LogExpFunctions
using Distances

abstract type AbstractModel end
abstract type AbstractState end
abstract type AbstractMetaState end
abstract type AbstractKernel end
abstract type AbstractAlgorithm end
abstract type AbstractLogProbEstimator end
abstract type SizeBasedLogProbEstimator <: AbstractLogProbEstimator end
abstract type QualityBasedLogProbEstimator <: AbstractLogProbEstimator end

# logProbEstimators
include("estimators/ZeroLogProbEstimator.jl")
export ZeroLogProbEstimator
include("estimators/CoresetLogProbEstimator.jl")
export CoresetLogProbEstimator
include("estimators/AusterityLogProbEstimator.jl")
export AusterityLogProbEstimator
include("estimators/ConfidenceLogProbEstimator.jl")
export ConfidenceLogProbEstimator
include("estimators/ModeLogProbEstimator.jl")
export ModeLogProbEstimator

# kernels
include("methods/kernels/SliceSamplerMD.jl")
export SliceSamplerMD
include("methods/kernels/QualityBasedMetropolisHastings.jl")
export QualityBasedMetropolisHastings
include("methods/kernels/Laplace.jl")
export LaplaceApproxDeterministic
export LaplaceApproxStochastic
include("methods/kernels/ULA.jl")
export ULA
include("methods/kernels/StochasticGradientHMC.jl")
export SGHMC
include("methods/kernels/SparseHamFlow.jl")
export SHF
include("methods/kernels/GibbsSR.jl")
export GibbsSR

# meta-algorithms
include("methods/meta_algorithms/CoresetMCMC.jl")
export CoresetMCMC
include("methods/meta_algorithms/QuasiNewtonCoreset.jl")
export QuasiNewtonCoreset

# models
include("models/LinearRegressionModel.jl")
export LinearRegressionModel
include("models/LogisticRegressionModel.jl")
export LogisticRegressionModel
include("models/PoissonRegressionModel.jl")
export PoissonRegressionModel
include("models/SparseRegressionModel.jl")
export SparseRegressionModel
include("models/log_potentials.jl")

# sampler state tracker
include("States.jl")
export State
export MetaState

# sampler API
include("Sampler.jl")
export sample!

# utilities
include("utilities.jl")

end # module MCMCsampler