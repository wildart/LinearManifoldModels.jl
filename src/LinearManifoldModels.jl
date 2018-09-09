module LinearManifoldModels

import Distributions: estimate, partype, pdf, logpdf, _logpdf, _logpdf!, _pdf, _pdf!, _rand!,
                      sampler, Distribution, Estimator, ContinuousMultivariateDistribution,
                      Sampleable, Multivariate, Discrete, MixtureModel
import StatsBase: params, entropy, fit, Histogram, sturges
import KernelDensity
import LMCLUS

export NonparametricEstimator,
       EmpiricalLinearManifold,
       EmpiricalLinearManifoldSampler,
       estimate,
       params,
       partype,
       entropy,
       pdf,
       logpdf,
       fit

mutable struct NonparametricEstimator{D<:Distribution, T} <: Estimator{D} end
NonparametricEstimator(::Type{D}, ::Type{T}) where {D<:Distribution, T} = NonparametricEstimator{D, T}()
estimate(e::NonparametricEstimator{D, T}, args...; kvargs...) where {D<:Distribution, T} = fit(D, T, args...; kvargs...)

struct EmpiricalLinearManifold{T<:Real, E} <: ContinuousMultivariateDistribution
    μ::Vector{T}         # Translation vector matrix N x 1
    B::Matrix{T}         # Basis vectors matrix N x K
    estimate::Vector{E}  # K+1 - subspace dimensions + orthogonal subspace distance
end

### Basic properties
Base.show(io::IO, d::EmpiricalLinearManifold) = print(io, "EmpiricalLinearManifold(D=$(size(d.B,2)))")

Base.length(d::EmpiricalLinearManifold) = length(d.μ)
params(d::EmpiricalLinearManifold) = (d.μ, d.B)
@inline partype(d::EmpiricalLinearManifold{T, E}) where {T<:Real, E} = T

_logpdf(d::EmpiricalLinearManifold, x::AbstractVector) = log.(_pdf(d, x))
_logpdf!(r::AbstractArray, d::EmpiricalLinearManifold, X::AbstractMatrix) = log.(_pdf!(r, d, X))

struct EmpiricalLinearManifoldSampler{T<:Real, E} <: Sampleable{Multivariate,Discrete}
    μ::Vector{T}            # Translation vector matrix N x 1
    estimate::Vector{E}     # K+1 - subspace dimensions + orthogonal subspace distance
end

_rand!(d::EmpiricalLinearManifold, x::AbstractVector) = _rand!(sampler(d), x)
sampler(d::EmpiricalLinearManifold{T,E}) where {T<:Real, E} = EmpiricalLinearManifoldSampler{T,E}(d.μ, d.estimate)
Base.length(s::EmpiricalLinearManifoldSampler) = length(s.μ)

include("hist.jl")
include("kde.jl")

end # module
