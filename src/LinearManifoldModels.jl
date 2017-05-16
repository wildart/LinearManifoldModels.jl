module LinearManifoldModels

using Distributions
import StatsBase
import KernelDensity
import LMCLUS


export NonparametricEstimator, EmpiricalLinearManifold

type NonparametricEstimator{D<:Distribution, T} <: Estimator{D} end
NonparametricEstimator{D<:Distribution, T}(::Type{D}, ::Type{T}) = NonparametricEstimator{D, T}()
Distributions.estimate{D<:Distribution, T}(e::NonparametricEstimator{D, T}, args...; kvargs...) = fit(D, T, args...; kvargs...)

immutable EmpiricalLinearManifold{T<:Real, E} <: ContinuousMultivariateDistribution
    μ::Vector{T}         # Translation vector matrix N x 1
    B::Matrix{T}         # Basis vectors matrix N x K
    estimate::Vector{E}  # K+1 - subspace dimensions + orthogonal subspace distance
end

### Basic properties
Base.show(io::IO, d::EmpiricalLinearManifold) = print(io, "EmpiricalLinearManifold(D=$(size(d.B,2)))")

Base.length(d::EmpiricalLinearManifold) = length(d.μ)
Distributions.params(d::EmpiricalLinearManifold) = (d.μ, d.B)
@inline Distributions.partype{T<:Real, E}(d::EmpiricalLinearManifold{T, E}) = T

### Fitting by histogram

function StatsBase.fit{T <: Real}(::Type{EmpiricalLinearManifold}, ::Type{StatsBase.Histogram},
                                  μ::AbstractVector{T}, B::AbstractMatrix{T}, X::AbstractMatrix{T}; kvargs...)

    function calcrange(x::AbstractVector, binNumber)
        histExtrema = extrema(x)
        histRange = histExtrema[2] - histExtrema[1]
        binSize = histRange/binNumber
        edges = collect(histExtrema[1]:binSize:histExtrema[2]+eps())
        edges[1] -= eps()
        edges[end] += eps()
        return edges
    end

    N, n = size(X)
    K = size(B,2)

    nbins = 0
    for (k,v) in kvargs
        if k == :nbins
            nbins = Int(v)
        end
    end
    if nbins == 0
        nbins = StatsBase.sturges(n)
    end

    estimates = Array{StatsBase.Histogram}(K+1)

    zk = B'*(X .- μ)
    for k in 1:K
        histEdges = calcrange(zk[k,:], nbins)
        estimates[k] = fit(StatsBase.Histogram, zk[k,:], histEdges, closed=:left)
    end

    d = LMCLUS.distance_to_manifold(X, μ, B)
    histEdges = calcrange(d, nbins)
    estimates[K+1] = fit(StatsBase.Histogram, d, histEdges, closed=:left)

	return EmpiricalLinearManifold(μ, B, estimates)
end

StatsBase.fit{T <: Real}(::Type{EmpiricalLinearManifold}, ::Type{StatsBase.Histogram},
                         LMC::LMCLUS.Manifold, X::AbstractMatrix{T}; kvargs...) =
                         fit(EmpiricalLinearManifold, StatsBase.Histogram,
                             LMCLUS.mean(LMC), LMCLUS.projection(LMC), X; kvargs...)

### Fitting by KDE

function StatsBase.fit{T <: Real}(::Type{EmpiricalLinearManifold}, ::Type{KernelDensity.UnivariateKDE},
                                  μ::AbstractVector{T}, B::AbstractMatrix{T}, X::AbstractMatrix{T}; kvargs...)
    function calcrange(x::AbstractVector, binNumber)
        bw = KernelDensity.default_bandwidth(x)
        bb = KernelDensity.kde_boundary(x, bw)
        dist = KernelDensity.kernel_dist(Normal, bw)
        stp = (bb[2]-bb[1])/(binNumber-1)
        mps = bb[1]:stp:bb[2]+eps()
        return mps, dist
    end

    K = size(B,2)

    nquants = 1024 # default
    for (k,v) in kvargs
        if k == :nquants
            nquants = Int(v)
        end
    end

    estimates = Array{KernelDensity.UnivariateKDE}(K+1)

    zk = B'*(X .- μ)
    for k in 1:K
        estimates[k] = KernelDensity.kde(zk[k,:], calcrange(zk[k,:], nquants)...)
    end

    d = LMCLUS.distance_to_manifold(X, μ, B)
    estimates[K+1] = KernelDensity.kde(d, calcrange(d, nquants)...)

	return EmpiricalLinearManifold(μ, B, estimates)
end

StatsBase.fit{T <: Real}(::Type{EmpiricalLinearManifold}, ::Type{KernelDensity.UnivariateKDE},
                         LMC::LMCLUS.Manifold, X::AbstractMatrix{T}; kvargs...) =
                         fit(EmpiricalLinearManifold, KernelDensity.UnivariateKDE,
                             LMCLUS.mean(LMC), LMCLUS.projection(LMC), X; kvargs...)

### Evaluation

function Distributions._pdf{T<:Real}(d::EmpiricalLinearManifold{T, StatsBase.Histogram}, x::AbstractVector{T})
    # Basis subspace
    K = size(d.B,2)
    probValues = zeros(T, K+1)
    binCount = length(d.estimate[1].weights)
    z = d.B'*(x - d.μ)
    dist = LMCLUS.distance_to_manifold(collect(x), d.μ, d.B)
    for k in 1:K+1
        val = k > K ? dist : z[k] # Orthonormal subspace distances for K+1
        edges = first(d.estimate[k].edges)
        binIndex = searchsortedlast(edges, val)
        probValues[k] = 0 < binIndex <= binCount ? d.estimate[k].weights[binIndex]/binCount : 0.
    end
    return prod(probValues)
end

rectint(y, r) = y*r
rectint(y, x1, x2) = rectint(y, x2 - x1)
trapint(y1, y2, r) = r.*(y1.+y2)/2
trapint(y1, y2, x1, x2) = trapint(y1, y2, x2 .- x1)
simpint(y1, y2, y3, x1, x2) = (x2 .- x1).*(y1 .+ 4*y2 .+ y3)/6

function Distributions._pdf{T<:Real}(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE}, x::AbstractVector{T})
    K = size(d.B,2)
    prob = one(T)
    z = d.B'*(x - d.μ)
    dist = LMCLUS.distance_to_manifold(collect(x), d.μ, d.B)
    for k in 1:K+1
        val = k > K ? dist : z[k] # Orthonormal subspace distances for K+1
        stp = step(d.estimate[k].x)

        # Search-based
        # binIndex = searchsortedlast(d.estimate[k].x, val)
        # dv, x1, x2 = if binIndex < 1
        #     d.estimate[k].density[1], d.estimate[k].x[1]-stp, d.estimate[k].x[1]
        # elseif binIndex > length(d.estimate[k].x)
        #     d.estimate[k].density[end], d.estimate[k].x[end], d.estimate[k].x[end]+stp
        # else
        #     d.estimate[k].density[binIndex], d.estimate[k].x[binIndex], d.estimate[k].x[binIndex]+stp
        # end
        # tmp = rectint(dv, x1, x2)

        # Approximation-based
        x1 = val-stp/2
        x2 = val
        x3 = val+stp/2
        # tmp = rectint(pdf(d.estimate[k], x2), x1, x3)
        tmp = trapint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x3), x1, x3)
        # tmp = simpint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x2), pdf(d.estimate[k], x3), x1, x3)

        # if probability is negative
        if tmp > 0.
            prob *= tmp
        else
            prob = 0.
        end
    end
    return prob
end

function Distributions._pdf!{T<:Real}(r::AbstractArray{T}, d::EmpiricalLinearManifold{T,KernelDensity.UnivariateKDE}, X::AbstractMatrix{T})
    K = size(d.B,2)
    N, n = size(X)
    Z = (d.B'*(X .- d.μ))'
    dist = LMCLUS.distance_to_manifold(X, d.μ, d.B)
    for i in 1 : n
        @inbounds r[i] = one(T)
    end
    for k in 1 : K+1
        val = k > K ? dist : Z[:,k] # Orthonormal subspace distances for K+1
        stp = step(d.estimate[k].x)
        x1 = val-stp/2
        x3 = val+stp/2
        tmp = trapint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x3), x1, x3)
        # tmp = simpint(pdf(d.estimate[k], x1), pdf(d.estimate[k], val), pdf(d.estimate[k], x3), x1, x3)
        for i in 1 : n
            @inbounds r[i] *= tmp[i] < zero(T) ? zero(T) : tmp[i]
        end
    end
    return r
end

Distributions._logpdf(d::EmpiricalLinearManifold, x::AbstractVector) = log(Distributions._pdf(d, x))
Distributions._logpdf!(r::AbstractArray, d::EmpiricalLinearManifold, X::AbstractMatrix) = log(Distributions._pdf!(r, d, X))

function Distributions.entropy{T<:Real}(d::EmpiricalLinearManifold{T, StatsBase.Histogram})
    H = 0.0
    for est in d.estimate
        probs = est.weights/sum(est.weights)
        filter!(x->x != zero(T), probs)
        H += -sum(probs.*log(probs))
    end
    return H
end

function Distributions.entropy{Q,R,T<:Real}(M::MixtureModel{Q,R,EmpiricalLinearManifold{T,KernelDensity.Histogram}})
    qb = first(first(components(M)).estimate).weights |> length
    c = length(components(M))
    H = zero(T)
    for i in 1:qb
        Pᵢⱼ = zero(T)
        for j in 1:c
            d = components(M)[j]
            pⱼ = probs(M)[j]
            prob = one(T)
            for est in d.estimate
                prob *= est.weights[i]/sum(est.weights)
            end
            Pᵢⱼ += pⱼ*prob
            # println("Pᵢⱼ($i, $j): $Pᵢⱼ")
        end
        H += Pᵢⱼ > zero(T) ? -Pᵢⱼ*log(Pᵢⱼ) : zero(T)
        # println("H($i): $H")
    end
    return H
end

# function Distributions.entropy{T<:Real}(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE})
#     H = 0.0
#     for est in d.estimate
#         tot = length(est.density)
#         ds = filter(x->x != zero(T), est.density)
#         H += -sum(log(ds))/tot
#     end
#     return H
# end

function Distributions.entropy{T<:Real}(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE})
    H = 0.0
    qb = first(d.estimate).x |> length
    dp = ones(qb-1)
    # for est in d.estimate
    #     stp = step(est.x)
    #     dp = [trapint(est.density[i], est.density[i+1], est.x[i], est.x[i+1]) for i in 1:length(est.density)-1]
    #     filter!(x->x != zero(T), dp)
    #     H += -sum(dp.*log(dp))
    # end
    for est in d.estimate
        stp = step(est.x)
        tmp = [trapint(est.density[i], est.density[i+1], est.x[i], est.x[i+1]) for i in 1:length(est.density)-1]
        dp = dp .* tmp
    end
    H = -sum(dp.*log(dp))
    return H
end

function Distributions.entropy{Q,R,T<:Real}(M::MixtureModel{Q,R,EmpiricalLinearManifold{T,KernelDensity.UnivariateKDE}})
    qb = first(first(components(M)).estimate).x |> length
    c = length(components(M))
    H = zero(T)
    for i in 1:qb-1
        Pᵢⱼ = zero(T)
        for j in 1:c
            d = components(M)[j]
            pⱼ = probs(M)[j]
            prob = one(T)
            for est in d.estimate
                prob *= trapint(est.density[i], est.density[i+1], est.x[i], est.x[i+1])
            end
            Pᵢⱼ += pⱼ*prob
            # println("Pᵢⱼ($i, $j): $Pᵢⱼ")
        end
        H += Pᵢⱼ > zero(T) ? -log(Pᵢⱼ) : zero(T)
        # H += Pᵢⱼ > zero(T) ? -Pᵢⱼ*log(Pᵢⱼ) : zero(T)
        # println("H($i): $H")
    end
    return H
end

### Sampling

# _rand!(d::EmpiricalLinearManifold, x::AbstractVector) = _rand!(sampler(d), x)
# sampler(d::EmpiricalLinearManifold) = EmpiricalLinearManifoldSampler(d)


end # module
