abstract type ConstitutiveLaw end

@params struct NeoHooke{T} <: ConstitutiveLaw
    μ::T
    κ::T
end
@params struct MooneyRivlin{T} <: ConstitutiveLaw
    C₁₀::T
    C₀₁::T
    κ::T
end
@params struct Yeoh{T,S} <: ConstitutiveLaw
    Cᵢ₀::T
    κ::S
end
@params struct ArrudaBoyce{T} <: ConstitutiveLaw
    μ::T
    λₘ::T
    κ::T
end
# @params struct Ogden{T,S} <: ConstitutiveLaw
#     μᵢ::T
#     αᵢ::T
#     κ::S
# end

function NeoHooke(ξ::Dict{Symbol,<:Real})
    μ = ξ[:μ]
    κ = ξ[:κ]
    _T = promote_type(typeof(μ), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return NeoHooke(T(μ),T(κ))
end
function MooneyRivlin(ξ::Dict{Symbol,<:Real})
    C₁₀ = ξ[:C₁₀]
    C₀₁ = ξ[:C₀₁]
    κ = ξ[:κ]
    _T = promote_type(typeof(C₁₀), typeof(C₀₁), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return MooneyRivlin(T(C₁₀),T(C₀₁),T(κ))
end
function Yeoh(ξ::Dict{Symbol,<:Any})
    Cᵢ₀ = ξ[:Cᵢ₀]
    κ = ξ[:κ]
    _T = promote_type(typeof(Cᵢ₀))
    T = _T <: Integer ? Float64 : _T
    _S = promote_type(typeof(κ))
    S = _T <: Integer ? Float64 : _S
    return Yeoh(T(Cᵢ₀),S(κ))
end
function ArrudaBoyce(ξ::Dict{Symbol,<:Real})
    μ = ξ[:μ]
    λₘ = ξ[:λₘ]
    κ = ξ[:κ]
    _T = promote_type(typeof(μ), typeof(λₘ), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return ArrudaBoyce(T(μ),T(λₘ),T(κ))
end
# function Ogden(ξ::Dict{Symbol,<:Any})
#     μᵢ = ξ[:μᵢ]
#     αᵢ = ξ[:αᵢ]
#     κ = ξ[:κ]
#     _T = promote_type(typeof(μᵢ),typeof(αᵢ))
#     T = _T <: Integer ? Float64 : _T
#     _S = promote_type(typeof(κ))
#     S = _T <: Integer ? Float64 : _S
#     return Ogden(T(μᵢ),T(αᵢ),S(κ))
# end