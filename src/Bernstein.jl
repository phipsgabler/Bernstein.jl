# There's enough code for it to get its own module.  This also makes testing easier.
module Bernstein

using Polynomials

import Base: convert, promote_rule, show
import LinearAlgebra: dot, norm
import Polynomials: poly


# Since we're in a module, we must define the exported things.  Since most of the things is just
# overloaded methods, only the type remains.
export BernsteinPoly

# You do this check so often, we can factor it out.  Alternative: using a separate type
# for intervals.
macro checkinterval(α, β)
    :(@assert $(esc(α)) <= $(esc(β)) "Invalid interval")
end


# Parametrization by a constrained type is better than fields of an abstract type:
# https://docs.julialang.org/en/v1.0/manual/performance-tips/#Avoid-fields-with-abstract-type-1
# But in the case of `n` and `i`, concrete `Int`s make more sense, since they represent natural numbers
# always (this makes conversions below easier).
# And I'm calling it `BernsteinPoly`, since it is a variant of `Poly`.
struct BernsteinPoly{T<:Number}
    n::Int
    i::Int
    α::T
    β::T
end

BernsteinPoly(n, i) = BernsteinPoly(n, i, 0.0, 1.0)


# Just implementing `show` using `print` is the recommended way for custom pretty-printing:
# https://docs.julialang.org/en/v1.0/manual/types/#man-custom-pretty-printing-1
# Here, we can reuse `Polynomials.printpoly` for nicer and more consistent formatting:
function show(io::IO, b::BernsteinPoly)
    print(io, "BernsteinPoly(")
    Polynomials.printpoly(io, poly(b))
    print(io, ")")
end


# `get_p` was essentially the conversion from `BernsteinPoly` to `Poly`.  We can replace this
# by the implementation of proper `convert` methods.  The default parameters are unnecessary,
# since already occuring defaulted in the `BernsteinPoly` constructor.
function convert(::Type{Poly{S}}, b::BernsteinPoly) where {S<:Number}
    n, i, α, β = b.n, b.i, convert(S, b.α), convert(S, b.β)
    @checkinterval α β
    
    return (binomial(n, i)
            * poly(fill(α, i))     # `fill` instead of list comprehension
            * poly(fill(β, n - i))
            * (-1)^(n - i)
            / (β - α)^n)
end

# This allows you to just say `convert(Poly, b)`, automatically reusing the inner type of `b`
convert(::Type{Poly}, b::BernsteinPoly{T}) where {T<:Number} =
    convert(Poly{T}, b)

# While we're at it: conversion between different `BernsteinPoly` values 
convert(::Type{BernsteinPoly{S}}, b::BernsteinPoly) where {S<:Number} =
    BernsteinPoly(b.n, b.i, convert(S, b.α), convert(S, b.β))

# If we're handling different representations, we sometimes need to determine the "most general"
# form, which is called promotion:
promote_rule(a::Type{BernsteinPoly{S}}, b::Type{Poly{T}}) where {S<:Number, T<:Number} =
    Poly{promote_type(S, T)}

promote_rule(::Type{BernsteinPoly{S}}, ::Type{BernsteinPoly{T}}) where {S<:Number, T<:Number} = 
    BernsteinPoly{promote_type(S, T)}

# Also add a method to the `poly` smart constuctor, which is now trivial:
poly(b::BernsteinPoly) = convert(Poly, b)




# Now to the linear algebra part.  We could add methods to `dot` and `norm` from `LinearAlgebra`:
#     function dot(p::Poly{T}, q::Poly{T}, α = zero(T), β = one(T)) where {T<:Number}
#         @checkinterval α β
#         polyint(p * q, α, β)
#     end
# And the norm induced by that inner product:
#     function norm(q::Poly{T}, α = zero(T), β = one(T)) where {T<:Number}
#         @checkinterval α β
#         √dot(q, q, α, β) / (β - α)
#     end
# But such "overloads from outside" are frowned upon.  As of writing this, there's a `norm` method
# in `Polynomials`[1], but not `dot`.  There's an issue about that [2]; basically, `dot` is not
# unique, hence we need to specify the intervals each time here.
# [1] https://github.com/JuliaMath/Polynomials.jl/blob/12d611fcd9b43300ecc83220b52c9127866b1a38/src/Polynomials.jl#L236
# [2] https://github.com/JuliaMath/Polynomials.jl/issues/110


# For Bernstein polynomials, on the other hand, the inner product is defined uniquely, if I
# understood correctly.
function dot(b::BernsteinPoly{T}, q::BernsteinPoly{T}) where {T<:Number}
    @checkinterval b.α q.α
    @checkinterval b.β q.β
    m, i, n, j, α, β = b.n, b.i, q.n, q.i, b.α, b.β
    (β - α) * binomial(m, i) * binomial(n, j) / ((m + n + 1) * binomial(m + n, i + j))
end

# And the induced norm, as before.
function norm(b::BernsteinPoly)
    √dot(b, b) / (b.β - b.α)
end


# As the last remaining question, how to do cross-type inner products.  If `dot` is defined for
# `Poly` as written above, we can use `dot(promote(p, q)..., α, β)`.  But it's difficult to get
# that working as a method.  I tried
#     dot(p::Union{Poly{T}, BernsteinPoly{T}}, q::Union{Poly{T}, BernsteinPoly{T}},
#         α = zero(T), β = one(T)) where {T<:Number} =
#         dot(promote(p, q)..., α, β)
# but that doesn't work.  It's probably not recommended to do that anyway.  Use `promote`
# explicitely where necessary.


# Examples: 
#     julia> B = BernsteinPoly(3, 2)
#     BernsteinPoly(3.0*x^2 - 3.0*x^3)
      
#     julia> Q = BernsteinPoly(4, 3)
#     BernsteinPoly(4.0*x^3 - 4.0*x^4)
      
#     julia> using LinearAlgebra
      
#     julia> dot(B, Q)
#     0.07142857142857142
      
#     julia> norm(B)
#     0.29277002188455997


end # module
