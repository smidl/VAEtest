export CDirac

"""
    CDirac{T}

Conditional Dirac distribution that maps an input of `zlength` to its mean of `xlength`.

# Arguments
- `xlength::Int`: length of mean
- `zlength::Int`: length of condition
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)

# Example
```julia-repl
julia> p = CDirac(3, 2, Dense(2, 3))
CDirac{Float64}(xlength=3, zlength=2, mapping=Dense(2, 3))

TODO  ...

julia> mean_var(p, ones(2))
([-0.339991, -0.061213, -0.769473] (tracked), [1.0, 1.0, 1.0])

julia> rand(p, ones(2))
Tracked 3×1 Array{Float64,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CDirac{T} <: AbstractCPDF{T}
    xlength::Int
    zlength::Int
    mapping
end

function CDirac(xlength, zlength, mapping::Function, T=Float32)
    CDirac{T}(xlength, zlength, mapping);
end

# make sure that constructor is called with parametric type by mapleaves
Flux.children(m::CDirac) = (m.xlength, m.zlength, m.mapping)

function mean_var(p::CDirac{T}, z::AbstractArray) where T
    μ = p.mapping(z)
    σ2 = fill!(similar(μ, xlength(p)), 0)
    return μ, σ2
end

function Base.show(io::IO, p::CDirac{T}) where T
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CDirac{$T}(xlength=$(p.xlength), zlength=$(p.zlength), mapping=$e)"
    print(io, msg)
end

function rand(p::CDirac{T}, z::AbstractArray) where T
    p.mapping(z)
end
function mean(p::CDirac{T}, z::AbstractArray) where T
    p.mapping(z)
end
