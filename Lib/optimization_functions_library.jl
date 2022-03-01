################################################################################################
## Optimization Functions: (Gradient and Proximal Algorithms)
# Library Dependencies:
using LinearAlgebra

#=
FUNCTION: Proximal Operator fot l1-norm minimization:
INPUT:  X = Matrix, type Real or Complex 
        lambda = thresholding paramtere
OUTPUT: Y = Soft Thresholded Output Matrix
=#
function softThresholdingProximal!(Y::Matrix{T}, X::Matrix{T}, λ::Float64) where {T<:Union{ComplexF64,Float64}} 
    Y .= @. X * (1 - λ/max(abs(X), λ));     # [Y]_ij = [X]_ij (1 - λ/max(|[X]_ij|, λ)); 
end

#=
FUNCTION: Proximal Operator fot l1-norm minimization:
INPUT:  X = Matrix, type Real or Complex
        lambda = thresholding paramtere
OUTPUT: Y = Soft Thresholded Output Matrix
=#
function boundConstraintProximal!(Y::Matrix{T}, X::Matrix{T}, Cx::Float64) where {T<:Union{ComplexF64,Float64}}
    for i = 1:size(X, 2)
        Y[:,i] .= X[:,i] * (Cx/max(Cx, norm(X[:,i])));
    end
    return Y;
end
################################################################################################
