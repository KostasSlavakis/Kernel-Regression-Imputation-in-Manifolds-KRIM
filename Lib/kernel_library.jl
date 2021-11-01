################################################################################################
## Kernel Construction Codes:
################################################################################################
## Library Dependencies
# None Required so far (7/12/2020)

## Kernel Parameters:
mutable struct kernelParams
    degree :: Int64                     # Polynomial kernel, degree of the kernel
    intercept :: Float64                # Polynomial kernel, constant term 
    alpha :: Float64                    # Exponential kernel (gaussian and laplacian), 
                                        #    Std. Deviation related parameter
    kernelParams() = new(2, 0, 1)       # Constructor to set default values
end

################################################################################################
## FUNCTIONS:
#= FUNCTIONS FOR COMPUTING KERNELS:
Calls the wrapper function 
Input:  data = data matrix with observations arranged along the columns  
        ktype = Array of Tuples with first element as the name of the kernel to be instructed, 
            and second element is the corressponding kernel parameters of type 'kernelParams'
Output: kernel =  Stacked Kernel matrix along the z-direction arranged along the diagonal of a bigger matrix.
=#
function MultiKernelConstruct(data::Array{T, 2}, ktype::Array) where {T<:Union{ComplexF64,Float64}}
    noKernel = length(ktype);
    N2 = size(data, 2);
    kernel = Array{eltype(data), 3}(undef, N2, N2, noKernel);
    for (index, type) in enumerate(ktype)  
        K = KernelConstruct(data, type[2], type[1]);
        kernel[:,:,index] = K; 
    end
    return kernel;
end

#= FUNCTIONS FOR COMPUTING KERNELS:
Wrapper function to generate different type of kernels. Currently available kernels
are : Gaussian, Laplacian, Polynomial kernels described on both Real and Complex data matrices.
Input:  data = data matrix with observations arranged along the columns  
        params = kernel parameters depending on what type of kernel is required, refer 
            the struct definition for the same
        type = String dtermining the type of the kernel to be generated
Output: kernel =  kernel matrix described above.
=#
function KernelConstruct(data::Array{T, 2}, params::kernelParams, type::String="gaussian") where {T<:Union{ComplexF64,Float64}}
    if type == "gaussian"
        kernel = ExponentialKernelComplex(data, params.alpha, type)
    elseif type == "laplacian"
        kernel = ExponentialKernelComplex(data, params.alpha, type)
    elseif type == "polynomial"
        kernel = PolynomialKernelComplex(data, params.degree, params.intercept)
    elseif type == "linear"
        kernel = PolynomialKernelComplex(data, 1, 0.0)
    end
    return kernel;
end

#= FUNCTIONS FOR COMPUTING KERNELS:
Function 2: Generates a polynomial kernel with degree and a fixed intercept value.
Input: data = data matrix with observations arranged along the columns  
       degree = (type int) degree of the polynomial
       intercept = (type int) intercept value for the polynomial
Output: kernel =  kernel matrix described above.
=#
function PolynomialKernelComplex(data::Array{T,2}, degree::Int64=1, intercept::Float64=0) where {T<:Union{ComplexF64,Float64}}
    # kernel = (<x,y> + c)^d
    println("Welcome to Polynomial Kernel")
    kernel = data' * data .+ intercept;
    kernel = kernel.^degree; 
    return kernel;
end

#= FUNCTIONS FOR COMPUTING KERNELS:
Function 1: Generate an exponential based kernel (Gaussian or Laplacian).
Input: data = (complex or real matrix) data matrix with observations arranged along the columns 
       α = (type float) coefficient for exponential kernels
       type = (string) type of the kernels [valid inputs: 'gaussian' and 'laplacian']           
Output: kernel matrix (symmetric matrix)
=#
function ExponentialKernelComplex(data::Array{T, 2}, α::Float64=1, type::String="gaussian") where {T<:Union{ComplexF64,Float64}}
    # kernel = exp(-α||x - y||^2)
    println("Welcome to Exponential Kernel Construction")
    N = size(data, 2);
    if type == "gaussian"
        power = 2;
    elseif type == "laplacian"
        power = 1;
    end
    kernel = typeof(data)(undef, N, N);
    for i = 1:N
        kernel[:,i] = [sum((data[:, i] - conj(data[:,j])).^power) for j = 1:N];
    end
    kernel = @. exp(-α * kernel);
    #conj!(kernel);
    return kernel
end