################################################################################################
## Importing Libraries required for the tasks:
using Distances, Random

################################################################################################
## Landmark Extraction Related Codes:
################################################################################################

#= LANDMARK POINT SELECTION USING MAXMIN/RANDOM ALGORITHM:
Detect landmark points using either a random  or Maxmin algorithm:
Inputs: data (real or complex matrix) - Data matrix with columns as the data points 
        noLandmark (int) - The number of landmark points to be identified from the data matrix
        type (string) - A String either "random" or "maxmin" determining the algorithm to use
Output: lambda (same as data) - Contains columns of data identified as the landmark points
=#
function landmarkExtraction(data::Array{T,2}, noLandmark::Int64, type::String = "maxmin") where {T<:Union{ComplexF64,Float64}}
    if type == "maxmin"
        landmark = maxminLandmark(data, noLandmark);
    elseif type == "random"
        N = size(data, 2);
        lidx = rand(1:N, noLandmark);
        landmark = data[:, lidx];
    end
    return landmark;
end

#= LANDMARK POINT SELECTION USING MAXMIN ALGORITHM:
Detect landmark points using the Maxmin algorithm:
Inputs: data (complex matrix) - Data matrix with columns as the data points 
        noLandmark (int) - The number of landmark points to be identified from the data matrix
Output: lambda (same as data) - Contains columns of data identified as the landmark points
=#
function maxminLandmark(data::Array{T, 2}, noLandmark::Int64) where {T<:Union{ComplexF64,Float64}}
    #print("Welcome to maxminLandmark..")
    ## Number of data points (columns) in the data matrix:
    N = size(data, 2);
    ## Random inital points to start with:
    seed = 2;
    ## Extracting seed points for the maxmin algorithm:
    lidx = rand(1:N, seed);
    lambda = data[:, lidx];
    ## Compute pairwise distances and :
    distance = minimum(pairwise(Minkowski(2), lambda, data, dims=2), dims=1);
    for i = (seed+1):noLandmark
        ## select the landmark point farthest from the already selected group of landmark points:
        maxDistance = maximum(distance);
        S = vec(distance .== maxDistance);
        ## add the newly identifed landmark point to the data list:
        lambda =  hcat(lambda, data[:, S]);
        ## compute the distance of the selected landmark point from the remaining points:
        distance1 = pairwise(Minkowski(2), reshape(lambda[:, i], length(lambda[:, i]), 1), data, dims=2);
        distance = min.(distance, distance1);
    end
    ## return the identified landmark data points:
    return lambda
end
################################################################################################