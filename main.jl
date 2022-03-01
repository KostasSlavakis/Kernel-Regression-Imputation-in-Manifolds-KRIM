# ###############################################################################################
# STEP 0: Global Variales:
# (Describing the type of data, sampling trajectory in use)
@__DIR__
server_name = "personal";
dataname = "INVIVOCARD-200256256.mat";
maskname = "Mask200256256-RxR.mat";
usamp_type = "retrospective";
#server_name = "personal";
#dataname = "PROSPECTIVE2-156192300.mat";
#maskname = "Mask156192300-PxC.mat";
#usamp_type = "prospective";

obsNo = 101;

println("Server name: ", server_name);
println("Data name: ", dataname);
println("Mask Name: ", maskname);
println("Sampling Method: ", usamp_type);
println("Observation Number: ", obsNo);
#println("Parameter File: ",  ARGS[5]);=-0962

k1 = findfirst(isequal('-'), dataname);
namedata = dataname[1:(k1-1)];
k1 = findfirst(isequal('-'), maskname);
k2 = findfirst(isequal('.'), maskname);
maskId = maskname[(k1+1):(k2-1)];

################################################################################################
#= Step 0: Including code library of functions required for
        execution. Landmark, Kernel, RSE and Reconstruction
        based library.
=#
using MAT, LinearAlgebra
include("./Lib/landmark_library.jl")
include("./Lib/kernel_library.jl")
include("./Lib/dimension_reduction_library.jl")
include("./Lib/mri_analysis_library.jl")
include("./Lib/mri_reconstruction_library.jl") 

################################################################################################
#= Step 0: Working, Data, Output Directory Initialization & MAT file extrseraction: 
    (requires the file 'server_specific_data_access.m')
    The following code should make the undersampled 
    (retrospective or prospective) 
    k-space data {Y}, navigator (centra kspace) data {Ynav} with dimensions 
    Np, Nf, Nfr, Nnav and the mask trajectory (Mask) available.
    
    Outputs Available: Y, Ynav, Mask, Np, Nf, Nfr, Nnav, 
                       ImageData (if retrospective undersampling)
=#
include("./server_specific_data_access.jl");
println("Data Initialization Complete ...");

################################################################################################
# Step 0: Parameter Initialization:
include("parameter_setup_PRO_0.jl")
println("Step 0: Parameter Initialization Complete ..."); println()

################################################################################################
#= Step 1: Using the Navigator Data extract landmark points which can be used for 
        kernel construction. (Strategies: Random & Max-min algorithms)
=#
## Initializations for required matrices ():
X = zeros(ComplexF64, Np*Nf, Nfr);
D = zeros(ComplexF64, Np*Nf, noK*d);
B = zeros(ComplexF64, noLandmark*noK, Nfr);
Khat = zeros(ComplexF64, d*noK, noLandmark*noK);
K = zeros(ComplexF64, noLandmark, noLandmark, noK);
Λ = zeros(ComplexF64, Nnav*Nf, noLandmark);
errConv = zeros(Float64, param.noIteration);

## Initializtion if it is multi-channel problem:
if Nc > 1
    X1 = zeros(ComplexF64, Np*Nf, Nfr, Nc);
    D1 = zeros(ComplexF64, Np*Nf, noK*d, Nc);
    B1 = zeros(ComplexF64, noLandmark*noK, Nfr, Nc);
    K1 = zeros(ComplexF64, noLandmark, noLandmark, noK, Nc);
    Khat1 = zeros(ComplexF64, d*noK, noLandmark*noK, Nc);
    Λ1 = zeros(ComplexF64, Nnav*Nf, noLandmark, Nc);
end

## Task Commencement:
for i = 1:Nc
    println("Task for Nc number: ", i, " commences ...");
    if Nc == 1
        global Λ = landmarkExtraction(Ynav, noLandmark, ltype);
    else
        global Λ = landmarkExtraction(Ynav[:,:,i], noLandmark, ltype);
    end 
    println("Step 1: Landmark Extraction Complete ..."); println()

    ################################################################################################
    #= Step 2: Kernel Consrtuctions using the identified landmark points.
            Strategies: Single Kernels - Gaussian, Linear, Polynomial
                        Multi Kernels - Combo of the above mentioned Kernels
                        Kernels constructed for both real/complex sensor data
    =#
    maxK = maximum(abs.(Λ));
    ldiv!(maxK, Λ);
    global K = MultiKernelConstruct(Λ, ktype);
    println("Step 2: Kernel Constrution Complete ..."); println()

    ################################################################################################
    #= Step 3: Dimension Reduction using the Robust Sparse Embedding 
            (using the code written in MATLAB)
    =#
    global Khat = DimReductionKernelRSE(K, dparams);
    println("Step 3: Dimension Reduction Complete ..."); println()

    ################################################################################################
    #= Step 4: Reconstruction framework to determine D. B using the 
            FMHSDM + FLEXA optimization frameworks:
    =#
    println("MRI Reconstrution Commencing ....")
    if usamp_type == "retrospective"
        @time global X, D, B, errConv = MriReconMKBiLMDM(Y, Khat, Mask, param, ImageData);
    elseif usamp_type == "prospective"
        if Nc != 1
            @time global X, D, B, errConv = MriReconMKBiLMDM(Y[:,:,i], Khat, Mask, param, [-1.0]);
        else
            @time global X, D, B, errConv = MriReconMKBiLMDM(Y[:,:,1], Khat, Mask, param, [-1.0]);
        end
    end
    println("Step 4: MRI Reconstruction Complete ..."); println()

    if Nc > 1
        global X1[:,:,i] = X;
        global D1[:,:,i] = D;
        global B1[:,:,i] = B;
        global K1[:,:,:,i] = K;
        global Khat1[:,:,i] = Khat;
        global Λ1[:,:,i] = Λ;
    end
    println("Tasks for Nc = ", i, " finished.");
end
################################################################################################
# Error Analysis and logging parameters:
if usamp_type == "retrospective"
    reconImageData = X;
    reconImageData1 = reshape(reconImageData, Np, Nf, Nfr);
    reconImageData = D*Khat*B; 
    reconImageData2 = reshape(reconImageData, Np, Nf, Nfr);
    err1 = checkError(ImageData, reconImageData1);
    err2 = checkError(ImageData, reconImageData2);
end

################################################################################################
# Display Code Attribtes:
println("Server Name: ", server_name)
println("Data Name: ", dataname)
println("Mask Rate: ", maskId)
println("Sampling Type: ", usamp_type); println()

println("LANDMARK SELECTION ATTRIBUTES:")
println("No. of landmark points: ", noLandmark);
println("Method of landmark selection: ", ltype); println();

println("KERNEL GENERATION ATTRIBUTES")
dump(ktype); println()

println("DIMENSION REDUCTION ATTRIBUTES")
dump(dparams); println()

println("RECONSTRUCTION ATTRIBUTES:")
dump(param); println()

#if usamp_type == "retrospective"
#    println("The Error in reference to X* : ", err1); println()
#    println("The Error in reference to DKB : ", err2); println()
#end

################################################################################################
# Save the output files:
filename = string(system, "_KV_", namedata,  "_", maskId, "_",  obsNo, ".mat");
outputdir = joinpath(scratch, "kernel_out")
if ~isdir(outputdir)
    mkdir(outputdir)
end
filename = joinpath(outputdir, filename);
if Nc == 1
matwrite(filename, Dict(
    "D" => D,
    "Khat" => Khat,
    "B" => B,
    "L" => Λ,
    "K" => K,
    "X" => X,
    "errConv" => errConv
));
else
    matwrite(filename, Dict(
    "D" => D1,
    "Khat" => Khat1,
    "B" => B1,
    "L" => Λ1,
    "K" => K1,
    "X" => X1
));
end
################################################################################################

