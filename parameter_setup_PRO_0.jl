################################################################################################
## Parameters for Landmark Extraction (Step 1):
noLandmark = 100;
ltype = "maxmin";

################################################################################################
## Initializing the kernel parameters for Kernel construction:
kparams1 = kernelParams()
kparams2 = kernelParams(); kparams2.alpha = 0.5;
kparams3 = kernelParams(); kparams3.degree = 1; kparams3.intercept = 0.5;
ktype = [("gaussian", kparams2)];
#ktype = [("linear", kparams1), ("gaussian", kparams2), ("polynomial", kparams3)];

################################################################################################
## Dimension Reduction using RSE:
λw = 1e0; 
αw = 0.5;
d = 4;
threshold = 1e-4;
noIteration = 1e5;
dparams = dimRedParams(λw, αw, d, threshold, noIteration);

################################################################################################
## Reconstruction framework:
Np = Np;
Nf = Nf;
Nfr = Nfr;
ζ = 1e-1;
γ = 0.9;
λ1 = 1e-1; 
λ2 = 8e-1; 
λ3 = 8e-2;
CD = 1e0;
τD = 1e-6;
τB = 1e-5;
αD = 0.5;
αB = 0.5;
noK = length(ktype);
threshold = 1e-8;
noIteration = 2500;
thresholdInner = 1e-5;
noIterationInner = 250;
param = OptimizerParams(Np, Nf, Nfr, ζ, γ, λ1, λ2, λ3, CD, τD, τB, αD, αB, noK, threshold, noIteration, thresholdInner, noIterationInner);

################################################################################################