################################################################################################
## Reconstruction Tasks:
################################################################################################
## Library Dependencies:
using Random, LinearAlgebra, FFTW
include("./mri_support_functions_library.jl")

# Recovery Task Parameters:
mutable struct OptimizerParams
    Np :: Int64                         # Phase Encoding Lines
    Nf :: Int64                         # Phase Frequency Lines
    Nfr :: Int64                        # Number of Frames
    ζ :: Float64                        # Decay Rate parameter for Main Reconstruction Task
    γ :: Float64                        # Step Size parameter for Main Reconstruction Task
    λ1 :: Float64                       # Regularization parameter 
    λ2 :: Float64                       # Regularization parameter
    λ3 :: Float64                       # Regularization parameter
    CD :: Float64                       # Bounding Constant parameter on Matrix U
    τD :: Float64                       # Sub Optimization Tasks Parameter for || D - Dn ||
    τB :: Float64                       # Sub Optimization Tasks Parameter for || B - Bn ||
    αD :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    αB :: Float64                       # Step Size Parameter for Sub Optimization Tasks
    noK :: Int64                        # Number of kernels employed
    threshold :: Float64                # Threshold of loss criterion for the Main Reconstruction Task                
    noIteration :: Int64                # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Main Reconstruction Task
    thresholdInner :: Float64           # Threshold of loss criterion for the Sub Optimization Tasks
    noIterationInner :: Int64           # Itertion Count Cut off in convergence criteria is 
                                        #    not met for the Sub Optimization Tasks
end


function MriReconMKBiLMDM(Y::Array{T1, 2}, K::Array{T1,2}, M::BitArray{2}, params::OptimizerParams, ImageData::Array{T2}) where {T1, T2 <:Union{Float64, ComplexF64}}
    ## Matrix Size initialization:
    rng = MersenneTwister(1234);
    Np = params.Np;
    Nf = params.Nf;
    Nfr = params.Nfr;
    Nk = Np*Nf; 
    (d, Nl) = size(K);
    Nker = Integer(Nl/params.noK);
    if size(ImageData, 1) != 1  
        img = reshape(ImageData, Nk, Nfr);
        imgLoss = -1*ones(params.noIteration);
        flag = 0;
    else
        flag = 1;
    end

    ## Reconstructions Hyperparameter Initialization:
    loss = 1e0;
    noIter = 0;
    γ = params.γ;
    ζ = params.ζ;
    λz = params.λ3/params.λ2;

    ## Parameter, Matrix Values to save computation in time in loops:
    Cz = λ2*Nfr;
    Cx = 1/(1 + Cz);
    Mc = .!M; 
    
    #Fourier Transform Plan
    pf  = plan_fft(zeros(Np, Nf, Nfr), [1 2], flags=FFTW.MEASURE);
    pft = plan_fft(zeros(Nk, Nfr), 2, flags=FFTW.MEASURE);
    
    ## STEP 5: Low-resolution FFT Reconstruction:
    Xlow = ifft2(Y, pf);

    ## STEP 6: Pterm Initialization:
    # Pterm = I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}
    Pterm = Matrix{ComplexF64}(I, Nker, Nker) - ldiv!(Nker, ones(ComplexF64, Nker, Nker));
    # Aterm = (1/Nker) * 1_{Nker, Nfr};
    Aterm = ldiv!(Nker, ones(ComplexF64, Nker, Nfr));

    ## STEP 7: Random Initialization:
    # X0:
    Xn = copy(Xlow);
    # D0:
    Dn = randn(rng, ComplexF64, Nk, d);
    DnK = zeros(ComplexF64, Nk, Nl);
    mul!(DnK, Dn, K);
    # B0:
    Bn = DnK \ Xn;
    #Bn = randn(rng, ComplexF64, Nl, Nfr);
    KBn = zeros(ComplexF64, d, Nfr);
    mul!(KBn, K, Bn);
    # Z0:
    Zn =  similar(Xlow);
    mul!(Zn, pft, Xn);
    #Zn = randn(rng, ComplexF64, Nk, Nfr); #Random Initialization
    # Others:
    Xhat_prev = copy(Xn);
    Xhat_DKB = similar(Xn); # NEW EDITION
    Xhat = similar(Xlow);
    Yhat = similar(Y);
    Dhat = similar(Dn);
    Bhat = similar(Bn);
    Zhat = similar(Zn);
    Aux = similar(Xn);
    KhDhDK = zeros(ComplexF64, Nl, Nl);
    KBBhKh = zeros(ComplexF64, d, d);
    XBhKh = zeros(ComplexF64, Nk, d);
    KhDhX = zeros(ComplexF64, Nl, Nfr);

    ## STEP 8: Reconstruction Task (Beginning of while loop):
    while (loss > params.threshold && noIter < params.noIteration)
        ## STEP 10: Step-size Initialization:
        γ = γ * (1 - ζ*γ);

        ## STEP 11: Pre-requisites:
        if (noIter != 1)
            mul!(DnK, Dn, K);                               # Product DnK
            mul!(KBn, K, Bn);                               # Product KBn
        end 
        DnKh = DnK';                                        # Hermitian DnKh = K'Dn' 
        KBnh = KBn';                                        # Hermitian KBnh = Bn'K'
              
        ## STEP 12: Obtain Xhat:
        mul!(Xhat, DnK, Bn);                                # Xhat = DnKBn
        ldiv!(Aux, pft, Zn);                                # Aux = invFt(Z)
        Aux = Cz .* Aux;                                    # Aux = λ2Nfr * invFt(Z)
        Xhat += Aux;                                        # Xhat = DnKBn + λ2Nfr * invFt(Z)
        Xhat = Cx .* Xhat;                                  # Xhat = 1/(1 + λ2Nfr) [DnKBn + λ2Nfr * invFt(Z)]
        Yhat = fft2(Xhat, pf);                              # Yhat = F(Xhat)
        Yhat = Mc .* Yhat;                                  # Yhat = Sc(F(Xhat))
        Yhat += Y;                                          # Yhat = Y + Sc(F(Xhat))
        Xhat = ifft2(Yhat, pf);                             # Xhat = invF(Y + Sc(F(Xhat)))

        ## STEP 13: Obtain Dhat:
        mul!(KBBhKh, KBn, KBnh);                            # Product KBBhKh = KBnBn'K'
        mul!(Dhat, Dn, KBBhKh);                             # Dhat = DnKBnBn'K'
        mul!(XBhKh, Xn, KBnh);                              # Product XnBn'K'
        Dhat = Dhat - XBhKh;                                # Dhat = DnKBnBn'K' - XnB'K'
        Dhat = params.τD .* Dhat;                           # Dhat = τD (DnKBnBn'K' - XnB'K')
        Dhat = Dn - Dhat;                                   # Dhat = Dn - τD (DnKBnBn'K' - XnB'K')
        boundConstraintProximal!(Dhat, Dhat, params.CD);    # Dhat = Prox(D)

        ## STEP 14: Obtain Bhat:
        mul!(KhDhX, DnKh, Xn);                              # Product KhDhX = K'D'X
        mul!(KhDhDK, DnKh, DnK);                            # Product KhDhDK = K'Dn'DnK
        Lb = opnorm(KhDhDK, 2) + params.τB;                 # Lb = ||K'Dn'DnK||_2 + τB
        solveforBhat!(Bhat, KhDhDK, KhDhX, Bn, Lb, Pterm, Aterm, params);

        ## STEP 15: Obtain Zhat: 
        mul!(Aux, pft, Xn);                                 # Aux = Ft(Xn)
        softThresholdingProximal!(Zhat, Aux, λz);           # Zhat = Shrinkage operatore on Ft(X)
                                                            #        attributed l1-norm on Zn

        ## STEP 16: Update:
        axpby!(γ, Xhat, (1-γ), Xn);                         # Xn+1 = γXhat + (1-γ)Xn
        axpby!(γ, Dhat, (1-γ), Dn);                         # Dn+1 = γDhat + (1-γ)Dn
        axpby!(γ, Bhat, (1-γ), Bn);                         # Bn+1 = γBhat + (1-γ)Bn
        axpby!(γ, Zhat, (1-γ), Zn);                         # Zn+1 = γZhat + (1-γ)Zn

        ## Termination Criteria Parameters:
        noIter += 1;
        Xhat_DKB = Dn * K * Bn;
        loss = norm(Xhat_DKB - Xhat_prev)/norm(Xhat_prev);
        Xhat_prev = copy(Xhat_DKB);
        if flag ==0
            imgLoss[noIter] = norm(img - Xhat_DKB)/norm(img);
            println("Main Task Iteration Number ", noIter, ": Loss Value ", loss, "; NRMSE ", imgLoss[noIter]);
        else
            println("Main Task Iteration Number ", noIter, ": Loss Value ", loss, ".");
        end
    end
    println("Main Task Terminated at Iteration ", noIter, " for Loss Value ", loss, ".");
    if flag != 0
        imgLoss = -1;
    end
    # show(Xn[1:10, 1:10])
    return Xn, Dn, Bn, imgLoss;
end

# Optimisation for Bn:
function solveforBhat!(Bhat::Array{T,2}, KhDhDK::Array{T,2}, KhDhX::Array{T,2}, Bn::Array{T,2}, Lb::Float64, pT::Array{T,2}, aT::Array{T,2}, params::OptimizerParams) where T <: Union{ComplexF64, Float64}
    ## Hyperparameter Initialization:
    lossB = 1;
    noIterB = 0;
    λ = 1.98 * (1 - params.αB)/Lb;                                          # λ = 0.99* 2[1-α]/Lb
    λλ1 = λ*params.λ1;                                                      # λλ1 Product
    α_1 = 1-params.αB;                                                      
    Nker = size(KhDhDK, 1);
    Nl = Integer(Nker/params.noK);

    ## Matrix Initialization:
    TH0 = similar(Bn);
    TH1 = similar(Bn);
    TαH0 = similar(Bn);

    # PT_grad = K'Dn'DnK + τB*I_{Nker, Nker}
    pT_grad = KhDhDK + (params.τB .* Matrix{ComplexF64}(I, Nker, Nker));
    #  AT_grad = K'Dn'X + τB*Bn  
    aT_grad = KhDhX + (params.τB .* Bn);
    # Gradient Function: 
    # ∇g(B) = (K'Dn'DnK + τB*I_{Nker, Nker}) * B - K'Dn'X - τB*Bn
    # ∇g(B) = K'Dn'DnKB + τB*B - K'Dn'X - τB*Bn
    @inline gradB(X::Matrix{ComplexF64}) = pT_grad * X - aT_grad;

    ## STEP 5: Initialization:
    H0 = copy(Bn);
    H1 = similar(H0);

    ## STEP 6: Computing T(B) and Tα(B)
    # pT = I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}
    # aT = (1/Nker) * 1_{Nker, Nfr}
    # T(H0) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H0 + (1/Nker) * 1_{Nker, Nfr}
    for i = 1:params.noK
        xStart = (i-1)*Nl + 1;
        xEnd = i*Nl;
        TH0[xStart:xEnd, :] = pT * H0[xStart:xEnd, :] + aT; 
    end
    axpby!(α_1, H0, params.αB, TH0);                            # Tα(H0) = αT(H0) + (1- α)H0   
    
    ## STEP 7: H_(1/2) Update
    gradH0 = λ .* gradB(H0);                                    # λ∇g(H0) = λ(K'Dn'DnKH0 + τB*H0 - K'Dn'X - τB*Bn)
    H1_2 =  TH0 - gradH0;                                       # H1_2 = T(H0) - λ∇g(H0)

    ## STEP 8: H_1 Update
    softThresholdingProximal!(H1, H1_2, λλ1);                   # [H1]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))

    ## STEP 9: While Loop
    while (lossB > params.thresholdInner && noIterB < params.noIterationInner)
        ## STEP 10: H_(k+1/2) update
        # T(H1) = (I_{Nker, Nker} - (1/Nker) * 1_{Nker, Nker}) * H1 + (1/Nker) * 1_{Nker, Nfr}
        for i = 1:params.noK
            xStart = (i-1)*Nl + 1;
            xEnd = i*Nl;
            TH1[xStart:xEnd, :] = pT * H1[xStart:xEnd, :] + aT; 
        end
        # ∇g(H1) = λ [K'Dn'DnKH1 + τB*H1 - K'Dn'X - τB*Bn]
        gradH1 = λ .* gradB(H1);
        # H1_2 = H1_2 + T(H1) - ∇g(H1) - Tα(H0) + ∇g(H0)
        H1_2 += TH1 - gradH1 - TH0 + gradH0;  
        TH0 = copy(TH1);
        H0 = copy(H1);
        gradH0 = copy(gradH1);
        axpby!(α_1, H0, params.αB, TH0);

        ## STEP 11: H_(k+2) update: [H2]_ij = [H1_2]_ij (1 - λλ1/max(|[H1_2]_ij|, λλ1))
        softThresholdingProximal!(H1, H1_2, λλ1);

        ## Termination Criteria Update:
        noIterB += 1;
        lossB = norm(H1 - H0, 2)/norm(H0, 2);
        #if noIterB <= 10
        #    println("Solve for B Task Iteration Number ", noIterB, ": Loss Value ", lossB, ".");
        #end
    end
    #println("Solve for B Terminated at Iteration ", noIterB, " for Loss Value ", lossB, ".");
    Bhat .= H1;
end