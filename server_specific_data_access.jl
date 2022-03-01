################################################################################################
## Library Dependencies
using MAT, SparseArrays, FFTW, LinearAlgebra

################################################################################################
## Working Directory Initialization based on the server name:
if server_name == "personal"
    # Working Directory Initialization and changing to the working directory: (PERSONAL)
    workdir = "C:\\Users\\shett\\Documents\\Dissertation\\my-research\\kernelversion7JVer";
    cd(workdir);
    # Location of the data:
    datadir = "G:\\My Drive\\GDrive University Files\\Dissertation Files\\MATLAB_DellXPS\\Datasets";
    # Directory for outputs:
    scratch = workdir;
    system = "personal";

elseif server_name == "diophantus"
    # Working Directory Initialization and changing to the working directory: (DIOPHANTUS)
    workdir = "/diophantus1/data/users/gauravna/matlabThesis/kernelversion7J";
    cd(workdir);
    # Location of the data:
    datadir = "/diophantus1/data/users/gauravna/data-files";
    # Directory for outputs:
    scratch = workdir;
    system = "diophantus";

elseif server_name == "manchester"
    # Working Directory Initialization and changing to the working directory: (MANCHESTER)
    workdir = "/manchester1/gauravna/Thesis/Codes/slavakis-gaurav/kernelversion7J";
    cd(workdir);
    # Location of the data:
    datadir = "/manchester1/gauravna/Thesis/Codes/data-files";
    # Directory for outputs:
    scratch = workdir;
    system = "manchester";

elseif server_name == "ccr" 
    # Working Directory Initialization and changing to the working directory: (CCR)
    workdir = "/projects/academic/kslavaki/gauravna/Thesis/kernelversion7JVer";
    cd(workdir);
    # Location of the data:
    datadir = "/projects/academic/kslavaki/gauravna/Thesis/Datasets";
    # Directory for outputs:
    scratch = workdir;
    system = "CCR";

end

################################################################################################
# MAT Variable Extracion (Mask matrix):
reader = matopen(joinpath(datadir, "Mask", maskname));
data = read(reader);
close(reader);
if issparse(data["Mask"])
    Mask = Matrix(data["Mask"]);
else
    Mask = data["Mask"];
end

###############################################################################################
## Initialization of the kspace data:
## MAT Variable Extracion :
reader = matopen(joinpath(datadir, dataname));
data = read(reader);
close(reader);
if usamp_type == "retrospective"
    # (ImageData (Image Domain data), Navigator Data):
    Ynav = data["Ynav"];
    ImageData = data["ImageData"];

    # Image Parameters initilization:
    if length(size(ImageData)) == 3
        (Np, Nf, Nfr) = size(ImageData);
        Nc = 1;
    else 
        (Np, Nf, Nfr, Nc) = size(ImageData);
    end
    Nnav = trunc(Int, size(Ynav, 1)/Nf);

    # Mask reshape: 
    if length(size(Mask)) == 3
        Mask = reshape(Mask, Np*Nf, Nfr);
    end

    # Mask Type Cast:
    Mask = .!iszero.(Mask);

    # k-Space Retrospective Undersampling:
    ImageData = ImageData/maximum(abs.(ImageData));
    Y = fft(ImageData, [1 2]);    
    Y = reshape(Y, (Np*Nf, Nfr));
    Y = Mask.*Y;

elseif usamp_type == "prospective"
    ## MAT Variable Extracion (ImageData (Image Domain data), Navigator Data):
    reader = matopen(joinpath(datadir, dataname));
    data = read(reader);
    close(reader);
    Ynav = data["Ynav"];
    Y = data["Y"];

    # Image Parameters initilization:
    if length(size(Y)) == 3
        (Np, Nf, Nfr) = size(Y);
        Nc = 1;
    elseif length(size(Y)) == 4
        (Np, Nf, Nfr, Nc) = size(Y);
    end
    Nnav = trunc(Int, size(Ynav, 1)/Nf);
    # Mask reshape: 
    if length(size(Mask)) == 3
        Mask = reshape(Mask, Np*Nf, Nfr);
    end

    # Mask Type Cast:
    Mask = .!iszero.(Mask);

    for i = 1:Nc
        Y[:,:,:,i] = Y[:,:,:,i]/(maximum(abs.(Y[:,:,:,i]))*Np*Nf);
    end
    
    # k-space Vectorization: (As the kspace is acquired from scanner; It is already undrersampled)
    Y = reshape(Y, Np*Nf, Nfr, Nc);
    
end

# Empty variables:
data = nothing;
reader = nothing;
###############################################################################################
