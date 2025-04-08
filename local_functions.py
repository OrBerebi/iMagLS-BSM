import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import AK_ILD_pytorch as ak
import sofar as sf
import platform
import sys
import os
import scipy
from tqdm.notebook import tqdm
from torchinfo import summary
import notch_mask


def import_matlab_data(data_path,device,shutup):
    """
    Import and process variables from a MATLAB .mat file into PyTorch tensors.

    Parameters:
    - data_path (str): File path to the MATLAB .mat file.
    - device (torch.device): Target device (e.g., CPU or GPU) for PyTorch tensors.
    - shutup (bool): Flag to suppress detailed variable information printing.

    Returns:
    - output_dict (dict): Dictionary containing processed data:
        - Keys correspond to variable names.
        - Values are PyTorch tensors or processed data.

    Workflow:
    1. Loads MATLAB data from `data_path` using `scipy.io.loadmat`.
    2. Converts selected variables into PyTorch tensors and assigns to local variables.
    3. Optionally prints detailed variable information if `shutup` is False.
    4. Returns a dictionary (`output_dict`) containing all processed variables.
    """
    mat = scipy.io.loadmat(data_path)

    c_ls      = mat["c_LS"]
    c_ls      = torch.from_numpy(c_ls).to(device)
    c_mls      = mat["c_MLS"]
    c_mls      = torch.from_numpy(c_mls).to(device)

    p_ref_bsm_f = mat["p_ref_bsm"]
    p_ref_bsm_f = torch.from_numpy(p_ref_bsm_f).to(device)
    p_ref_bsm_f = torch.permute(p_ref_bsm_f,[2,0,1])
    p_ref_horiz_f = mat["p_ref_horizontal"]
    p_ref_horiz_f = torch.from_numpy(p_ref_horiz_f).to(device)
    p_ref_horiz_f = torch.permute(p_ref_horiz_f,[2,0,1])
    p_ref_lebedev_f = mat["p_ref_lebedev"]
    p_ref_lebedev_f = torch.from_numpy(p_ref_lebedev_f).to(device)
    p_ref_lebedev_f = torch.permute(p_ref_lebedev_f,[2,0,1])

    V_k_bsm = mat["V_k_bsm"]
    V_k_bsm = torch.from_numpy(V_k_bsm).to(device)
    V_k_horiz = mat["V_k_horiz"]
    V_k_horiz = torch.from_numpy(V_k_horiz).to(device)
    V_k_lebedev = mat["V_k_lebedev"]
    V_k_lebedev = torch.from_numpy(V_k_lebedev).to(device)

    omega_bsm        = mat["Omega_bsm"]
    omega_horizontal = mat["Omega_horizontal"]
    omega_lebedev    = mat["Omega_lebedev"]
    

    
    f_cut_magLS = mat["f_cut_magLS"][0,0]
    f_vec       = mat["f_vec"]
    nfft        = np.uint16((f_vec.shape[0]-1)*2)
    fs          = mat["fs"][0,0]
    N_high      = mat["N_high"][0,0]
    HRTFpath    = str(mat["HRTFpath"][0])
    arrayType   = str(mat["arrayType"][0])
    gamma       = np.rad2deg(mat["gamma"][0,0])
    
    if not(shutup):
        print("\n\n----------Variables------------")
        print("c_ls\t\t",c_ls.shape, "\t",c_ls.dtype)
        print("c_mls\t\t",c_mls.shape, "\t",c_mls.dtype)
        
        print("")
        print("p_ref_bsm_f\t",p_ref_bsm_f.shape, "\t",p_ref_bsm_f.dtype)
        print("p_ref_horiz_f\t",p_ref_horiz_f.shape, "\t",p_ref_horiz_f.dtype)
        print("p_ref_lebedev_f\t",p_ref_lebedev_f.shape, "\t",p_ref_lebedev_f.dtype)
        
        print("")
        print("V_k_bsm\t\t",V_k_bsm.shape, "\t",V_k_bsm.dtype)
        print("V_k_horiz\t",V_k_horiz.shape, "\t",V_k_horiz.dtype)
        print("V_k_lebedev\t",V_k_lebedev.shape, "\t",V_k_lebedev.dtype)

        print("")
        print("omega_bsm\t",omega_bsm.shape, "\t\t\t",omega_bsm.dtype)
        print("omega_horizontal",omega_horizontal.shape, "\t\t\t",omega_horizontal.dtype)
        print("omega_lebedev\t",omega_lebedev.shape, "\t\t\t",omega_lebedev.dtype)
        print("")
        
        print("f_cut_magLS\t",f_cut_magLS, "\t\t\t\t",f_cut_magLS.dtype)
        print("f_vec\t\t",f_vec.shape, "\t\t\t",f_vec.dtype)
        print("nfft\t\t",nfft, "\t\t\t\t",nfft.dtype)
        print("fs\t\t",fs, "\t\t\t\t",fs.dtype)
        print("N_high\t\t",N_high, "\t\t\t\t",N_high.dtype)
        print("")
        
        print("arrayType\t",arrayType)
        print("HRTFpath\t",HRTFpath)

        


    output_dict = {
        "c_ls": c_ls,
        "c_mls": c_mls,
        "p_ref_bsm_f": p_ref_bsm_f,
        "p_ref_horiz_f": p_ref_horiz_f,
        "p_ref_lebedev_f": p_ref_lebedev_f,
        "V_k_bsm": V_k_bsm,
        "V_k_horiz": V_k_horiz,
        "V_k_lebedev": V_k_lebedev,
        "omega_bsm": omega_bsm,
        "omega_horizontal": omega_horizontal,
        "omega_lebedev": omega_lebedev,
        "arrayType": arrayType,
        "f_cut_magLS": f_cut_magLS,
        "f_vec": f_vec,
        "nfft": nfft,
        "fs": fs,
        "gamma": gamma,
        "N_high": N_high,
        "HRTFpath": HRTFpath,
    }
    
    return output_dict


def BSM_binaural_repo(c, x, device='cpu',is_time=False):
    """
    Performs binaural spatial mapping using frequency-domain signals and optionally computes the time-domain representation.

    Parameters:
    - c (torch.Tensor): Complex-valued tensor of shape [num_channels, num_channels, num_freq_bins].
                        Contains binaural spatial transfer functions.
    - x (torch.Tensor): Complex-valued tensor of shape [num_channels, num_samples, num_freq_bins].
                        Contains input signals in the frequency domain.
    - device (torch.device, optional): Device where the tensors should be allocated (default: 'cpu').
    - is_time (bool, optional): Flag indicating whether to compute the time-domain signal (default: False).

    Returns:
    - torch.Tensor: If is_time=True, returns a tensor of shape [num_channels, num_samples, nFFT].
                    Contains the time-domain binaural signals.
                    Otherwise, returns a tensor of shape [num_channels, num_samples, num_freq_bins].

    Notes:
    - Assumes c and x are PyTorch tensors containing complex values.
    - If is_time=True, computes the time-domain signal using inverse FFT (irfft).
    - Returns p_binaural_f if is_time=False, otherwise returns p_binaural_t.
    """
    nFFT = 2 * (x.size(2) - 1)
    p_binaural_f = torch.zeros(2, x.size(1), x.size(2), dtype=torch.complex128, device=device)

    for f_idx in range(c.size(2)):
        x_idx = x[:, :, f_idx]
        c_idx = c[:, :, f_idx].conj().transpose(0, 1)  # Conjugate transpose
        p_binaural_f[:, :, f_idx] = torch.matmul(c_idx, x_idx)

    if is_time:
        p_binaural_t = torch.fft.irfft(p_binaural_f, n=nFFT, dim=2)  
        p_binaural_t = torch.roll(p_binaural_t, shifts=nFFT // 2, dims=2)
        p_binaural_t = torch.permute(p_binaural_t,[1,2,0]) # space x time x ears
        return p_binaural_t

    return p_binaural_f




class NN(torch.nn.Module):
    def __init__(self,M,freq_bins,first_idx):
        super().__init__()

        self.linear1 = torch.nn.Linear(M*2,M*2, dtype=torch.complex128)
        self.linear2 = torch.nn.Linear(freq_bins - first_idx, freq_bins- first_idx, dtype=torch.complex128)

        # Initialize weights for identity transformation
        torch.nn.init.eye_(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        torch.nn.init.eye_(self.linear2.weight)
        self.linear2.bias.data.fill_(0)

        self.M = M
        self.freq_bins = freq_bins
        self.first_idx = first_idx

    
    def forward(self, x):
        x_re  = torch.reshape(x,(2*self.M,self.freq_bins))
        
        # Separate the first bins (to bypass the network) and the rest
        first_bins     = x_re[:, :self.first_idx]       # Shape: [M, ears, first_bins]
        remaining_bins = x_re[:, self.first_idx:]   # Shape: [M, ears, rest_bins]

        # Apply the neural network to the remaining bins
        processed_bins = torch.transpose(remaining_bins, 0, 1)
        processed_bins = self.linear1(processed_bins)
        processed_bins = torch.transpose(processed_bins, 0, 1)
        processed_bins = self.linear2(processed_bins)

        
        
        # Concatenate the first  bins with the processed bins and skip fist bin
        output = torch.cat((first_bins, processed_bins), dim=1)
        output = torch.reshape(output,(self.M, 2 ,self.freq_bins))
        return output



def clc_e_nmse(p_ref,p_hat,norm_flag=True):
    #calc the absolute diffrence
    e_mse_space_freq_lr = torch.pow(torch.abs(p_ref - p_hat), 2) # ears x directions x nfft bins
    # norm and avarage over all directions ears and freqncies
    if norm_flag:
        p_ref_squared       = torch.pow(torch.abs(p_ref), 2)
        e_mse_space_freq_lr = e_mse_space_freq_lr / p_ref_squared                 # normalize
        e_mse_space_freq_lr = 10*torch.log10(e_mse_space_freq_lr)                 # calc dB error
        e_mse_freq_lr       = torch.mean(e_mse_space_freq_lr, dim = 1).squeeze()  # E[] over space
    else:
        e_mse_freq_lr  = torch.mean(e_mse_space_freq_lr, dim = 1).squeeze()       # E[] over space
    e_mse_freq     = torch.mean(e_mse_freq_lr,dim = 0).squeeze()                  # E[] over ears
    return e_mse_freq

def clc_e_mag(p_ref,p_hat,M_dB,norm_flag=True):
    #calc the magnitude absolute diffrence
    e_mag_space_freq_lr = torch.pow(torch.abs(torch.abs(p_ref) - torch.abs(p_hat)), 2)

    # mask the only posirive gradient bins
    if M_dB.numel() > 0: 
        M_linear = 10 ** (M_dB / 10)
        e_mag_space_freq_lr = e_mag_space_freq_lr * M_linear 
    
    # norm and avarage over all directions ears and freqncies
    if norm_flag:
        p_ref_squared       = torch.pow(torch.abs(p_ref), 2)
        e_mag_space_freq_lr = e_mag_space_freq_lr / p_ref_squared                # Normalize
        e_mag_space_freq_lr = 10*torch.log10(e_mag_space_freq_lr)                # Calc dB error
        e_mag_freq_lr       = torch.mean(e_mag_space_freq_lr, dim = 1).squeeze() # E[] over space
    else:
        e_mag_freq_lr  = torch.linalg.norm(e_mag_space_freq_lr, dim = 1).squeeze()      # E[] over space
    
    e_mag_freq     = torch.mean(e_mag_freq_lr,dim = 0).squeeze()                 # E[] over ears
    
    return e_mag_freq

def clc_e_mag_diff(p_hat,p_ref):
    p_hat_l = torch.diff(torch.abs(p_hat[0,:,:].squeeze()),n=1, dim=1)
    p_hat_r = torch.diff(torch.abs(p_hat[1,:,:].squeeze()),n=1, dim=1)

    p_ref_l = torch.diff(torch.abs(p_ref[0,:,:].squeeze()),n=1, dim=1)
    p_ref_r = torch.diff(torch.abs(p_ref[1,:,:].squeeze()),n=1, dim=1)

    
    e_diff_l = torch.linalg.norm(torch.abs(p_hat_l - p_ref_l), dim=0).squeeze()
    e_diff_r = torch.linalg.norm(torch.abs(p_hat_r - p_ref_r), dim=0).squeeze()
    
    e_diff_out_freq = ((e_diff_l + e_diff_r) / 2)


    return e_diff_out_freq

    
def loss_func(input_dict,iteration):
    # Unload input dictionary
    p_lebedev_f = input_dict["p_lebedev_f"]
    c           = input_dict["c"]
    V_f_horiz   = input_dict["V_f_horiz"]
    V_f_lebedev = input_dict["V_f_lebedev"]
    device      = input_dict["device"]
    AK_f_c      = input_dict["AK_f_c"]
    AK_Nmax     = input_dict["AK_Nmax"]
    AK_n        = input_dict["AK_n"]
    AK_C        = input_dict["AK_C"]
    nfft        = input_dict["nfft"]
    fs          = input_dict["fs"]
    ILD_band    = input_dict["ILD_band"]
    cutOffFreq  = input_dict["cutOffFreq"]
    M_dB        = input_dict["M_dB"] 
    ILD_ref     = input_dict["ILD_ref"]
    lambda_vec  = input_dict["lambda_vec"]
    ILD_W       = input_dict["ILD_W_M"]
    
    # Space interpulation
    p_hat_horiz_t   = BSM_binaural_repo(c, V_f_horiz, device,is_time=True)
    p_hat_lebedev_f = BSM_binaural_repo(c, V_f_lebedev, device,is_time=False)
    # NMSE
    e_nmse       = clc_e_nmse(p_lebedev_f,p_hat_lebedev_f,norm_flag=False)
    monitor_nmse = clc_e_nmse(p_lebedev_f,p_hat_lebedev_f,norm_flag=True)
    # Magnitude
    e_mag       = clc_e_mag(p_lebedev_f,p_hat_lebedev_f,M_dB=torch.tensor([]),norm_flag=False)
    monitor_mag = clc_e_mag(p_lebedev_f,p_hat_lebedev_f,M_dB=torch.tensor([]),norm_flag=True)
    # Masked Magnitude
    #e_mmag       = clc_e_mag(p_lebedev_f,p_hat_lebedev_f,M_dB,norm_flag=False)
    #monitor_mmag = clc_e_mag(p_lebedev_f,p_hat_lebedev_f,M_dB,norm_flag=True)
    # ILD
    ILD_hat  = ak.clc_ILD(p_hat_horiz_t, AK_f_c, fs, ILD_band, AK_Nmax, AK_n, AK_C, nfft) # [freq x directions]
    #e_ILD    = torch.abs(torch.log10(torch.abs(ILD_ref - ILD_hat)/2)) # [freq x directions] 
    e_ILD    = torch.abs(ILD_ref - ILD_hat)*ILD_W # [freq x directions]
    

    
    #print(ILD_W[1,:].squeeze())
    #==============================================================================
    # Calculate the derivitive magnitude, to smooth results
    #==============================================================================
    e_mag_diff = clc_e_mag_diff(p_hat_lebedev_f,p_lebedev_f)

    # Error band indexs
    f_vec   = np.arange(0,e_nmse.shape[0])*((fs/2)/(e_nmse.shape[0]-1))
    #cutOffFreq_half_octave = cutOffFreq *np.sqrt(2)
    cutOffFreq_half_octave = cutOffFreq
    idx_max_mag = (np.abs(f_vec - 20e3)).argmin()
    idx_min_mag = (np.abs(f_vec - cutOffFreq_half_octave)).argmin()
    
    
    # Mix errors
    lambda_0 = lambda_vec[0]
    lambda_1 = lambda_vec[1]
    #lambda_2 = lambda_vec[2]
    lambda_2 = lambda_vec[2]
    lambda_3 = lambda_vec[3]
    h0 = torch.mean(e_nmse[:idx_min_mag])
    h1 = torch.mean(e_mag[idx_min_mag:idx_max_mag])
    #h2 = torch.mean(e_mmag[idx_min_mag:idx_max_mag])
    #h3 = torch.mean(e_ILD)
    h2 = torch.linalg.norm( torch.linalg.norm(e_ILD,dim=1),dim=0)
    #h3 = torch.linalg.norm( e_ILD_f,dim=0)
    h3 = torch.mean(e_mag_diff[idx_min_mag+1:idx_max_mag])
    # Define lambda parameters and h variables
    #lambda_values = [lambda_0, lambda_1, lambda_2, lambda_3, lambda_4]
    lambda_values = [lambda_0, lambda_1, lambda_2, lambda_3]
    h_values      = [h0, h1, h2, h3]
    #h_values      = [h0, h1, h2, h3, h4]
    # Initialize e_total
    e_total = 0

    
    if iteration ==0:
        for i in range(len(lambda_values)):
            if lambda_values[i] != 0:
                # save current normalized error values
                lambda_values[i] = (float((lambda_values[i] /  h_values[i]).detach().numpy()))
                # update the lambda vector with the initial lambda values for later iterations
                lambda_vec[i] = lambda_values[i]
    
    
    # Add terms to e_total based on non-zero lambda values
    for lambda_val, h_val in zip(lambda_values, h_values):
            e_total += lambda_val * h_val

    # Save Output dictionary
    output_dict = {
        "e_total": e_total,
        "monitor_nmse": monitor_nmse,
        "monitor_mag": monitor_mag,
        #"monitor_mmag": monitor_mmag,
        "h_nmse": h0,
        "h_mag": h1,
        #"h_mmag": h2,
        "h_eILD": h2,
        "h_diff": h3,
        "ILD": ILD_hat,
        "e_ILD": e_ILD,
        "e_nmse": e_nmse,
        "e_mag": e_mag,
        #"e_mmag": e_mmag,
        "e_mag_diff": e_mag_diff,
        "lambda_vec": lambda_vec,
    }
    return output_dict
    



def start(lambda_vec,epochs, lr_curr, data_path,shutup,is_save):
    os.makedirs(data_path+"/figures/", exist_ok=True)
    # Setup device
    has_gpu = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_built()
    device = "mps" if torch.backends.mps.is_built() \
        else "gpu" if torch.cuda.is_available() else "cpu"
    if not(shutup):
        print(f"Python Platform: {platform.platform()}")
        print(f"PyTorch Version: {torch.__version__}")
        print()
        print(f"Python {sys.version}")
        print("NVIDIA/CUDA GPU is", "available" if has_gpu else "NOT AVAILABLE")
        print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
    device = "cpu"
    if not(shutup):
        print(f"Target device is {device}")
            
    import_dict = import_matlab_data(data_path+"pytorch_data.mat",device,shutup)
    # Estimate refference ILD over horizontal plane
    p_ref_horiz_t            = torch.fft.irfft( import_dict["p_ref_horiz_f"], n=import_dict["nfft"], dim=2)  # space x time x left/right
    p_ref_horiz_t            = torch.roll(p_ref_horiz_t, shifts=import_dict["nfft"] // 2, dims=2)
    p_ref_horiz_t            = torch.permute(p_ref_horiz_t,[1,2,0])
    ILD_band                 = [1.5e3, 20e3]
    AK_Nmax,AK_f_c,AK_n,AK_C = ak.AKerbILD_short_p1(p_ref_horiz_t, ILD_band, import_dict["fs"])
    AK_C                     = torch.from_numpy(AK_C).to(device)
    ILD_ref                  = ak.clc_ILD(p_ref_horiz_t, AK_f_c, import_dict["fs"], ILD_band, AK_Nmax, AK_n, AK_C, import_dict["nfft"])
    
    
    
    
    
    # Calc the notch estimate mask
    notch_param_dict = {
            "cutoff_threshold": 6,
            "openinng_ang": 60,
            "threshold": 4,
            "max_notch_num": 4,
            "JND_CC": 0.15
        }
    sofa_path = import_dict["HRTFpath"]
    space_flag = "full_contra" # can be: "full" / "median" / "horizontal"/ "full_contra"(removes contralateral nothces) / "frontal" (only front facing angels are concidered)
    fig_name = "M-notch_mask_left_ear.png"
    notch_mask_0 = notch_mask.calc_notch_mask(sofa_path,space_flag,cutoff_threshold = notch_param_dict["cutoff_threshold"], k = notch_param_dict["max_notch_num"],threshold = notch_param_dict["threshold"],is_plot=False,plot_ang=[0,35],is_sort_full = False,openinng_ang= notch_param_dict["openinng_ang"],figures_savepath= ".",fig_name = fig_name,is_save = False,shutup= False)
    
    notch_mask_0 = notch_mask_0.permute([2,0,1])


    # Wighting matrix for ILD directions
    mask_val = 1
    az_vec = np.rad2deg(import_dict["omega_horizontal"][:,1])
    angle_range = (-60, 60)  # Define the desired angle range
    
    # Create the weighting matrix
    weights = torch.ones_like(torch.from_numpy(az_vec))  # Initialize with ones
    mask_1 = (az_vec >= angle_range[0]) & (az_vec <= angle_range[1])
    mask_2 = az_vec <= (-180  - angle_range[0])
    mask_3 = az_vec >= (180 - angle_range[1])

    weights[mask_1] = mask_val  # Set weights for indices within the desired angle range


    
    
    # Repeat the weights to match the shape of the error tensor
    ILD_weights_matrix = weights.unsqueeze(0).repeat(ILD_ref.shape[0], 1)
    
    # Ensure the weighting matrix is non-trainable
    ILD_weights_matrix = ILD_weights_matrix.requires_grad_(True)
    
    
    # Set init bsm solution
    c_hat = import_dict["c_ls"]
    c_hat.to(device)
    
    input_dict = {
            "p_lebedev_f": import_dict["p_ref_lebedev_f"],
            "c": c_hat,
            "V_f_horiz": import_dict["V_k_horiz"],
            "V_f_lebedev": import_dict["V_k_lebedev"],
            "device": device,
            "AK_f_c": AK_f_c,
            "AK_Nmax": AK_Nmax,
            "AK_n": AK_n,
            "AK_C": AK_C,
            "nfft": import_dict["nfft"],
            "fs": import_dict["fs"],
            "ILD_band": ILD_band,
            "cutOffFreq": import_dict["f_cut_magLS"],
            "M_dB": notch_mask_0,
            "ILD_W_M": ILD_weights_matrix,
            "ILD_ref": ILD_ref,
            "lambda_vec": lambda_vec,
        }
    
    output_dict_ls = loss_func(input_dict,1)
    
    # Set init bsm solution
    c_hat = import_dict["c_mls"]
    c_hat.to(device)
    input_dict["c"] = c_hat
    
    output_dict_mls = loss_func(input_dict,1)
    
    #cutOffFreq_half_octave = import_dict["f_cut_magLS"] *np.sqrt(2)
    cutOffFreq_half_octave = import_dict["f_cut_magLS"]
    idx_max_mag = (np.abs(import_dict["f_vec"] - 20e3)).argmin()
    idx_min_mag = (np.abs(import_dict["f_vec"] - cutOffFreq_half_octave)).argmin()

    
    # Set NauralNet stuff
    model = NN(c_hat.shape[0],c_hat.shape[2],idx_min_mag)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_curr)
    optimizer.zero_grad()
    
    if not(shutup):  
        print("\n\n-----------NN summary---------")
        result = summary(model,input_size= c_hat.shape,dtypes=[torch.complex128],verbose=2,col_width=13,
            col_names=["kernel_size","input_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"],)
    
    
    h_nmse  = []
    h_mag   = []
    h_mmag  = []
    h_ILD   = []
    h_diff  = []
    h_total = []
    
    
    c_hat = import_dict["c_mls"]
    c_hat.to(device)
    input_dict["c"] = c_hat
    


    
    # Optimize for lambda_vec
    for epoch in tqdm (range (epochs), desc="Loading..."):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            input_dict["c"]  = model(c_hat)
            output_dict      = loss_func(input_dict,epoch)
            output_dict["e_total"].backward(retain_graph=True)
    
            if epoch == 0:
                input_dict["lambda_vec"]  = output_dict["lambda_vec"]
                #if not(shutup):
                    #print("new lambda vec: ",input_dict["lambda_vec"])
            
            
            # Check gradients for NaN
            nan_detected = False
    
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f'NaN gradients detected in parameter "{name}"')
                    nan_detected = True
                    optimizer.zero_grad()
                    break
            if not nan_detected:
                optimizer.step()
            else:
                break

            
            # Collect train history
            # lambda[0] = NMSE
            # lambda[1] = Magnitude
            # lambda[2] = Masked Magnitude
            # lambda[3] = ILD
            h_total = np.append(h_total, output_dict["e_total"].detach().cpu().numpy()) 
            h_ILD   = np.append(h_ILD,  (output_dict["h_eILD"].detach().cpu().numpy() * output_dict["lambda_vec"][2])) 
            h_nmse  = np.append(h_nmse, (output_dict["h_nmse"].detach().cpu().numpy() * output_dict["lambda_vec"][0]))
            h_mag   = np.append(h_mag,  (output_dict["h_mag"].detach().cpu().numpy() * output_dict["lambda_vec"][1]))
            #h_mmag  = np.append(h_mmag, (output_dict["h_mmag"].detach().cpu().numpy() * output_dict["lambda_vec"][2]))
            h_diff  = np.append(h_diff, (output_dict["h_diff"].detach().cpu().numpy() * output_dict["lambda_vec"][3]))
            # Early stop logic
            if epoch >= 1:
                improvement = 100*(np.abs(h_total[epoch] - h_total[epoch-1]) / abs(h_total[epoch-1]))
                if improvement < 0.001:
                    print('Stopping: Loss improvement is below 0.001% threshold.')
                    optimizer.zero_grad()
                    break
    
            
    
    
    
    
    results_data = {
        "ILD_ref": input_dict["ILD_ref"].numpy(),
        "ILD_ls": output_dict_ls["ILD"].numpy(),
        "ILD_mls": output_dict_mls["ILD"].numpy(),
        "ILD_imls": output_dict["ILD"].detach().numpy(),
        "ILD_az_vec": az_vec.squeeze(),
    
        "e_nmse_ls": output_dict_ls["monitor_nmse"].numpy().squeeze(),
        "e_nmse_mls": output_dict_mls["monitor_nmse"].numpy().squeeze(),
        "e_nmse_imls": output_dict["monitor_nmse"].detach().numpy().squeeze(),
        "e_mag_ls": output_dict_ls["monitor_mag"].numpy().squeeze(),
        "e_mag_mls": output_dict_mls["monitor_mag"].numpy().squeeze(),
        "e_mag_imls": output_dict["monitor_mag"].detach().numpy().squeeze(),
        "f_vec": import_dict["f_vec"].squeeze(),
        "AK_f_c": AK_f_c.squeeze(),
    
        "h_nmse": h_nmse,
        "h_mag": h_mag,
        #"h_mmag": h_mmag,
        "h_ILD": h_ILD,
        "h_diff": h_diff,
        "fs": import_dict["fs"],
        "ILD_band": ILD_band,
        "cutOffFreq": import_dict["f_cut_magLS"],
        "gamma": import_dict["gamma"],
        "data_path": data_path,
        }
    
    # Save as .mat file
    scipy.io.savemat(data_path+"/results_data.mat", results_data)

    # Save as Sofa files
    C_ls = import_dict["c_ls"]
    C_mls = import_dict["c_mls"]
    C_imls = input_dict["c"].detach()




def plot_results_graphs_via_m_file(data_path,shutup,is_save):
    # reading the .m results file
    mat = scipy.io.loadmat(data_path+"/results_data.mat")
    
    ILD_ref = mat["ILD_ref"]
    ILD_ls = mat["ILD_ls"]
    ILD_mls = mat["ILD_mls"]
    ILD_imls = mat["ILD_imls"]
    az_vec = mat["ILD_az_vec"].squeeze()
    f_ak = mat["AK_f_c"].squeeze()
    
    e_nmse_ls = mat["e_nmse_ls"].squeeze()
    e_nmse_mls = mat["e_nmse_mls"].squeeze()
    e_nmse_imls = mat["e_nmse_imls"].squeeze()
    e_mag_ls = mat["e_mag_ls"].squeeze()
    e_mag_mls = mat["e_mag_mls"].squeeze()
    e_mag_imls = mat["e_mag_imls"].squeeze()
    f_vec = mat["f_vec"].squeeze()
    
    h_nmse = mat["h_nmse"].squeeze()
    h_mag = mat["h_mag"].squeeze()
    #h_mmag = mat["h_mmag"].squeeze()
    h_ILD = mat["h_ILD"].squeeze()
    h_diff = mat["h_diff"].squeeze()
    
    data_path_res = str(mat["data_path"][0])
    
    
    
    # Plotting
    
    #Traning curves
    plt.figure(figsize=(6, 3))
    plt.plot(h_nmse, label='e NMSE')
    plt.plot(h_mag, label='e Magnitude')
    plt.plot(h_ILD, label='e ILD')
    plt.plot(h_diff, label='e diff')
    plt.grid()
    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.title("Training Curves")
    plt.legend(loc="upper right");
    if is_save:
        file_name = "/figures/Training_Curves.png"
        plt.savefig(data_path_res+file_name)
    if not(shutup):
        plt.show()
    else:
        plt.close()
    
    
    plt.figure(figsize=(6, 3))
    plt.semilogx(f_vec, e_nmse_ls, label='NMSE - LS')
    plt.semilogx(f_vec, e_nmse_imls, label='NMSE - iMagLS')
    plt.semilogx(f_vec, e_mag_mls, label='Mag - MagLS')
    plt.semilogx(f_vec, e_mag_imls, label='Mag - iMagLS')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim([20 , 20e3])
    plt.ylim([-45 , 10])
    plt.title("Frequency errors")
    plt.grid(True)
    plt.legend()
    if is_save:
        file_name = "/figures/Frequency_errors.png"
        plt.savefig(data_path_res+file_name)
    if not(shutup):
        plt.show()
    else:
        plt.close()
    
    plt.figure(figsize=(6, 3))
    plt.plot(az_vec,np.mean(ILD_ref,axis=0).squeeze(), label='Ref')
    plt.plot(az_vec,np.mean(ILD_ls,axis=0).squeeze(), label='LS')
    plt.plot(az_vec,np.mean(ILD_mls,axis=0).squeeze(), label='MagLS')
    plt.plot(az_vec,np.mean(ILD_imls,axis=0).squeeze(), label='iMagLS')
    plt.xlabel('Angle [deg]')
    plt.ylabel('ILD')
    plt.title("ILD curves")
    plt.xlim([az_vec[0] , az_vec[-1]])
    plt.grid(True)
    plt.legend()
    if is_save:
        file_name = "/figures/ILD_curves.png"
        plt.savefig(data_path_res+file_name)
    if not(shutup):
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(az_vec,np.mean(np.abs(ILD_ref-ILD_ls),axis=0).squeeze(), label='error: LS')
    plt.plot(az_vec,np.mean(np.abs(ILD_ref-ILD_mls),axis=0).squeeze(), label='error: MagLS')
    plt.plot(az_vec,np.mean(np.abs(ILD_ref-ILD_imls),axis=0).squeeze(), label='error: iMagLS')
    plt.xlabel('Angle [deg]')
    plt.ylabel('Error [dB]')
    plt.title("ILD errors")
    plt.xlim([az_vec[0] , az_vec[-1]])
    plt.grid(True)
    plt.legend()
    if is_save:
        file_name = "/figures/ILD_errors_ang.png"
        plt.savefig(data_path_res+file_name)
    if not(shutup):
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(f_ak,np.mean(np.abs(ILD_ref-ILD_ls),axis=1).squeeze(), label='error: LS')
    plt.plot(f_ak,np.mean(np.abs(ILD_ref-ILD_mls),axis=1).squeeze(), label='error: MagLS')
    plt.plot(f_ak,np.mean(np.abs(ILD_ref-ILD_imls),axis=1).squeeze(), label='error: iMagLS')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Error [dB]')
    plt.title("ILD errors")
    plt.xlim([f_ak[0] , f_ak[-1]])
    plt.grid(True)
    plt.legend()
    if is_save:
        file_name = "/figures/ILD_errors_freq.png"
        plt.savefig(data_path_res+file_name)
    if not(shutup):
        plt.show()
    else:
        plt.close()
    

