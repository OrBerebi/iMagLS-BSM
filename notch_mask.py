import pyfar as pf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch


def calc_notch_mask(sofa_path,space_flag="full",cutoff_threshold = [], k = 4,threshold = 2.0,is_plot=False,plot_ang=[0,35],is_sort_full = False,openinng_ang=45,figures_savepath=[],fig_name=[],is_save=False,shutup=True):
    hrir, coardinate, *_ = pf.io.read_sofa(sofa_path)
    match space_flag:
        case "full":
            # get the full plane
            if is_sort_full:
                sort = np.argsort(coardinate.polar)
                coardinate = coardinate[sort]
                hrir = hrir[sort]
                ang_vec = np.rad2deg(coardinate.polar)
            else:
                ang_vec = np.rad2deg(coardinate.azimuth)
        case "median":
            # get the meadian plane
            mask = np.squeeze(np.logical_or(coardinate.azimuth == 0, coardinate.azimuth == np.pi))
            coardinate_new_plane = coardinate[mask]
            sort = np.argsort(coardinate_new_plane.polar)
            coardinate_new_plane = coardinate_new_plane[sort]
            ang_vec = np.rad2deg(coardinate_new_plane.polar)
            hrir = hrir[mask,:]
            hrir = hrir[sort] 
        case "horizontal":
            tol = np.deg2rad(0.5)
            mask = np.squeeze(np.logical_and(coardinate.elevation <= (0+tol),coardinate.elevation >= (0-tol)))
            coardinate_new_plane = coardinate[mask]
            sort = np.argsort(coardinate_new_plane.azimuth)
            coardinate_new_plane = coardinate_new_plane[sort]
            ang_vec = np.rad2deg(coardinate_new_plane.azimuth)
            hrir = hrir[mask,:]
            hrir = hrir[sort]
        case "full_contra":
            # get the full plane
            if is_sort_full:
                sort = np.argsort(coardinate.polar)
                coardinate = coardinate[sort]
                hrir = hrir[sort]
                ang_vec = np.rad2deg(coardinate.polar)
            else:
                ang_vec = np.rad2deg(coardinate.azimuth)

            idx_left = np.squeeze(coardinate.lateral >= np.deg2rad(90 - openinng_ang))
            idx_right = np.squeeze(coardinate.lateral <= np.deg2rad(openinng_ang - 90))
        case "frontal":
            
            # get the full plane
            if is_sort_full:
                sort = np.argsort(coardinate.polar)
                coardinate = coardinate[sort]
                hrir = hrir[sort]
                ang_vec = np.rad2deg(coardinate.polar)
            else:
                ang_vec = np.rad2deg(coardinate.azimuth)

            coardinate_tmp = coardinate
            coardinate_tmp.radius  = 1
            # get the frontal direction
            p = pf.Coordinates.from_cartesian(1, 0, 0) 
            
            idx_frontal = coardinate_tmp.find_within(p, distance=np.deg2rad(openinng_ang), distance_measure='spherical_radians', atol=None, return_sorted=True, radius_tol=None)
            #coardinate.show(idx_frontal)
            #plt.show()
            #coardinate = coardinate[index_out]
            
            
        case _:
            raise ValueError(f"Invalid space_flag value: {space_flag}")
            
            
    hrir_smooth_low  = pf.dsp.smooth_fractional_octave(hrir, 1)[0]
    hrir_smooth_high = pf.dsp.smooth_fractional_octave(hrir, 11)[0]

    f_cutoff = 18e3
    f_vec = hrir.frequencies
    f_cutoff_idx = np.argmin(np.abs(f_cutoff - f_vec))
    oct = 1/4
    f_cutoff_low = 5.5e3 * (2 ** -oct)
    f_cutoff_idx_low = np.argmin(np.abs(f_cutoff_low - f_vec))

    fade_out = fadeout_lin_filter(f_cutoff, f_vec,oct=1/8)
    fade_in = (1 - fadeout_lin_filter(f_cutoff_low, f_vec,oct))

    band_pass = fade_out*fade_in
    band_pass = band_pass.reshape(band_pass.shape[0], 1)

    
    hrir_diff_np = 20*np.log10(np.abs(hrir_smooth_low.freq)) - 20*np.log10(np.abs(hrir_smooth_high.freq))
    # remove all negative values
    hrir_diff_np[hrir_diff_np < 0] = 0

    """
    # Pad the last dimension to compensate for the high freqncies
    pad_width = ((0, 0), (0, 0), (0, 1+ f_vec.shape[0] - f_cutoff_idx))  # No padding for the first two dimensions, pad  zeros at the end of the last dimension
    hrir_diff_np = np.pad(hrir_diff_np, pad_width, mode='constant', constant_values=0)

     # Pad the last dimension to compensate for the low freqncies
    pad_width = ((0, 0), (0, 0), (f_cutoff_idx_low, 0))  # No padding for the first two dimensions, pad  zeros at the end of the last dimension
    hrir_diff_np = np.pad(hrir_diff_np, pad_width, mode='constant', constant_values=0)
    """

    
    # find k peaks and thire band width
    peak_indices = find_k_peaks(hrir_diff_np, k,threshold)   # [space x ears x peaks]
    bw_indices   = find_peak_bw(hrir_diff_np,peak_indices)   # [space x ears x peaks x low/high index]
    
    
    mask = zero_non_bw(hrir_diff_np,bw_indices)
    hrir_diff_np[~mask] = 0
    
    band_pass_space = np.tile(band_pass, (1, hrir_diff_np.shape[0])).T
    hrir_diff_np[:,0,:] = hrir_diff_np[:,0,:].squeeze() * band_pass_space
    hrir_diff_np[:,1,:] = hrir_diff_np[:,1,:].squeeze() * band_pass_space

    # limit the value of the nothces to cutoff_threshold
    if cutoff_threshold:
        hrir_diff_np[hrir_diff_np > cutoff_threshold] = cutoff_threshold

    if space_flag == "full_contra":
        hrir_diff_np[idx_left,1,:] = 0
        hrir_diff_np[idx_right,0,:] = 0


    if space_flag == "frontal":
        # Create a mask with all indices set to True
        mask = np.ones(hrir_diff_np.shape[0], dtype=bool)
        # Set the indices in L to False
        mask[idx_frontal] = False
        hrir_diff_np[mask] = -60
        

    
    
    
    if is_plot:
        # get plot_ang index in the pf.Coordinates object
        azimuth    = np.deg2rad(plot_ang[0])
        colatitude = np.deg2rad(plot_ang[1])
        radius     = 3.25
        [x,y,z]    = pf.classes.coordinates.sph2cart(azimuth,colatitude,radius)
        to_find    = pf.Coordinates(x, y, z)
        index, distance = coardinate.find_nearest(to_find)
        
        plt.figure()
        plt.plot(hrir_diff_np[index,:,:].squeeze().T)
        # left ear peaks
        plt.plot(peak_indices[index,0,:].squeeze(),hrir_diff_np[index,0,peak_indices[index,0,:].squeeze()].squeeze(), 'o')
        # right ear peaks
        plt.plot(peak_indices[index,1,:].squeeze(),hrir_diff_np[index,1,peak_indices[index,1,:].squeeze()].squeeze(), 'o')
        
        # leaft ear bw
        plt.plot(bw_indices[index,0,:,0].squeeze(),hrir_diff_np[index,0,bw_indices[index,0,:,0].squeeze()].squeeze(), 'o')
        plt.plot(bw_indices[index,0,:,1].squeeze(),hrir_diff_np[index,0,bw_indices[index,0,:,1].squeeze()].squeeze(), 'o')
        # right ear bw
        plt.plot(bw_indices[index,1,:,0].squeeze(),hrir_diff_np[index,1,bw_indices[index,1,:,0].squeeze()].squeeze(), 'o')
        plt.plot(bw_indices[index,1,:,1].squeeze(),hrir_diff_np[index,1,bw_indices[index,1,:,1].squeeze()].squeeze(), 'o')
        plt.title("Smooth non-smooth diffrence")
        plt.show()


        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot for the left ear
        plt.sca(ax1)  # Set the current Axes to ax1
        pf.plot.freq(hrir[index, 0])
        pf.plot.freq(hrir_smooth_low[index, 0])
        pf.plot.freq(hrir_smooth_high[index, 0])
        ax1.set_title("example HRIR (left ear)")
        
        # Plot for the right ear
        plt.sca(ax2)  # Set the current Axes to ax2
        pf.plot.freq(hrir[index, 1])
        pf.plot.freq(hrir_smooth_low[index, 1])
        pf.plot.freq(hrir_smooth_high[index, 1])
        ax2.set_title("example HRIR (right ear)")
        plt.show()



        
        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6)) 
        # Plot for the left ear
        peak_mask_left = hrir_diff_np[:, 0, :].squeeze().T
        im1 = ax1.imshow(peak_mask_left, aspect='auto', origin='lower', extent=[ang_vec[0], ang_vec[-1], f_vec[0], f_vec[-1]])
        #ax1.set_title('Notch detection mask (Left ear)')
        ax1.set_xlabel('Angle [Deg]')
        ax1.set_ylabel('Frequency [Hz]')
        #im1.set_clim(-20, cutoff_threshold)
        im1.set_clim(0, 10)
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label('[dB]')
        
        # Plot for the right ear
        peak_mask_right = hrir_diff_np[:, 1, :].squeeze().T
        im2 = ax2.imshow(peak_mask_right, aspect='auto', origin='lower', extent=[ang_vec[0], ang_vec[-1], f_vec[0], f_vec[-1]])
        #ax2.set_title('Notch detection mask (Right ear)')
        ax2.set_xlabel('Angle [Deg]')
        ax2.set_ylabel('Frequency [Hz]')
        #im2.set_clim(-20, cutoff_threshold)
        im2.set_clim(0, 10)
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label('[dB]')
        plt.show()

        '''
        #f_vec = f_vec/1e3
        fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6)) 
        # Plot for the left ear
        peak_mask_left = hrir_diff_np[:, 0, :].squeeze().T
        #im1 = ax1.imshow(peak_mask_left, aspect='auto', origin='lower', extent=[ang_vec[0], ang_vec[-1], f_vec[0], f_vec[-1]])
        im1 = ax1.imshow(peak_mask_left, aspect='auto', origin='lower')
        #ax1.set_title('Notch detection mask (Left ear)')        
        # Set logarithmic scale for y-axis
        ax1.set_yscale('log')
        ax1.set_ylim([f_vec[0], f_vec[-1]])  # Ensure correct y-axis limits

        ax1.set_xlabel('Polar Angle [Deg]')
        ax1.set_ylabel('Frequency [kHz]')
        #im1.set_clim(-20, cutoff_threshold)
        im1.set_clim(0, 10)
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label('Magnitude [dB]')
        '''


        # Sample data for demonstration (replace with your actual data)
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))  # Adjust figure size as needed (here, half-column width)
        
        # Use pcolormesh for the plot, swapping ang_vec and f_vec
        mesh = ax1.pcolormesh(f_vec, ang_vec, peak_mask_left.T, shading='auto', cmap='viridis')
        
        # Set logarithmic scale for x-axis
        ax1.set_xscale('log')
        ax1.set_xlim([200, 20e3])  # Ensure correct x-axis limits
        mesh.set_clim(0, 10)  # Example values for color limits (adjust as per your data)
        xticks = [200, 400, 600, 1000, 2000, 4000, 6000, 10000, 20000]  # Frequency values in Hz
        ax1.set_xticks(xticks)
        #ax1.set_xticklabels([f'{int(tick/1e3)}k' for tick in xticks])  # Format labels as '0.2k', '0.4k', etc.
        ax1.set_xticklabels([f'{tick}' if tick < 1000 else f'{int(tick/1e3)}k' for tick in xticks])  # Format labels correctly
        yticks = [-50, 0, 50, 100, 150,200, 250]  # Frequency values in kHz
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{int(tick)}' for tick in yticks])  # Format labels as '1k', '2k', etc.


        # Set labels and colorbar
        ax1.set_xlabel('Frequency [kHz]')
        ax1.set_ylabel('Polar Angle [Deg]')
        cbar1 = fig.colorbar(mesh, ax=ax1)
        cbar1.set_label('Magnitude [dB]')
        plt.tight_layout()
        
        if is_save:
            plt.savefig(figures_savepath + "/" + fig_name)
        if not shutup:
            plt.show()
        else:
            plt.close()
        

    mask_out = torch.from_numpy(hrir_diff_np).permute([0,2,1])
    return mask_out


def fadeout_lin_filter(f_c, f_vec,oct):
    num_bins = f_vec.shape[0]
    # Calculate the frequency range for one octave
    f_lower = f_c
    f_upper = f_c * (2 ** oct)
    
    # Convert these frequencies back to bin indices
    bin_lower = np.argmin(np.abs(f_vec - f_lower))
    bin_upper = np.argmin(np.abs(f_vec - f_upper))
    
    # Ensure indices are within valid range
    bin_lower = max(bin_lower, 0)
    bin_upper = min(bin_upper, num_bins - 1)
    
    # Create the cross-fade filter
    fade_out = np.zeros(num_bins)
    fade_range = bin_upper - bin_lower + 1
    fade_out[bin_lower:bin_upper + 1] = np.linspace(1, 0, fade_range)
    fade_out[:bin_lower] = 1
    return fade_out

def find_k_peaks(matrix, k,threshold=1.2):
    num_samples, num_channels, num_values = matrix.shape
    all_peak_indices = np.full((num_samples, num_channels, k), -1, dtype=int)  # to store indices of peaks

    for sample in range(num_samples):
        for channel in range(num_channels):
            values = matrix[sample, channel, :]
            
            # Use find_peaks to get all peak indices
            peak_indices, _ = scipy.signal.find_peaks(values,height = threshold,distance = 20 )
            
            # Sort peak indices by corresponding value (in descending order) and select the top k indices
            sorted_peak_indices = sorted(peak_indices, key=lambda idx: values[idx], reverse=True)[:k]
            
            # If there are less than k peaks, pad with 0
            if len(sorted_peak_indices) < k:
                sorted_peak_indices += [0] * (k - len(sorted_peak_indices))
                
            all_peak_indices[sample, channel, :len(sorted_peak_indices)] = sorted_peak_indices

    return all_peak_indices


def find_peak_bw(matrix,peak_indices):
    mask = (matrix == 0)
    bw_out = np.zeros([mask.shape[0],mask.shape[1],peak_indices.shape[2],2]) # [space x ears x peaks x low/high index]
    for space_idx in range(mask.shape[0]):
        for ear_idx in range(mask.shape[1]):
            #print("\n\n\tear idx: ", ear_idx)
            m_idx  = mask[space_idx,ear_idx,:]
            p_idx  = peak_indices[space_idx,ear_idx,:]
            for peaks in range(p_idx.shape[0]):
                # find the peak low band
                #print("\npeak index: ",peaks)
                idx_low = p_idx[peaks]
                while not(m_idx[idx_low]):
                    idx_low = idx_low - 1

                #print("notch low index: ", idx_low)
                bw_out[space_idx,ear_idx,peaks,0] = idx_low
            
                # find the peak high band
                idx_high = p_idx[peaks]
                while not(m_idx[idx_high]):
                    idx_high = idx_high + 1
                    if idx_high == (mask.shape[2] - 1):
                        break
                #print("notch high index: ", idx_high)
                bw_out[space_idx,ear_idx,peaks,1] = idx_high
    return bw_out.astype(int)

def zero_non_bw(values,bw_indices):
    indices = np.arange(values.shape[2])
    mask = np.zeros(values.shape, dtype=bool) # all false values
    for peak_idx in range(bw_indices.shape[-2]):
        # true inices highier then low_bw
        foo_low = bw_indices[:,:,peak_idx,0].squeeze()
        foo_low = foo_low[:, :, np.newaxis] # [space x ears x 1]
        mask_low = indices > foo_low
        # true inices low then high_bw
        foo_high = bw_indices[:,:,peak_idx,1].squeeze()
        foo_high = foo_high[:, :, np.newaxis] # [space x ears x 1]
        mask_high = indices < foo_high
        # and them to find the band bins
        mask_p = mask_low*mask_high
        # append results to the main mask
        mask   = np.logical_or(mask_p,mask)


    return mask