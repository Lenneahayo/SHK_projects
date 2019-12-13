import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from random import random
import matplotlib
from netCDF4 import Dataset
import sys
from slopes_and_binning import *

import pandas as pd

def add_buffer(A,n_extra):
    """Adds n_extra cells/columns in x and y direction to array A. Works with 2d and 3d arrays, super advanced stuff right here. """
    if A.ndim == 2:
        A_extra = np.vstack([A[-n_extra:,:],A,A[:n_extra,:]])
        A_extra = np.hstack([A_extra[:,-n_extra:],A_extra,A_extra[:,:n_extra]])
    if A.ndim == 3:
        A_extra = np.concatenate((A[:,-n_extra:,:],A,A[:,:n_extra,:]),axis=1)
        A_extra = np.concatenate((A_extra[:,:,-n_extra:],A_extra,A_extra[:,:,:n_extra]),axis=2)
        
    
    return A_extra
 
def cluster_2D(A,buffer_size=30 ):
    """
    For Lennéa, isn't pretty. 
    
    Uses inefficient np.where, but should be fine for 2D
    
    A is 2D matrix of 1 (cloud) and 0 (no cloud)
    buffer_size is the percentile added to each side to deal with periodic boundary domains
    
    returns labeled_clouds, is 0 where no cloud
    """
    #Uses a default periodic boundary domain
    n_max = A.shape[0]
 
    n_buffer = int(buffer_size/100.*n_max)

    #Explanding c and w fields with a buffer on each edge to take periodic boundaries into account. 
    A_buf=add_buffer(A,n_buffer)
    
    #labeled_clouds  = np.zeros_like(A_buf)
    
    #This is already very impressive, ndi.label detects all areas with marker =1 that are connected and gives each resulting cluster an individual integer value 
    labeled_clouds,n_clouds  = ndi.label(A_buf)
    
    
    
    #Going back from the padded field back to the original size
    # OK, calculate index means, then only look at those with a mean inside the original box
    # We ignore the cells with the mean outside, they will be cut off or overwritten
    # For those inside we check if they have something outside original box, and if so a very ugly hard coded overwritting is done. 
    # In the very end the segmentation box is cut back down to the original size

    
    
    
    #fancy quick sorting. 
    unique_labels, unique_label_counts = np.unique(labeled_clouds,return_counts=True)
    lin_idx       = np.argsort(labeled_clouds.ravel(), kind='mergesort')
    lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(labeled_clouds.ravel())[:-1]))
    
    for c in range(1,n_clouds+1): 
        idx_x,idx_y = np.unravel_index(lin_idx_split[c],labeled_clouds.shape)
    
    

        #idx_x,idx_y = np.where(labeled_clouds==c)
        idx_x_m = np.mean(idx_x)
        idx_y_m = np.mean(idx_y)

        if idx_x_m< n_buffer or idx_x_m>n_buffer+n_max or idx_y_m< n_buffer or idx_y_m>n_buffer+n_max:
            #cluster is outside, chuck it
            #print(c,'cluster out of bounds',idx_x,idx_y)
            #segmentation_cp[segmentation==c] = 0
            bla = 1

        else:
            idx_x_max = np.max(idx_x)
            idx_x_min = np.min(idx_x)
            idx_y_min = np.min(idx_y)
            idx_y_max = np.max(idx_y)
            if idx_x_min< n_buffer or idx_x_max>n_buffer+n_max or idx_y_min< n_buffer or idx_y_max>n_buffer+n_max:
                #print(c,'this is our guniea pig')
                if idx_x_min<n_buffer:
                    idx_x_sel = idx_x[idx_x<n_buffer]+n_max
                    idx_y_sel = idx_y[idx_x<n_buffer]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_x_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_x>=n_buffer+n_max]-n_max
                    idx_y_sel = idx_y[idx_x>=n_buffer+n_max]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_min<n_buffer:
                    idx_x_sel = idx_x[idx_y<n_buffer]
                    idx_y_sel = idx_y[idx_y<n_buffer]+n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_y>=n_buffer+n_max]
                    idx_y_sel = idx_y[idx_y>=n_buffer+n_max]-n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c



    #Now cut to the original domain
    labeled_clouds_orig = labeled_clouds[n_buffer:-n_buffer,n_buffer:-n_buffer]
    
    #And to clean up the missing labels 
    def sort_and_tidy_labels_2D(segmentation):
        """
        For a given 2D integer array sort_and_tidy_labels will renumber the array 
        so no gaps are between the the integer values and replace them beginning with 0 upward. 
        Also, the integer values will be sorted according to their frequency. 
        
        1D example: 
        [4,4,1,4,1,4,4,3,3,3,3,4,4]
        -> 
        [0,0,2,0,2,0,0,1,1,1,1,0,0]
        """
       
        unique_labels, unique_label_counts = np.unique(segmentation,return_counts=True)
        n_labels = len(unique_labels)
        unique_labels_sorted = [x for _,x in sorted(zip(unique_label_counts,unique_labels))][::-1]
        new_labels = np.arange(n_labels)
       
        lin_idx       = np.argsort(segmentation.ravel(), kind='mergesort')
        lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(segmentation.ravel())[:-1]))
        #think that I can now remove lin_idx, as it is an array with the size of the full domain. 
        del(lin_idx)
       
        for l in range(n_labels):
            c = unique_labels[l]
            idx_x,idx_y = np.unravel_index(lin_idx_split[c],segmentation.shape)
            segmentation[idx_x,idx_y] = new_labels[l]
       
        return segmentation 
    

    labeled_clouds_clean = sort_and_tidy_labels_2D(labeled_clouds_orig)
    
    
    return labeled_clouds_clean


def plot_cloud_slope(dist,time,timestep,bin_min,bin_max,bin_n):
    """
    Written by Lennéa Hayo, 19-11-28
    
    Creates a plot with slopes of cloud size distribution. Can be used either for one specific timestep or for a series of timesteps.
    
    Parameters:
        dist: distribution of logarithmic data (here cloud sizes)
        time: at this time the slope is being calculated
        timestep: if it is a single timestep: False, if it is a time series: True
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins
        
    """
    if (timestep):
        l2D = data_ql.variables[dist][time,:,:]

        l2D_bi = np.zeros_like(l2D).astype(int)

        l2D_bi[l2D>1e-6]=1
        labeled_clouds = cluster_2D(l2D_bi,buffer_size=20)

        #Grosse jeder wolken
        label, cl_pixels = np.unique(labeled_clouds.ravel(),return_counts=True)

        cl_size = np.sqrt(cl_pixels*25.*25.)
        bins_log_mm, ind, CSD = log_binner_minmax(cl_size[1:],1,300,100)
        x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if math.isnan(CSD[i]):
                  j = 0
            elif j > global_j:
                global_j = j
                global_i = i     
        if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
            print('there are no clouds for this timestep')
        else:
            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)

        plt.plot(x_bins_log_mm,CSD,'-o')
        plt.plot(x_bins_log_mm,f2)
        plt.xscale('log')
        plt.yscale('log')
        plt.show
    else:
        time_unraveled = time.ravel()
        slope = []
        time_nozeros = []
        time_hack = np.arange(time.size)/2.+6.
        for k in range(time.size):
            l2D = data_ql.variables[dist][k,:,:]

            l2D_bi = np.zeros_like(l2D).astype(int)

            l2D_bi[l2D>1e-6]=1
            labeled_clouds = cluster_2D(l2D_bi,buffer_size=20)

            #Grosse jeder wolken
            label, cl_pixels = np.unique(labeled_clouds.ravel(),return_counts=True)

            if (len(cl_pixels)>1):
                cl_size = np.sqrt(cl_pixels*25.*25.)
                bins_log_mm, ind, CSD = log_binner_minmax(cl_size[1:],1,300,100)
                x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

                j = 0
                global_j = 0
                global_i = 0
                for i in range(CSD.size):
                    j += 1
                    if math.isnan(CSD[i]):
                          j = 0
                    elif j > global_j:
                        global_j = j
                        global_i = i     
                if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
                    continue
                else:
                    m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
                    f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
                    slope.append(m2)
                    time_nozeros.append(time_hack[k])
            else: 
                k = k+1
        plt.plot(time_nozeros,slope)
        
def plot_plumes_slope(area,time,bin_min,bin_max,bin_n,prop_plumes,series=True,timestep=None):
    """
    Written by Lennéa Hayo, 19-11-28
    
    Creates a plot with slopes of plume size distribution. Can be used either for one specific timestep or for a series of timesteps.
    
    Parameters:
        area: distribution of logarithmic data (here plume sizes)
        time: at this time the slope is being calculated
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins
        prop_plumes: plumes from data_file
        series: if it is a single timestep: False, if it is a time series: True
        timestep: specific moment in time to plot dist and slope, is only used for series=False
        
    """
    plumes_time_area = prop_plumes[[time,area]]
    del(prop_plumes)
    plumes_time_area = plumes_time_area.loc[plumes_time_area[area]<25600]
    plume_times = np.unique(plumes_time_area[time])
    
    if not series:
        if timestep is None:
            raise ValueError("timestep must be set when series is False")
        #calculate slope of plumes at spezific timestep
        plume_time_area_timestep = plumes_time_area.loc[plumes_time_area[time]==plume_times[timestep-1]]
        bins_log_mm, ind, CSD = log_binner_minmax(plume_time_area_timestep[area],bin_min,bin_max,bin_n)
        x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if math.isnan(CSD[i]):
                j = 0
            elif j > global_j:
                global_j = j
                global_i = i

            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
        plt.plot(x_bins_log_mm,CSD,'-o')
        plt.plot(x_bins_log_mm,f2)
        plt.xscale('log')
        plt.yscale('log')
    else:
        #calculates slopes for a series of timesteps
        timehack = np.arange(plume_times.size)/2.+6.5
        time_nozeros_plumes = []
        slope_plumes = []
        for k in range(timehack.size):
            plume_time_area_timestep = plumes_time_area.loc[plumes_time_area[time]==plume_times[k-1]]    

            bins_log_mm, ind, CSD = log_binner_minmax(plume_time_area_timestep[area],bin_min,bin_max,bin_n)
            x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

            j = 0
            global_j = 0
            global_i = 0
            for i in range(CSD.size):
                j += 1
                if math.isnan(CSD[i]):
                        j = 0
                elif j > global_j:
                    global_j = j
                    global_i = i     
            if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
                continue
            else:
                m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
                f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
                slope_plumes.append(m2)
                time_nozeros_plumes.append(timehack[k]) 
        plt.plot(time_nozeros_plumes[:-1],slope_plumes[:-1])          