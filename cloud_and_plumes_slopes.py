import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from random import random
import matplotlib
from netCDF4 import Dataset
import sys
from slopes_and_binning import *

import pandas as pd

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
        prop_plumes:
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