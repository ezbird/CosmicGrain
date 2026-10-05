import numpy as np
import h5py 

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as patheffects

import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from reproject import reproject_from_healpix
from astropy_healpix import HEALPix

# Define the conversion 
k=1.1
offset=0.07


def query_from_map(fname, sample_coord, nside='combined'):
    with h5py.File(fname, 'r') as f:
        distance_bins= f['distance_bins'][:]
        bin_length = distance_bins[1:]-distance_bins[:-1]
        map_nside = f[f'map_{nside}/nside'][...]
        hp_order = f[f'map_{nside}/hp_order'][...].astype('str')
        R55_map_int = f[f'map_{nside}/R55_map_int'][:]
        R55_map_diff = f[f'map_{nside}/R55_map_diff'][:]
        AV_map_int = f[f'map_{nside}/AV_map_int'][:]
        AV_map_diff = f[f'map_{nside}/AV_map_diff'][:]
        if nside=='combined':
            nside_int = f[f'map_{nside}/nside_map_int'][:]
            nside_diff = f[f'map_{nside}/nside_map_diff'][:]     

    distance_sample = sample_coord.distance.kpc
    distance_idx = np.searchsorted(distance_bins, distance_sample)-1
    hp = HEALPix(nside=map_nside, order=hp_order)
    healpix_idx = hp.lonlat_to_healpix(sample_coord.l, sample_coord.b)

    R55_int_1d = np.zeros(sample_coord.shape[0])
    R55_diff_1d = np.zeros(sample_coord.shape[0])
    AV_int_1d = np.zeros(sample_coord.shape[0])
    AV_diff_1d = np.zeros(sample_coord.shape[0])
    nside_int_1d = np.zeros(sample_coord.shape[0])
    nside_diff_1d = np.zeros(sample_coord.shape[0])
    
    out_of_range = (distance_idx>len(distance_bins)-1)
    R55_int_1d[out_of_range] = None
    R55_diff_1d[out_of_range] = None
    AV_int_1d[out_of_range] = None
    AV_diff_1d[out_of_range] = None
    nside_int_1d[out_of_range] = None
    nside_diff_1d[out_of_range] = None

    R55_int_1d[~out_of_range] = R55_map_int[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
    R55_diff_1d[~out_of_range] = R55_map_diff[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
    AV_int_1d[~out_of_range] = AV_map_int[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
    AV_diff_1d[~out_of_range] = AV_map_diff[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
    nside_int_1d[~out_of_range] = nside_int[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
    nside_diff_1d[~out_of_range] = nside_diff[distance_idx[~out_of_range], healpix_idx[~out_of_range]]
   
    return R55_int_1d, R55_diff_1d, AV_int_1d, AV_diff_1d, nside_int_1d, nside_diff_1d

def hist_mean_weighed(x, y, quantity, weight=np.array([1.]), xbins=100, ybins=50):
    
    idx_valid = np.isfinite(quantity)
    x = x[idx_valid]
    y = y[idx_valid]
    quantity = quantity[idx_valid]
    if len(weight>1):
        weight=weight[idx_valid]
    
    x_grids = np.linspace(np.min(x), np.max(x), xbins+1)
    y_grids = np.linspace(np.min(y), np.max(y), ybins+1)
    x_idx = np.searchsorted(x_grids, x)-1
    y_idx = np.searchsorted(y_grids, y)-1
    H_sum = np.zeros((xbins, ybins))
    H_num = np.zeros((xbins, ybins))
    
    np.add.at(H_num, (x_idx, y_idx), weight)
    np.add.at(H_sum, (x_idx, y_idx), quantity*weight)
    
    return H_sum, H_num, x_idx, y_idx, x_grids, y_grids

def mollweide_wcs(w, frame):
    coordsys,ctype0,ctype1 = {
        'galactic': ('GAL', 'GLON-MOL', 'GLAT-MOL'),
        'icrs':     ('EQU', 'RA---MOL', 'DEC--MOL')
    }[frame]
    target_header = dict(
        naxis=2,
        naxis1=w,
        naxis2=w//2,
        ctype1=ctype0,
        crpix1=w//2+0.5,
        crval1=0.0,
        cdelt1=-0.675*480/w,
        cunit1='deg',
        ctype2=ctype1,
        crpix2=w//4+0.5,
        crval2=0.0,
        cdelt2=0.675*480/w,
        cunit2='deg',
        coordsys=coordsys
    )
    wcs = WCS(target_header)
    return wcs


def plot_mollweide(fig, subplot, healpix_data, clabel=None, w=480,  **kwargs):
    target_header = dict(
        naxis=2,
        naxis1=w,
        naxis2=w//2,
        ctype1='GLON-MOL',
        crpix1=w//2+0.5,
        crval1=0.0,
        cdelt1=-0.675*480/w,
        cunit1='deg',
        ctype2='GLAT-MOL',
        crpix2=w//4+0.5,
        crval2=0.0,
        cdelt2=0.675*480/w,
        cunit2='deg',
        coordsys='GAL',
    )
    wcs = WCS(target_header)

    array, footprint = reproject_from_healpix(
        (healpix_data, 'galactic'),
        wcs, nested=True,
        shape_out=(w//2,w),
        order='nearest-neighbor'
    )

    
    ax = fig.add_subplot(
        subplot, projection=wcs,
        frame_class=EllipticalFrame
    )
    im = ax.imshow(array, **kwargs)

    pe = [
        patheffects.Stroke(linewidth=1.5, foreground='white'),
        patheffects.Normal()
    ]

    ax.coords.grid(color='gray', alpha=0.2)
    ax.coords['glon'].set_ticklabel(color='black', path_effects=pe, fontsize=6)
    ax.coords['glat'].set_ticklabel(color='black', path_effects=pe, fontsize=6)
    ax.coords['glat'].set_ticklabel_position('v')
    ax.coords['glon'].set_ticks_visible(False)
    ax.coords['glat'].set_ticks_visible(False)

    return fig, ax, wcs



def draw_distance_slice(fname, nside='combined'):

    with h5py.File(fname, 'r') as f:
        distance_bins= f['distance_bins'][:]
        nside_map = f[f'map_{nside}/nside'][...]
        hp_order = f[f'map_{nside}/hp_order'][...].astype('str')
        R55_map_int = f[f'map_{nside}/R55_map_int'][:]
        R55_map_diff = f[f'map_{nside}/R55_map_diff'][:]
        if nside=='combined':
            nside_int = f[f'map_{nside}/nside_map_int'][:]
            nside_diff = f[f'map_{nside}/nside_map_diff'][:]        

    Rv_center=3.1
    vmin=Rv_center-1.2
    vmax=Rv_center+1.2
  
    for i in range(len(distance_bins)-1):
        cmap=cm.coolwarm_r
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        fig = plt.figure(figsize=(6,5.5),dpi=200)
        gs = GridSpec(
            2,2,
            height_ratios=[1,1],
            width_ratios=[1,0.01],
            wspace=0.0,
            hspace=0.1,
            left=0.04,
            right=0.9,
            bottom=0.05,
            top=0.93
        )
        w=1024
        
        R55_hpix_map = R55_map_diff[i]
        fig, ax0, wcs = plot_mollweide(fig, gs[0,0], k*R55_hpix_map+offset, clabel=None, w=w, norm=norm, cmap=cmap)
        #ax0.set_title(r'$\mathrm{\ Zhang}\ \mathit{et\ al.}\ \mathrm{(in\ prep)}$')
        ax0.set_title(fr'${distance_bins[i]:.2f}<\mathrm{{d}}<{distance_bins[i+1]:.2f}\ \mathrm{{kpc}}$')

        R55_hpix_map = R55_map_int[i]
        
        fig, ax1, wcs = plot_mollweide(fig, gs[1,0], k*R55_hpix_map+offset, clabel=None, w=w, norm=norm, cmap=cmap)
        #ax0.set_title(r'$\mathrm{\ Zhang}\ \mathit{et\ al.}\ \mathrm{(in\ prep)}$')
        ax1.set_title(fr'$\mathrm{{Integrated\ (nside={nside})}}$')   
    

        ax_colorbar = fig.add_subplot(gs[:,-1])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                               orientation='vertical', cax=ax_colorbar, use_gridspec=True,   )
    
        cbar.ax.tick_params(axis='both',direction = 'in', which='major', length=2)
        cbar.set_label(r'$R_V$')
        
        if i<10:
            plt.savefig(f'slices/0{i}.png')
        else:
            plt.savefig(f'slices/{i}.png')
        plt.close()

        cmap=cm.viridis
        norm = colors.LogNorm(vmin=64, vmax=256)

        if nside=='combined':
            fig = plt.figure(figsize=(6,5.5),dpi=200)
            gs = GridSpec(
                2,2,
                height_ratios=[1,1],
                width_ratios=[1,0.01],
                wspace=0.0,
                hspace=0.1,
                left=0.04,
                right=0.9,
                bottom=0.05,
                top=0.93
            )
            w=1024
            
            fig, ax0, wcs = plot_mollweide(fig, gs[0,0], nside_diff[i], clabel=None, w=w, norm=norm, cmap=cmap)
            #ax0.set_title(r'$\mathrm{\ Zhang}\ \mathit{et\ al.}\ \mathrm{(in\ prep)}$')
            ax0.set_title(fr'${distance_bins[i]:.2f}<\mathrm{{d}}<{distance_bins[i+1]:.2f}\ \mathrm{{kpc}}$')
    
            R55_hpix_map = R55_map_int[i]
            
            fig, ax1, wcs = plot_mollweide(fig, gs[1,0], nside_int[i], clabel=None, w=w, norm=norm, cmap=cmap)
            #ax0.set_title(r'$\mathrm{\ Zhang}\ \mathit{et\ al.}\ \mathrm{(in\ prep)}$')
            ax1.set_title(fr'$\mathrm{{Integrated\ (nside={nside})}}$')   
        
    
            ax_colorbar = fig.add_subplot(gs[:,-1])
            cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                                   orientation='vertical', cax=ax_colorbar, use_gridspec=True,   )
            cbar.ax.set_yticks([64,128,256], ['64', '128', '256'])        
            cbar.ax.tick_params(axis='both',direction = 'in', which='major', length=2)
            cbar.set_label(r'$\mathrm{nside}$')
            
            if i<10:
                plt.savefig(f'slices/nside_0{i}.png')
            else:
                plt.savefig(f'slices/nside_{i}.png')
            plt.close()


if __name__=='__main__':
    nside = 'combined'
    draw_distance_slice('./Rv_map_new.h5', nside=nside)

    # Generate a sample 
    nsample = int(1e8)
    x_sample = (np.random.rand(nsample)-0.5)*6
    y_sample = (np.random.rand(nsample)-0.5)*6
    z_sample = (np.random.rand(nsample)-0.5)*0.8

    # Select by projected distance
    proj_distance= (x_sample**2+y_sample**2)**0.5
    within_circle = (proj_distance<2.6)

    x_sample = x_sample[within_circle]
    y_sample = y_sample[within_circle]
    z_sample = z_sample[within_circle]
    
    sample_coord = SkyCoord(u=x_sample*u.kpc, v=y_sample*u.kpc, w=z_sample*u.kpc, unit='kpc', frame='galactic',
                         representation_type='cartesian')
    sample_coord = (sample_coord.icrs).galactic
    
    R55_int_1d, R55_diff_1d, AV_int_1d, AV_diff_1d, nside_int_1d, nside_diff_1d = query_from_map('./Rv_map_new.h5', sample_coord, nside=nside)
        
    H_sum, H_num, x_idx, y_idx, x_grids, y_grids = hist_mean_weighed(x_sample, 
                                                                     y_sample, 
                                                                     k*R55_diff_1d+offset, 
                                                                     weight=AV_diff_1d, 
                                                                     xbins=200, 
                                                                     ybins=200)
    density_show = H_sum/(H_num+0.000001)
    density_show[H_num<200]=None

    Rv_center=3.1
    vmin=Rv_center-0.8
    vmax=Rv_center+0.8
    
    cmap=cm.coolwarm_r
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    plt.imshow(density_show.T[::-1], 
               extent=(x_grids[0], x_grids[-1], y_grids[0], y_grids[-1]),
               cmap=cmap,
               norm=norm,
              )
    plt.xlabel('$x\,\mathrm{(kpc)}$')
    plt.ylabel('$y\,\mathrm{(kpc)}$')
    cbar = plt.colorbar(norm=norm,cmap=cmap)
    cbar.ax.tick_params(axis='both',direction = 'in', which='major', length=2)
    cbar.set_label(r'$R(V)$')
    plt.savefig('example_birds_eye_RV.png')
    plt.close()


    H_sum, H_num, x_idx, y_idx, x_grids, y_grids = hist_mean_weighed(x_sample, 
                                                                     y_sample, 
                                                                     AV_diff_1d, 
                                                                     weight=np.ones(AV_diff_1d.shape), 
                                                                     xbins=200, 
                                                                     ybins=200)
    density_show = H_sum/(H_num+0.000001)
    density_show[H_num<10]=None
    
    cmap=cm.inferno
    norm = colors.Normalize(vmin=0,vmax=1)
    
    plt.imshow(density_show.T[::-1], 
               extent=(x_grids[0], x_grids[-1], y_grids[0], y_grids[-1]),
               cmap=cmap,
               norm=norm,
              )
    plt.xlabel('$x\,\mathrm{(kpc)}$')
    plt.ylabel('$y\,\mathrm{(kpc)}$')
    cbar = plt.colorbar(norm=norm,cmap=cmap)
    cbar.ax.tick_params(axis='both',direction = 'in', which='major', length=2)
    cbar.set_label(r'$\Delta A(V)/\Delta d\, \mathrm{(mag/kpc)}$')
    plt.savefig('example_birds_eye_AV.png')
    plt.close()
