#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""HMF plots"""

import numpy as np
import utilities_statistics as us
import common
import functools


##################################
mlow = 8
mupp = 15
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xmf = mbins + dm/2.0

# Constants
GyrToYr = 1e9
zsun = 0.0189
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

ssfr_thresh = 10**(-10.75)

def plot_individual_seds(plt, outdir, obsdir, h0, total_sfh_z0, gal_props_z0, LBT, redshift, bhid, bhm, mdot_bh, total_sm):

    ############### plot star formation histories ##################################################
    xtit="$\\rm LBT/Gyr$"
    ytit="$\\rm log_{10}(SFR/M_{\odot} yr^{-1})$"

    lbt_target = us.look_back_time(redshift)
    xmax = 2 #max(LBT) - lbt_target

    xmin, ymin, ymax = 0, -3.2, 3.5
    xleg = xmax + 0.025 * (xmax-xmin)
    yleg = ymax - 0.07 * (ymax-ymin)

    fig = plt.figure(figsize=(6.5,5))
    mbins = (10.0,10.1,10.2,10.3,10.4,10.5,10.75,12.5)
    labels = ['10.05', '10.15', '10.25', '10.35', '10.45', '10.65', '11']
    colors = ('Navy','Blue','RoyalBlue','SkyBlue','Teal','DarkTurquoise','Aquamarine','Yellow', 'Gold',  'Orange','OrangeRed', 'LightSalmon', 'Crimson', 'Red', 'DarkRed')
    colors = ('Navy','SkyBlue','Aquamarine','Gold','Orange', 'LightSalmon','Red')
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 1, 1))
    ngals = np.zeros ( shape = len(mbins)-1)
    N_max = 2000
    sfh_all_gals = np.zeros(shape = (N_max, len(LBT)))
    smg_all_gals = np.zeros(shape = (N_max, len(LBT)))

    bh_mass_growth = np.zeros ( shape = (len(mbins)-1, N_max, len(LBT)))
    bh_rate = np.zeros ( shape = (len(mbins)-1, N_max, len(LBT)))
    total_sm_q = np.zeros ( shape = (len(mbins)-1, N_max, len(LBT)))
    for j in range(0,len(mbins)-1):
        ax.text(xmin + j*0.25,3.6, labels[j], fontsize=11, color=colors[j])
        ind = np.where((gal_props_z0[:,1] >= 10**mbins[j]) & (gal_props_z0[:,1] < 10**mbins[j+1]) & (gal_props_z0[:,3]/gal_props_z0[:,1] <= ssfr_thresh) & (gal_props_z0[:,4] == 0))
        if(len(gal_props_z0[ind]) > 0):
           m_in = gal_props_z0[ind,1]
           ssfr_in = gal_props_z0[ind,3]/gal_props_z0[ind,1]
           type_gin = gal_props_z0[ind,4]
           print("Number of passive galaxies in mass bin", j, " is", len(gal_props_z0[ind]))
           ids_quench = gal_props_z0[ind,5]
           ids_quench = ids_quench[0]
           age_selec= gal_props_z0[ind,0]
           tot_sfh_selec = total_sfh_z0[ind,:]
           tot_sfh_selec = tot_sfh_selec[0,:]
           tot_sm_selec = total_sm[ind,:]
           tot_sm_selec = tot_sm_selec[0,:]
           age_selec     = gal_props_z0[ind,0]
           typesg        = gal_props_z0[ind,4]
           numgals = len(age_selec[0])
           ngals[j] = numgals
           total_sm_q[j,0:numgals,:] = tot_sm_selec
           nsnaps = len(tot_sfh_selec[0,:]) 

           nloop = numgals
           if(numgals > N_max):
               nloop = N_max
           for gal in range(0,nloop): 
               sfh_in = tot_sfh_selec[gal,:]
               smg_in = tot_sm_selec[gal,:]
               if(j > 0):
                  sfh_all_gals[gal + int(ngals[j]),:] = sfh_in[:]
                  smg_all_gals[gal + int(ngals[j]),:] = smg_in[:]
               else:
                  sfh_all_gals[gal,:] = sfh_in[:]
                  smg_all_gals[gal,:] = smg_in[:]

               lowsfr = np.where(sfh_in < 1e-4)
               sfh_in[lowsfr] = 1e-4
               ax.plot(LBT - lbt_target, np.log10(sfh_in), color=colors[j], linewidth=1)

               #select BH of interest
               select = np.where(bhid == ids_quench[gal])
               bhh_selected = bhm[select,:]
               bhm_selected = mdot_bh[select,:]
               bh_mass_growth[j,gal,:] = bhh_selected[0]
               bh_rate[j,gal,:] = bhm_selected[0]
    #common.prepare_legend(ax, colors, bbox_to_anchor=(0.98, 0.1))
    ax.plot([xmin,xmax], [np.log10(300.0),np.log10(300.0)], ls='solid',color='gray')
    ax.text(xmin+0.1,np.log10(300.0)+0.1, "$\\rm 300\\, M_{\\odot}\\, yr^{-1}$", fontsize=11, color='gray')
    ax.text(1.3,3, "SHARK", fontsize=13)
    plt.tight_layout()
    common.savefig(outdir, fig, "SFHs_massive_passive_galaxies_z"+str(redshift)+"_centrals.pdf")

    ############### plot stellar mass growth histories #################################################################
    ytit="$\\rm log_{10}(M_{\\rm stars}/M_{\odot})$"
    ymin, ymax = 8,12
    fig = plt.figure(figsize=(6.5,5))

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 1, 1))

    for j in range(0,len(mbins)-1):
        ax.text(xmin + j*0.25, 12.05, labels[j], fontsize=11, color=colors[j])

        for gal in range(0,int(ngals[0])):
            ax.plot(LBT - lbt_target, np.log10(total_sm_q[j,gal,:]), color=colors[j], linewidth=1)
    ax.text(1.3,11.7, "SHARK", fontsize=13)
    ax.plot([xmin,xmax], [10,10], ls='solid',color='gray')
    plt.tight_layout()
    common.savefig(outdir, fig, "SMGrowth_massive_passive_galaxies_z"+str(redshift)+"_centrals.pdf")


    ############### plot BH growth histories #################################################################
    ytit="$\\rm log_{10}(M_{\\rm BH}/M_{\odot})$"
    ymin, ymax = 5,9.5
    fig = plt.figure(figsize=(6.5,5))

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 1, 1))

    for j in range(0,len(mbins)-1):
        ax.text(xmin + j*0.25, 9.6, labels[j], fontsize=11, color=colors[j])

        for gal in range(0,int(ngals[0])):
            ax.plot(LBT - lbt_target, np.log10(bh_mass_growth[j,gal,:]), color=colors[j], linewidth=1)

    plt.tight_layout()
    common.savefig(outdir, fig, "BHHs_massive_passive_galaxies_z"+str(redshift)+"_centrals.pdf")

    ############### plot BH mass rate histories #################################################################
    ytit="$\\rm log_{10}(\\dot{M}_{\\rm BH}/M_{\odot} yr^{-1})$"
    ymin, ymax = -4,1
    fig = plt.figure(figsize=(6.5,5))

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.5, 0.5, 1, 1))

    for j in range(0,len(mbins)-1):
        ax.text(xmin + j*0.25, 1.1, labels[j], fontsize=11, color=colors[j])
        for gal in range(0,int(ngals[0])):
            ax.plot(LBT - lbt_target, np.log10(bh_rate[j,gal,:]), color=colors[j], linewidth=1)

    ax.plot([xmin,xmax], [np.log10(0.17),np.log10(0.17)], ls='solid',color='gray')
    ax.text(xmin+0.1,np.log10(0.17)+0.1, "$\\rm 10^{45}\\, erg\\,s^{-1}$", fontsize=11, color='gray')

    plt.tight_layout()
    common.savefig(outdir, fig, "BHHs_rate_massive_passive_galaxies_z"+str(redshift)+"_centrals.pdf")


def plot_age_gals(plt, outdir, obsdir, h0, gal_props_z0, redshift, LBT):


    lbt_target = us.look_back_time(redshift)

    ############### plot ages_galaxies ##################################################
    xtit="$\\rm log_{10}(M_{\\rm stars}/M_{\odot})$"
    ytit="$\\rm age_{\\rm 50,90}/Gyr$"

    xmin, xmax, ymin, ymax = 9.9, 12, 0, 2 #max(LBT) - lbt_target
    xleg = xmax + 0.025 * (xmax-xmin)
    yleg = ymax - 0.07 * (ymax-ymin)

    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.25, 0.25, 0.25, 0.25))

    #select central, massive and passive galaxies
    ind = np.where((gal_props_z0[:,1] >= 1e10) & (gal_props_z0[:,3]/gal_props_z0[:,1] <= ssfr_thresh) & (gal_props_z0[:,4] == 0))
    m_in = np.log10(gal_props_z0[ind,1])
    n_gal = len(m_in[0])
    age_50 = gal_props_z0[ind,6]
    age_80 = gal_props_z0[ind,7]

    ax.plot(m_in[0], age_50[0] - lbt_target, ls=None, linewidth=0, marker='s', color='red', label = '$\\rm age_{50}$')
    ax.plot(m_in[0], age_80[0] - lbt_target, ls=None, linewidth=0, marker='o', color='blue', label = '$\\rm age_{80}$')

    m_in = m_in[0]
    age_50 = age_50[0]
    age_80 = age_80[0]
    for j in range(0,n_gal):
        ax.plot([m_in[j],m_in[j]], [age_50[j] - lbt_target,age_80[j] - lbt_target],ls='dotted', color='grey',linewidth=1)

    common.prepare_legend(ax, ['red','blue'], loc = 4)
    ax.text(10,1.7, "SHARK", fontsize=13)

    plt.tight_layout()
    common.savefig(outdir, fig, "age_mass_passive_galaxies_z"+str(redshift)+"_centrals.pdf")

def prepare_data(hdf5_data, sfh, id_gal, bh_id, bhm, mdothh, mdotsb, index, LBT, delta_t):
   
    #star_formation_histories and SharkSED have the same number of galaxies in the same order, and so we can safely assume that to be the case.
    #to select the same galaxies in galaxies.hdf5 we need to ask for all of those that have a stellar mass > 0, and then assume that they are in the same order.
    bin_it   = functools.partial(us.wmedians, xbins=xmf)
    (h0, volh, mdisk, mbulge, mhalo, mshalo, typeg, age, 
     sfr_disk, sfr_burst, mbh, mhot, mreheated) = hdf5_data
    vol = volh / h0**3

    (bulge_diskins_hist, bulge_mergers_hist, disk_hist) = sfh
    (bhh_hist) = bhm
    (mdot_bh_hh) = mdothh
    (mdot_bh_sb) = mdotsb
    sfr_tot = (sfr_disk + sfr_burst)/1e9/h0
    mdot_bh = mdothh[0] + mdotsb[0]

    #components:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    #ignore last band which is the top-hat UV of high-z LFs.
    ind = np.where(((mdisk + mbulge)/h0 >= 1e10) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10)))
    ngals       = len(mdisk[ind])
    print("Number of galaxies with Mstar >= 1e10 and sSFR<10^(-10) yr^(-1)", ngals)

    ind = np.where(((mdisk + mbulge)/h0 >= 1e10) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10.5)))
    ngals       = len(mdisk[ind])
    print("Number density of galaxis with Mstar >= 1e10 and sSFR<10^(-10.5) yr^(-1)", ngals/vol)
    ind = np.where(((mdisk + mbulge)/h0 >= 10**(10.5)) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10.5)))
    ngals       = len(mdisk[ind])
    print("Number density of galaxis with Mstar >= 10^10.5 and sSFR<10^(-10.5) yr^(-1)", ngals/vol)
    ind = np.where(((mdisk + mbulge)/h0 >= 10**(10.75)) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10.5)))
    ngals       = len(mdisk[ind])
    print("Number density of galaxis with Mstar >= 10^10.75 and sSFR<10^(-10.5) yr^(-1)", ngals/vol)
    ind = np.where(((mdisk + mbulge)/h0 >= 10**(10.9)) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10.5)))
    ngals       = len(mdisk[ind])
    print("Number density of galaxis with Mstar >= 10^(10.9) and sSFR<10^(-10.5) yr^(-1)", ngals/vol)

    ind = np.where(((mdisk + mbulge)/h0 >= 1e11) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10.5)))
    ngals       = len(mdisk[ind])
    print("Number density of galaxis with Mstar >= 1e11 and sSFR<10^(-10.5) yr^(-1)", ngals/vol)

    
    ind = np.where(mdisk + mbulge > 0)
    mtot        = (mdisk[ind] + mbulge[ind])/h0 #total mass
    ngals       = len(mdisk[ind])
    nsnap       = len(bulge_diskins_hist[0,:])
    print("Number of snapshots", nsnap)
    total_sfh = np.zeros(shape = (ngals, nsnap))
    total_sm  = np.zeros(shape = (ngals, nsnap)) 
    sb_sfh    = np.zeros(shape = (ngals, nsnap))
    disk_sfh  = np.zeros(shape = (ngals, nsnap))
    gal_props = np.zeros(shape = (ngals, 8))

    gal_props[:,0] = 13.6-age[ind]
    gal_props[:,1] = (mdisk[ind] + mbulge[ind])/h0
    gal_props[:,2] = mbulge[ind] / (mdisk[ind] + mbulge[ind])
    gal_props[:,3] = (sfr_burst[ind] + sfr_disk[ind])/1e9/h0
    gal_props[:,4] = typeg[ind]
    gal_props[:,5] = id_gal[ind]

    mdisk = mdisk[ind]
    mbulge = mbulge[ind]
    sfr_burst = sfr_burst[ind]
    sfr_disk = sfr_disk[ind]
    typeg = typeg[ind]

    for s in range(0,nsnap):
        total_sfh[:,s] = (bulge_diskins_hist[:,s] + bulge_mergers_hist[:,s] + disk_hist[:,s])/h0 #in Msun/yr
        sb_sfh[:,s]    = (bulge_diskins_hist[:,s] + bulge_mergers_hist[:,s])/h0
        disk_sfh[:,s]  = (disk_hist[:,s])/h0
        for j in range(0,ngals):
            if(s == 0):
                total_sm[j,s] = (total_sfh[j,s] * 1e9 * delta_t[s]) * (1 - 0.4588) # mass remaining
            else:
                total_sm[j,s] = total_sm[j,s-1] + (total_sfh[j,s] * 1e9 * delta_t[s]) * (1 - 0.4588) #mass remaining
            if((total_sm[j,s] > 0.5 * mtot[j]) & (gal_props[j,6] == 0)):
                gal_props[j,6] = LBT[s]
            if((total_sm[j,s] > 0.8 * mtot[j]) & (gal_props[j,7] == 0)):
                gal_props[j,7] = LBT[s]

    ind = np.where(((mdisk + mbulge)/h0 >= 1e10) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) < 10**(-10)))
    galsprops_tosave = np.zeros(shape = (len(mdisk[ind]), 3))
    galsprops_tosave[:,0] = (mdisk[ind] + mbulge[ind])/h0
    galsprops_tosave[:,1] = (sfr_burst[ind] + sfr_disk[ind])/1e9/h0
    galsprops_tosave[:,2] = typeg[ind]
    sfh_print = total_sfh[ind,:]
    sm_print = total_sm[ind,:]
    np.savetxt('PassiveGalaxies_PMill2.txt', galsprops_tosave)
    np.savetxt('SFHs_PassiveGalaxies_PMill2.txt', sfh_print[0])
    np.savetxt('Masshistory_PassiveGalaxies_PMill2.txt', sm_print[0])
    np.savetxt('LBT_PMill.txt', LBT)

    ind = np.where(((mdisk + mbulge)/h0 >= 1e10) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) > 10**(-9.9)) & (typeg == 0))
    ssfr_ms = np.log10(np.median((sfr_burst[ind] + sfr_disk[ind])/1e9/(mdisk[ind] + mbulge[ind])))

    ind = np.where(((mdisk + mbulge)/h0 >= 1e10) & ((sfr_burst+ sfr_disk)/1e9/(mdisk + mbulge) > 10**(ssfr_ms-0.2)) & (typeg == 0))
    galsprops_tosave = np.zeros(shape = (len(mdisk[ind]), 3))
    galsprops_tosave[:,0] = (mdisk[ind] + mbulge[ind])/h0
    galsprops_tosave[:,1] = (sfr_burst[ind] + sfr_disk[ind])/1e9/h0
    galsprops_tosave[:,2] = typeg[ind]
    sfh_print = total_sfh[ind,:]
    sm_print = total_sm[ind,:]
    np.savetxt('ActiveGalaxies_PMill.txt', galsprops_tosave)
    np.savetxt('SFHs_ActiveGalaxies_PMill.txt', sfh_print[0])
    np.savetxt('Masshistory_ActiveGalaxies_PMill.txt', sm_print[0])

    return (total_sfh, sb_sfh, disk_sfh, gal_props, bh_id, bhh_hist[0], mdot_bh, total_sm)

def main(model_dir, outdir, redshift_table, subvols, obsdir):

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    fields = {'galaxies': ('mstars_disk', 'mstars_bulge', 'mvir_hosthalo',
                           'mvir_subhalo', 'type', 'mean_stellar_age',
                           'sfr_disk', 'sfr_burst', 'm_bh',
                           'mhot','mreheated')}


    sfh_fields = {'bulges_diskins': ('star_formation_rate_histories'),
                  'bulges_mergers': ('star_formation_rate_histories'),
                  'disks': ('star_formation_rate_histories')}
    gal_idx = {'galaxies': ('id_galaxy', 'id_subhalo_tree')}

    bhh_idx = {'galaxies': ('id_galaxy')}
    bhh_mass = {'galaxies': ('m_bh_history')}
    bh_mdothh = {'galaxies': ('bh_accretion_rate_hh_history')}
    bh_mdotsb = {'galaxies': ('bh_accretion_rate_sb_history')}

    z = [3] #0.5, 1, 1.5, 2, 3)
    snapshots = redshift_table[z]

    # Create histogram
    for index, snapshot in enumerate(snapshots):
        bh_ids_all = None
        gal_ids_all = None
        for j, ivol in enumerate(subvols):
            bh_id, delta_t, LBT = common.read_bhh(model_dir, snapshot, bhh_idx, [ivol])
            galid, subid = common.read_data(model_dir, snapshot, gal_idx, [ivol], include_h0_volh = False)
            subid = int((ivol+1) * 1e8)
            bh_id = bh_id[0] + subid
            galid = galid + subid
            if(j == 0):
               bh_ids_all = bh_id
               gal_ids_all = galid
            else:
               gal_ids_all = np.concatenate([gal_ids_all, galid])
               bh_ids_all = np.concatenate([bh_ids_all, bh_id]) 
        hdf5_data = common.read_data(model_dir, snapshot, fields, subvols)
        sfh, delta_t, LBT = common.read_sfh(model_dir, snapshot, sfh_fields, subvols)
        bhm, delta_t, LBT = common.read_bhh(model_dir, snapshot, bhh_mass, subvols)
        mdothh, delta_t, LBT = common.read_bhh(model_dir, snapshot, bh_mdothh, subvols)
        mdotsb, delta_t, LBT = common.read_bhh(model_dir, snapshot, bh_mdotsb, subvols)


        (total_sfh, sb_sfh, disk_sfh, gal_props, bhid, bhm, mdot_bh, total_sm) = prepare_data(hdf5_data, sfh, gal_ids_all, bh_ids_all, bhm, mdothh, mdotsb, index, LBT, delta_t)

        h0, volh = hdf5_data[0], hdf5_data[1]
        plot_individual_seds(plt, outdir, obsdir, h0, total_sfh, gal_props, LBT, z[index], bhid, bhm, mdot_bh, total_sm)
        plot_age_gals(plt, outdir, obsdir, h0, gal_props, z[index], LBT)


if __name__ == '__main__':
    main(*common.parse_args())
