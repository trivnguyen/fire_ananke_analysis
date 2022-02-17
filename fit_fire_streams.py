import numpy as np
import deepdish as dd
import gizmo_analysis as ga
import utilities as ut
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde

import read_fire_streams


def plot_pretty(dpi=175, fontsize=15):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def load_potential(halo='m12i_res7100', sim_dir='/work/08412/tg877111/stampede2/data/fire_potentials/', nsnap=600):
    try:
        import agama
    except:
        print('Agama error.')
        return

    # From notebook by Arpit Arora
    symmlabel = {'a': 'axi', 'n': 'none'}  # axisymmetry or no symmetry
    # pick the required symmetry and multipole coef.
    mult_l = 4  # only option
    sym = 'a'  # axisymmetry, gives the ability to compute actions

    sim_dir = sim_dir + '%s/' % halo
    print('Loading multipole and cubicspline files')
    try:
        pxr_DM = agama.Potential(file=f'{sim_dir}{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul_DR')  # DM halo + cold gas (multipole expansion)
        pxr_bar = agama.Potential(file=f'{sim_dir}{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp_DR')  # bar + hot gas (CubicSpline expansion)
    except:
        convert_potential(halo=halo, nsnap=nsnap)
        pxr_DM = agama.Potential(file=f'{sim_dir}{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul_DR')  # DM halo + cold gas (multipole expansion)
        pxr_bar = agama.Potential(file=f'{sim_dir}{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp_DR')  # bar + hot gas (CubicSpline expansion)
    print('Files loaded!')
    pot_model = agama.Potential(pxr_DM, pxr_bar)
    pot_model  # a potential class object that can be used to determine potential, density and other properties for our pot_model
    return pot_model


def convert_potential(halo='m12i_res7100', nsnap=600, sim_dir='/work/08412/tg877111/stampede2/data/fire_potentials/'):
    # From notebook by Arpit Arora
    symmlabel = {'a': 'axi', 'n': 'none'}  # axisymmetry or no symmetry
    # pick the required symmetry and multipole coef.
    mult_l = 4  # only option
    sym = 'a'  # axisymmetry, gives the ability to compute actions

    sim_dir = sim_dir + '%s/' % halo
    sim_dir_old = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/'
    sim_dir_old = sim_dir_old + '%s/' % halo + '/potential/10kpc/'
    fdm_old = f'{sim_dir_old}{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul'
    fbar_old = f'{sim_dir_old}{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp'
    fdm = f'{sim_dir}{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul'
    fbar = f'{sim_dir}{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp'
    os.system('cp %s %s' % (fdm_old, fdm))
    os.system('cp %s %s' % (fbar_old, fbar))
    os.system('perl ~/code/Agama/data/convertcoefs.pl %s' % fdm)
    os.system('perl ~/code/Agama/data/convertcoefs.pl %s' % fbar)
    os.system('mv %s.ini %s_DR' % (fdm, fdm))
    os.system('mv %s.ini %s_DR' % (fbar, fbar))
    os.system('rm %s' % fdm)
    os.system('rm %s' % fbar)


def fit_all_streams(time=7, n=100, plotting=True, return_orbits=False, return_peris_apos=True, stream_version=3):
    halos = read_fire_streams.get_halos()
    peris = []
    apos = []

    for h in halos:
        try:
            print('LOADING IN fit_all_streams')
            snap = read_fire_streams.load_halo(h)
        except:
            continue
        pi, ai = fit_streams(halo=h, snap=snap, time=time, n=n, plotting=plotting, return_orbits=return_orbits, return_peris_apos=return_peris_apos, stream_version=stream_version)
        peris.append(pi)
        apos.append(ai)
        plt.close('all')

    if return_peris_apos:
        return peris, apos


def fit_streams(stream_ids='all', halo='m12i_res7100', elvis_host=None, snap=None, potential=None, time=7, n='all', plotting=False, return_orbits=True, return_peris_apos=True, stream_version=3):
    if snap is None:
        print('LOADING IN fit_streams')
        snap = read_fire_streams.load_halo(halo)
    if potential is None:
        potential = load_potential(halo)

    coords = read_fire_streams.get_star_coords(halo, elvis_host, snap, stream_version=stream_version)

    if stream_ids == 'all':
        stream_ids = np.arange(len(coords))

    orbits = []
    peris = []
    apos = []
    for stream_id in stream_ids:
        xvi = coords[stream_id]
        if n != 'all' and n < len(xvi):
            indices = np.arange(len(xvi))
            indices = np.random.choice(indices, replace=False, size=n)
            xvi = xvi[indices, :]
        print(xvi.shape)

        x0 = np.median(xvi[:, 0])
        idx = np.argmin(np.abs(xvi[:, 0] - x0))
        xv0 = xvi[idx]

        orbit = agama.orbit(ic=xvi, potential=potential, time=time, trajsize=100)
        orbit0 = agama.orbit(ic=xv0, potential=potential, time=time, trajsize=100)
        orbits.append((orbit, orbit0))

        if return_peris_apos:
            t = orbit[0][0]
            r = np.zeros((orbit.shape[0], orbit[0][0].shape[0]))
            pi = []
            ai = []
            for i in range(len(orbit)):
                ri = np.linalg.norm(orbit[i][1][:, :3], axis=1)
                r[i] = ri
                pi.append(np.min(ri))
                ai.append(np.max(ri))
            np.save('output/peris_agama_%s_%i.npy' % (halo, stream_id), pi)
            np.save('output/apos_agama_%s_%i.npy' % (halo, stream_id), ai)
            peris.append(np.median(pi))
            apos.append(np.median(ai))
            # if np.median(pi) > 50:
            # print('Large peri! Stream id = %i' %stream_id)
            # peris.append(pi)
            # apos.append(ai)

        if plotting:
            plot_orbit(orbit, orbit0, time, stream_id, halo, stream_version=stream_version)

    if return_peris_apos and return_orbits:
        return orbits, peris, apos
    elif return_orbits:
        return orbits
    elif return_peris_apos:
        return peris, apos


def plot_orbit(orbit, orbit0, time, stream_id, halo='m12i_res7100', stream_version=3):
    plt.figure()

    for i in range(len(orbit)):
        plt.plot(orbit[i][1][:, 0], orbit[i][1][:, 1], lw=0.5, alpha=0.8)
    plt.plot(orbit0[1][:, 0], orbit0[1][:, 1], c='k', lw=3, zorder=1000)

    x0 = []
    y0 = []
    for i in range(orbit.shape[0]):
        x0.append(orbit[:, 1][i][0][0])
        y0.append(orbit[:, 1][i][0][1])
    plt.scatter(x0, y0, c='k', s=5)

    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    axlim = np.max([np.abs(ymin), np.abs(ymax), np.abs(xmin), np.abs(xmax)])
    plt.xlim(-axlim, axlim)
    plt.ylim(-axlim, axlim)
    plt.title('%s, stream %i' % (halo, stream_id))
    plt.savefig('../plots/%s_v%i_s%i_%.1fGyr_orbit.png' % (halo, stream_version, stream_id, time))

    plt.figure()
    t = orbit[0][0]
    r = np.zeros((orbit.shape[0], orbit[0][0].shape[0]))
    peris = []
    apos = []
    for i in range(len(orbit)):
        ri = np.linalg.norm(orbit[i][1][:, :3], axis=1)
        r[i] = ri
        peris.append(np.min(ri))
        apos.append(np.max(ri))
        plt.plot(orbit[i][0], ri, lw=0.5, alpha=0.4, zorder=0)

    # percs = np.percentile(r, [16, 50, 84], axis=0)
    # plt.plot(t, percs[1], 'C0', ls='--', zorder=901)
    # plt.fill_between(t, percs[0], percs[2], color='C0', alpha=0.4, zorder=900)

    p_peri = np.percentile(peris, [16, 50, 84])
    p_apo = np.percentile(apos, [16, 50, 84])

    plt.axhline(p_peri[1], c='C0', lw=2, ls='--', zorder=901)
    plt.axhline(p_apo[1], c='C0', lw=2, ls='--', zorder=901)
    plt.axhspan(p_peri[0], p_peri[2], color='C0', alpha=0.4, zorder=900)
    plt.axhspan(p_apo[0], p_apo[2], color='C0', alpha=0.4, zorder=900)

    r0 = np.linalg.norm(orbit0[1][:, :3], axis=1)
    plt.plot(t, r0, lw=3, zorder=1000, c='k')

    plt.xlabel('t (Gyr)')
    plt.ylabel('d (kpc)')
    plt.title('%s, stream %i' % (halo, stream_id))
    plt.savefig('../plots/%s_%i_s%i_%.1fGyr_orbit_dist.png' % (halo, stream_version, stream_id, time))


def plot_peri_apo_dists(stream_version=3, elvis=False, ebars=True, method='last'):
    halos = read_fire_streams.get_halos(elvis=elvis)

    if method not in ['last', 'min_max']:
        print('Invalid method, setting method = last')
        method = 'last'

    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']
    markers = ['^', 'o', 's', 'p', '*', 'D', 'v']

    xmin, xmax = 0, 200
    ymin, ymax = 0, 350

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(r'$r_{\rm peri}\ \mathrm{(kpc)}$')
    ax.set_ylabel(r'$r_{\rm apo}\ \mathrm{(kpc)}$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(0, xmax - 10, 50))
    ax.set_yticks(np.arange(0, ymax - 10, 50))

    pmw = [5.8, 3.8, 17.6, 12.1, 15.7, 12.7, 15.9, 16.0, 9.1, 12.3, 10.7, 13.1, 8.3, 17.]  # from Li21 + Sgr
    amw = [45.8, 13.7, 37.9, 38.2, 35.9, 19.1, 43.9, 58.2, 27.8, 17.9, 39.0, 60.3, 35.2, 80.]

    ax.plot(pmw, amw, 'kx', ms=8, label='Milky Way')
    x_bins = np.linspace(xmin, xmax, 45)
    y_bins = np.linspace(ymin, ymax, 25)
    ax_histx.hist(pmw, x_bins, histtype='step', density=True, lw=2, ls='-', color='k', zorder=1000)
    ax_histy.hist(amw, y_bins, histtype='step', density=True, lw=2, ls='-', color='k', zorder=1000, orientation='horizontal')

    mps_outer = []
    mas_outer = []
    # for j, h in enumerate(halos):
    j = 0
    jj = 0
    elvis_host = 1
    while True:
        if j >= len(halos):
            break
        h = halos[j]
        # for j, h in enumerate(['m12f_res7100']):
        print(j, h)

        if 'elvis' in h:
            f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_host%i_*.npy' % (stream_version, method, h, elvis_host))
            f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_host%i_*.npy' % (stream_version, method, h, elvis_host))
            f_peris.sort()
            f_apos.sort()
        else:
            f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_*.npy' % (stream_version, method, h))
            f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_*.npy' % (stream_version, method, h))
            f_peris.sort()
            f_apos.sort()

        mps = []
        mas = []
        sps = []
        sas = []
        for i in range(len(f_peris)):
            peris = np.load(f_peris[i])
            apos = np.load(f_apos[i])
            cut = (peris > 100) & (np.abs(peris - apos) < 30)
            peris[cut] = np.nan
            apos[cut] = np.nan
            print(f_peris[i], np.nanmedian(peris), np.nanmedian(apos))
            mps.append(np.nanmedian(peris))
            mas.append(np.nanmedian(apos))
            mps_outer.append(np.nanmedian(peris))
            mas_outer.append(np.nanmedian(apos))

            if ebars:
                p_percs = np.nanpercentile(peris, [16, 50, 84])
                a_percs = np.nanpercentile(apos, [16, 50, 84])
                sps.append((p_percs[1] - p_percs[0], p_percs[2] - p_percs[1]))
                sas.append((a_percs[1] - a_percs[0], a_percs[2] - a_percs[1]))

        sps = np.asarray(sps)
        sas = np.asarray(sas)

        # ax.plot(mps, mas, marker=markers[j], ms=6, color='C1', linestyle=None)
        label = h.split('_')[0]
        if 'elvis' in h:
            label = h.split('_')[2] + ' %i' % elvis_host
        if ebars:
            ax.errorbar(mps, mas, xerr=sps.T, yerr=sas.T, fmt='.', color=colors[jj], label='%s' % label)
        else:
            ax.plot(mps, mas, marker=markers[j], ms=7, color=colors[jj], linestyle='none', label='%s' % label)

        if 'elvis' in h and elvis_host == 1:
            elvis_host = 2
        elif 'elvis' in h and elvis_host == 2:
            elvis_host = 1
            j += 1
        else:
            j += 1
        jj += 1

    ax_histx.hist(mps_outer, x_bins, histtype='step', density=True, lw=2, ls='-', color='C0')
    ax_histy.hist(mas_outer, y_bins, histtype='step', density=True, lw=2, ls='-', color='C0', orientation='horizontal')

    plt.sca(ax)
    plt.legend(loc='center right', fontsize=12, frameon=False)

    # ax.set_xlim(0,200)
    # ax.set_ylim(0,300)

    plt.tight_layout()
    if ebars:
        plt.savefig('../plots/peris_apos/snap_peris_apos_all_v%i_%s_ebars.png' % (stream_version, method), bbox_inches='tight')
    else:
        plt.savefig('../plots/peris_apos/snap_peris_apos_all_v%i_%s.png' % (stream_version, method), bbox_inches='tight')


def plot_stars_peris_apos(halo='m12i_res7100', elvis_host=1, stream_version=3, ebars=False, method='last', prog_only=False, prog_percentile=99):
    if method not in ['last', 'min_max']:
        print('Invalid method, setting method = last')
        method = 'last'

    # f_peris = glob.glob(data_dir + 'peris_%s_*.npy' % halo)
    # f_apos = glob.glob(data_dir + 'apos_%s_*.npy' % halo)
    if 'elvis' in halo:
        f_peris = glob.glob('output/streams_v%i/peris_apos/%s/acc/snap_peris_%s_host%i_*.npy' % (stream_version, method, halo, elvis_host))
        f_apos = glob.glob('output/streams_v%i/peris_apos/%s/acc/snap_apos_%s_host%i_*.npy' % (stream_version, method, halo, elvis_host))
    else:
        f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_*.npy' % (stream_version, method, halo))
        f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_*.npy' % (stream_version, method, halo))
    f_peris.sort()
    f_apos.sort()

    pmw = [5.8, 3.8, 17.6, 12.1, 15.7, 12.7, 15.9, 16.0, 9.1, 12.3, 10.7, 13.1, 8.3, 17.]  # from Li21 + Sgr
    amw = [45.8, 13.7, 37.9, 38.2, 35.9, 19.1, 43.9, 58.2, 27.8, 17.9, 39.0, 60.3, 35.2, 80.]

    xlims = {'m12b_res7100': [0, 125], 'm12c_res7100': [0, 125], 'm12f_res7100': [0, 200], 'm12i_res7100': [0, 100], 'm12m_res7100': [0, 140], 'm12r_res7100': [0, 150], 'm12w_res7100': [0, 120]}
    ylims = {'m12b_res7100': [0, 600], 'm12c_res7100': [0, 400], 'm12f_res7100': [0, 450], 'm12i_res7100': [0, 425], 'm12m_res7100': [0, 450], 'm12r_res7100': [0, 300], 'm12w_res7100': [0, 300]}

    plt.figure()
    for i in range(len(f_peris)):
        peris = np.load(f_peris[i])
        apos = np.load(f_apos[i])
        zorder = 1 / len(peris)

        if prog_only:
            if 'elvis' in halo:
                xv_0 = np.load('output/snaps/streams_v%i/%s_host%i_%i_600.npy' % (stream_version, halo, elvis_host, i))
            else:
                xv_0 = np.load('output/snaps/streams_v%i/%s_%i_600.npy' % (stream_version, halo, i))
            prog_idx = get_prog(xv_0[:, :3], q=prog_percentile)
            peris = peris[prog_idx]
            apos = apos[prog_idx]

        cut = (peris > 100) & (np.abs(peris - apos) < 30)
        peris[cut] = np.nan
        apos[cut] = np.nan
        print('testing min: ', np.nanmin(np.abs(peris - apos)))
        print(f_peris[i], np.nanmedian(peris), np.nanmedian(apos))

        if not ebars:
            plt.scatter(peris, apos, s=1, alpha=0.2, zorder=zorder)
        else:
            p_percs = np.nanpercentile(peris, [16, 50, 84])
            a_percs = np.nanpercentile(apos, [16, 50, 84])
            plt.errorbar(p_percs[1], a_percs[1], xerr=np.array([[p_percs[1] - p_percs[0], p_percs[2] - p_percs[1]]]).T, yerr=np.array([[a_percs[1] - a_percs[0], a_percs[2] - a_percs[1]]]).T, fmt='.')

    plt.plot(pmw, amw, 'kx', ms=5)
    plt.xlabel(r'$r_{\rm peri}\ \mathrm{(kpc)}$')
    plt.ylabel(r'$r_{\rm apo}\ \mathrm{(kpc)}$')
    plt.title('%s' % halo)
    try:
        plt.xlim(xlims[halo])
        plt.ylim(ylims[halo])
    except:
        plt.xlim(0, 200)
        plt.ylim(0, 400)

    plt.tight_layout()
    fname = 'snap_peris_apos_%s_v%i_%s' % (halo, stream_version, method)
    if 'elvis' in halo:
        fname = 'snap_peris_apos_%s_host%i_v%i_%s' % (halo, elvis_host, stream_version, method)
    if prog_only:
        fname += '_prog'
    if ebars:
        fname += '_ebars'

    plt.savefig('../plots/peris_apos/%s.png' % fname, bbox_inches='tight')


def match_snaps(halo='m12i_res7100', nmax=600, nmin=0, stream_version=3, star_sel='tcut'):
    print('LOADING IN match_snaps 0')
    snap_0 = read_fire_streams.load_halo(halo, nsnap=600)

    masses_0, inds_0 = read_fire_streams.get_star_indices(halo=halo, elvis_host=1, sorting=True, stream_version=stream_version, star_sel=star_sel, snap=snap_0)
    if 'elvis' in halo:
        masses_02, inds_02 = read_fire_streams.get_star_indices(halo=halo, elvis_host=2, sorting=True, stream_version=stream_version, star_sel=star_sel, snap=snap_0)

    pos_0 = snap_0['star'].prop('host1.distance')
    vel_0 = snap_0['star'].prop('host1.velocity')
    if 'elvis' in halo:
        pos_02 = snap_0['star'].prop('host2.distance')
        vel_02 = snap_0['star'].prop('host2.velocity')
    del snap_0

    # Save snap 600
    for i, inds in enumerate(inds_0):
        if 'elvis' in halo:
            np.save('output/snaps/streams_v%i/%s_host1_%i_%i.npy' % (stream_version, halo, i, 600), np.hstack([pos_0[inds], vel_0[inds]]))
        else:
            np.save('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, 600), np.hstack([pos_0[inds], vel_0[inds]]))

    # If ELVIS, second host
    if 'elvis' in halo:
        for i, inds in enumerate(inds_02):
            np.save('output/snaps/streams_v%i/%s_host2_%i_%i.npy' % (stream_version, halo, i, 600), np.hstack([pos_02[inds], vel_02[inds]]))

    # Save all snaps
    nsnaps = np.arange(nmax, nmin - 1, -1)
    for n in nsnaps:
        if n == 600:
            continue
        elif os.path.exists('output/snaps/streams_v%i/%s_host1_%i_%i.npy' % (stream_version, halo, i, n)) and os.path.exists('output/snaps/streams_v%i/%s_host2_%i_%i.npy' % (stream_version, halo, i, n)):
            continue
        elif 'elvis' not in halo and os.path.exists('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n)):
            continue

        print('LOADING IN match_snaps i')
        snap_i = read_fire_streams.load_halo(halo, nsnap=n, assign_pointers=True)
        pointers = snap_i.Pointer.get_pointers(species_name_from='star', species_names_to='star')
        inds_i = [pointers[inds_0[i]] for i in range(len(inds_0))]
        inds_i = [np.delete(inds_ij, (inds_ij < 0)) for inds_ij in inds_i]
        if 'elvis' in halo:
            inds_i2 = [pointers[inds_02[i]] for i in range(len(inds_02))]
            inds_i2 = [np.delete(inds_ij, (inds_ij < 0)) for inds_ij in inds_i2]

        pos_i = snap_i['star'].prop('host1.distance')
        vel_i = snap_i['star'].prop('host1.velocity')
        if 'elvis' in halo:
            pos_i2 = snap_i['star'].prop('host2.distance')
            vel_i2 = snap_i['star'].prop('host2.velocity')
        del snap_i

        for i, inds in enumerate(inds_i):
            xv_i = np.zeros((len(inds), 6))
            xv_i[(inds > 0), :] = np.hstack([pos_i[inds[inds > 0]], vel_i[inds[inds > 0]]])
            if 'elvis' in halo:
                np.save('output/snaps/streams_v%i/%s_host1_%i_%i.npy' % (stream_version, halo, i, n), xv_i)
            else:
                np.save('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n), xv_i)

        if 'elvis' in halo:
            for i, inds in enumerate(inds_i2):
                xv_i = np.zeros((len(inds), 6))
                xv_i[(inds > 0), :] = np.hstack([pos_i2[inds[inds > 0]], vel_i2[inds[inds > 0]]])
                np.save('output/snaps/streams_v%i/%s_host2_%i_%i.npy' % (stream_version, halo, i, n), xv_i)


def plot_snap_pos(halo='m12i_res7100', elvis_host=1, nmax=600, nmin=500, high_mem=True, rerun=False, stream_version=3, saving=True, star_sel='tcut'):
    masses, inds = read_fire_streams.get_star_indices(halo, elvis_host=elvis_host, sorting=True, stream_version=stream_version, star_sel=star_sel)
    n_streams = len(inds)

    snap_times = np.loadtxt('/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/%s/snapshot_times.txt' % halo)
    snap_times = snap_times[:, 3][::-1]

    nsnaps = np.arange(nmax, nmin - 1, -1)
    for ii, n in enumerate(nsnaps):
        if 'elvis' not in halo and os.path.exists('../plots/snaps/streams_%s_%i.png' % (halo, n)) and not rerun:
            continue
        elif 'elvis' in halo and os.path.exists('../plots/snaps/streams_%s_host%i_%i.png' % (halo, elvis_host, n)) and not rerun:
            continue

        tt = snap_times[ii]

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for axi in ax:
            axi.set_aspect('equal')
            axi.set_xlim(-200, 200)
            axi.set_ylim(-200, 200)
        ax[0].set_xlabel('x (kpc)')
        ax[0].set_ylabel('y (kpc)')
        ax[1].set_xlabel('y (kpc)')
        ax[1].set_ylabel('z (kpc)')
        ax[2].set_xlabel('z (kpc)')
        ax[2].set_ylabel('x (kpc)')

        for i in range(n_streams):
            print()
            print('Snap %i, Stream %i' % (n, i))
            if len(inds[i]) > 1e4 and not high_mem:
                print('Skipping massive stream...rerun with high mem!')
                continue
            # if os.path.exists('../plots/snaps/streams_%s_%i.png' % (halo, n)) and not rerun:
                # continue
            try:
                if 'elvis' in halo:
                    xv_i = np.load('output/snaps/streams_v%i/%s_host%i_%i_%i.npy' % (stream_version, halo, elvis_host, i, n))
                else:
                    xv_i = np.load('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n))
                # print(xv_i.shape)
            except:
                print('Skipping %s, %i' % (halo, i))
                continue

            sel = (xv_i[:, 0] == 0)
            print('Removing %i stars' % sel.sum())
            print()
            ax[0].scatter(xv_i[~sel, 0], xv_i[~sel, 1], s=1, alpha=0.2)
            ax[1].scatter(xv_i[~sel, 1], xv_i[~sel, 2], s=1, alpha=0.2)
            ax[2].scatter(xv_i[~sel, 2], xv_i[~sel, 0], s=1, alpha=0.2)
            ax[2].annotate('%.1f Gyr' % tt, (90, 150))

        plt.tight_layout()
        if saving:
            if 'elvis' in halo:
                plt.savefig('../plots/snaps/streams_%s_host%i_v%i_%i.png' % (halo, elvis_host, stream_version, n), bbox_inches='tight')
            else:
                plt.savefig('../plots/snaps/streams_%s_v%i_%i.png' % (halo, stream_version, n), bbox_inches='tight')
            plt.close()


def get_tmins(halo='m12i_res7100', elvis_host=1, high_mem=True, stream_version=3, star_sel='tcut'):
    try:
        if 'elvis' in halo:
            tmins = np.load('output/streams_v%i/tmins_%s_host%i.npy' % (stream_version, halo, elvis_host))
        else:
            tmins = np.load('output/streams_v%i/tmins_%s.npy' % (stream_version, halo))
        return tmins
    except:
        pass

    # tmaxs = np.full_like(tmins, 14)

    masses, inds = read_fire_streams.get_star_indices(halo, elvis_host=elvis_host, sorting=True, stream_version=stream_version, star_sel='acc')
    n_streams = len(inds)

    snap_times = np.loadtxt('/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/%s/snapshot_times.txt' % halo)
    time_sp = UnivariateSpline(snap_times[:, 0], snap_times[:, 3], s=0)
    snap_times = snap_times[:, 3][::-1]

    tmins = []

    nsnaps = np.arange(600, 0 - 1, -1)
    for i in range(n_streams):
        print()
        print('Stream %i' % i)
        if len(inds[i]) > 1e4 and not high_mem:
            print('Skipping massive stream...rerun with high mem!')
            continue
        r_array = np.zeros((len(nsnaps), len(inds[i])))
        n = 600
        while True:
            n_idx = np.argmin(np.abs(n - nsnaps))
            try:
                if 'elvis' in halo:
                    xv_i = np.load('output/snaps/streams_v%i/acc/%s_host%i_%i_%i.npy' % (stream_version, halo, elvis_host, i, n))
                else:
                    xv_i = np.load('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n))
                # sel = (xv_i[:, 0] == 0)
            except:
                n += 1
                break
            if n < 0:
                n += 1
                break

            r = np.linalg.norm(xv_i[:, :3], axis=1)
            r[r == 0] = np.nan
            r_array[n_idx, :len(r)] = r  # temporary?

            n -= 1

        r_array[r_array == 0] = np.nan

        halo_rads = get_halo_radius(nsnaps, halo=halo)
        r_med = np.nanmedian(r_array, axis=1)
        r_diff = r_med - halo_rads
        n_acc = nsnaps[np.where(r_diff < 0)[0][-1]]
        t_acc = time_sp(n_acc)
        print(t_acc)

        tmins.append(t_acc)

    if 'elvis' in halo:
        np.save('output/streams_v%i/tmins_%s_host%i.npy' % (stream_version, halo, elvis_host), tmins)
    else:
        np.save('output/streams_v%i/tmins_%s.npy' % (stream_version, halo), tmins)

    return tmins


def get_halo_radius(snaps, halo='m12i_res7100'):
    halo_dat = np.load('output/mass_main_' + halo + '.npy')
    halo_rad = halo_dat[:, 3]
    halo_snap = halo_dat[:, 0]
    idx = np.argsort(halo_snap)
    radius_sp = UnivariateSpline(halo_snap[idx], halo_rad[idx], s=0)
    return radius_sp(snaps)


def plot_mass_accretion(halo='m12i_res7100', elvis_host=1, snap=None, stream_version=3, star_sel='tcut', peri_method='min_max'):
    masses, inds = read_fire_streams.get_star_indices(halo, elvis_host=elvis_host, sorting=True, stream_version=stream_version, star_sel=star_sel, snap=snap)
    n_streams = len(inds)

    try:
        tmins = np.load('output/streams_v%i/tmins_%s.npy' % (stream_version, halo))
    except:
        get_tmins(halo=halo, elvis_host=elvis_host, high_mem=high_mem, stream_version=stream_version, star_sel=star_sel)
        tmins = np.load('output/streams_v%i/tmins_%s.npy' % (stream_version, halo))

    f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_*.npy' % (stream_version, peri_method, halo))
    f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_*.npy' % (stream_version, peri_method, halo))
    f_peris.sort()
    f_apos.sort()

    mps = []
    for i in range(len(f_peris)):
        peris = np.load(f_peris[i])
        apos = np.load(f_apos[i])
        cut = (peris > 100) & (np.abs(peris - apos) < 30)
        peris[cut] = np.nan
        mps.append(np.nanmedian(peris))
    mps = np.asarray(mps)

    plt.figure()
    im = plt.scatter(tmins, masses, s=30, c=mps, cmap='magma', marker='^')

    plt.yscale('log')
    plt.xlim(0, 14)
    # plt.ylim(1e5, 1e9)

    plt.xlabel(r'$t_{\rm acc}\ \mathrm{(Gyr)}$')
    plt.ylabel(r'$M_*\ (M_{\rm \odot})$')

    # phase-mixed
    # dat = np.genfromtxt('../data/stars_accretion_history_m12i.csv', delimiter=',', names=True)
    dat = np.genfromtxt('../data/stars_accretion_history_extra_m12i.csv', delimiter=',', names=True)
    sel = dat['subhalo_stellar_mass_stars'] > 1e5
    dat = dat[sel]
    cosmo = ut.cosmology.CosmologyClass()
    t_acc = cosmo.get_time(dat['redshift_stars'], value_kind='redshift')
    stellar_mass = np.unique(dat['subhalo_stellar_mass_stars'])
    pm_mass = []
    pm_tacc = []
    for sm in stellar_mass:
        idx = np.where(dat['subhalo_stellar_mass_stars'] == sm)
        pm_mass.append(sm)
        pm_tacc.append(np.min(t_acc[idx]))

    # plt.scatter(t_acc, dat['subhalo_stellar_mass_stars'], s=1, alpha=0.2, zorder=0)
    plt.scatter(pm_tacc, pm_mass, s=10, zorder=0)

    cb = colorbar(im)
    cb.set_label(r'$r_{\rm peri}\ \mathrm{(kpc)}$')

    plt.savefig('../plots/mass_tacc_%s_v%i.png' % (halo, stream_version))


def integrate_peri(t0, t1, xv0, xv1, halo, nsnap, potential=None, forward_only=True):
    if potential is None:
        potential = load_potential(halo, nsnap=nsnap)

    dt = t1 - t0
    orbit0 = agama.orbit(ic=xv0, potential=potential, time=dt, trajsize=100)
    orbit1 = agama.orbit(ic=xv1, potential=potential, time=-dt, trajsize=100)

    times0 = orbit0[0]
    r0 = np.linalg.norm(orbit0[1][:, :3], axis=1)
    peri0 = np.min(r0)
    tperi0 = t0 + times0[np.argmin(r0)]

    times1 = orbit1[0]
    r1 = np.linalg.norm(orbit1[1][:, :3], axis=1)
    peri1 = np.min(r1)
    tperi1 = t1 + times1[np.argmin(r1)]

    if forward_only:
        return peri0,  tperi0
    else:
        return np.median([peri0, peri1]), np.median([tperi0, tperi1])


def plot_snap_dists(halo='m12i_res7100', elvis_host=1, nmax=600, nmin=350, snap=None, high_mem=True, rerun=False, stream_version=3, star_sel='tcut', integrate=False, method='last', plotting=True):
    masses, inds = read_fire_streams.get_star_indices(halo, elvis_host=elvis_host, sorting=True, stream_version=stream_version, star_sel=star_sel, snap=snap)
    n_streams = len(inds)

    snap_times = np.loadtxt('/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/%s/snapshot_times.txt' % halo)
    time_sp = UnivariateSpline(snap_times[:, 0], snap_times[:, 3], s=0)
    snap_times = snap_times[:, 3][::-1]

    if method not in ['last', 'min_max']:
        print('Method options are last or min_max, using last.')
        method = 'last'

    # Get tmin (accretion time)/tmax
    # try:
    #     if 'elvis' in halo:
    #         tmins = np.load('output/streams_v%i/tmins_%s_host%i.npy' % (stream_version, halo, elvis_host))
    #     else:
    #         tmins = np.load('output/streams_v%i/tmins_%s.npy' % (stream_version, halo))
    # except:
    #     try:
    #         get_tmins(halo=halo, elvis_host=elvis_host, high_mem=high_mem, stream_version=stream_version, star_sel=star_sel)
    #         if 'elvis' in halo:
    #             tmins = np.load('output/streams_v%i/tmins_%s_host%i.npy' % (stream_version, halo, elvis_host))
    #         else:
    #             tmins = np.load('output/streams_v%i/tmins_%s.npy' % (stream_version, halo))
    #     except:
    #         tmins = np.zeros(n_streams)
    tmins = get_tmins(halo=halo, elvis_host=elvis_host, high_mem=high_mem, stream_version=stream_version, star_sel=star_sel)
    tmaxs = np.full_like(tmins, 14)

    # nsnaps = np.arange(nmax, nmin - 1, -1)
    nsnaps = np.arange(600, 0 - 1, -1)
    for i in range(n_streams):
        if os.path.exists('output/streams_v%i/peris_apos/%s/snap_peris_%s_host1_%i.npy' % (stream_version, method, halo, i)) and os.path.exists('output/streams_v%i/peris_apos/%s/snap_peris_%s_host2_%i.npy' % (stream_version, method, halo, i)) and not rerun:
            print('Exists, skipping')
        elif 'elvis' not in halo and os.path.exists('output/streams_v%i/peris_apos/%s/snap_peris_%s_%i.npy' % (stream_version, method, halo, i)) and not rerun:
            print('Exists, skipping')
            continue

        print()
        print('Stream %i' % i)
        if len(inds[i]) > 1e4 and not high_mem:
            print('Skipping massive stream...rerun with high mem!')
            continue

        r_array = np.zeros((len(nsnaps), len(inds[i])))

        n = nmax
        while True:
            n_idx = np.argmin(np.abs(n - nsnaps))
            try:
                if 'elvis' in halo:
                    xv_i = np.load('output/snaps/streams_v%i/%s_host%i_%i_%i.npy' % (stream_version, halo, elvis_host, i, n))
                else:
                    xv_i = np.load('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n))
                # sel = (xv_i[:, 0] == 0)
            except:
                n += 1
                break
            if n < nmin:
                n += 1
                break
            r = np.linalg.norm(xv_i[:, :3], axis=1)
            r[r == 0] = np.nan
            r_array[n_idx, :len(r)] = r

            n -= 1

        r_array[r_array == 0] = np.nan

        # CALCULATE TURNING POINTS
        peris = []
        apos = []
        tperis = []
        tapos = []
        for j in range(r_array.shape[1]):
            rj_diff = r_array[1:, j] - r_array[:-1, j]
            tlim = np.max(snap_times)
            turns = [tmins[i]]
            sign = -1
            while True:
                tcut = (snap_times[:-1] > tmins[i]) & (snap_times[:-1] < tlim)
                widx = np.where((rj_diff[tcut] / np.abs(rj_diff[tcut])) == sign)[0]
                if len(widx) > 0:
                    t = snap_times[:-1][tcut][widx[0]]
                    turns.append(t)
                    sign *= -1
                    tlim = t
                else:
                    break
            turns = np.sort(turns)
            tdiff = np.array(turns[1:]) - np.array(turns[:-1])
            turns = turns[1:][tdiff / np.mean(tdiff) > 0.5]  # drops tacc and any non periodic turns, hopefully
            # turns[0] should be first pericenter

            try:
                # Last
                if method == 'last':
                    t1 = turns[-1]
                    t2 = turns[-2]
                    idx1 = np.argmin(np.abs(t1 - snap_times))
                    idx2 = np.argmin(np.abs(t2 - snap_times))
                    d1 = r_array[np.argmin(np.abs(t1 - snap_times)), j]
                    d2 = r_array[np.argmin(np.abs(t2 - snap_times)), j]
                    if d1 > d2:
                        apo = d1
                        peri = d2
                        t_apo = t1
                        t_peri = t2
                        idx_apo = idx1
                        idx_peri = idx2
                        print('peri, apo = ', peri, apo)
                    elif d2 > d1:
                        apo = d2
                        peri = d1
                        t_apo = t2
                        t_peri = t1
                        idx_apo = idx2
                        idx_peri = idx1
                        print('peri, apo = ', peri, apo)
                    else:
                        print('Problem!')
                elif method == 'min_max':
                    idx_turns = np.asarray([np.argmin(np.abs(ti - snap_times)) for ti in turns])
                    r_turns = r_array[idx_turns, j]
                    peri = np.min(r_turns)
                    apo = np.max(r_turns)
                    t_peri = turns[np.argmin(r_turns)]
                    t_apo = turns[np.argmax(r_turns)]
                    idx_peri = np.argmin(np.abs(peri - r_array[:, j]))
                    idx_apo = np.argmin(np.abs(apo - r_array[:, j]))
                if integrate:
                    n = nsnaps[idx_peri]
                    xv0 = np.load('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n - 1))
                    xv1 = np.load('output/snaps/streams_v%i/%s_%i_%i.npy' % (stream_version, halo, i, n + 1))
                    xv0 = xv0[j, :]
                    xv1 = xv1[j, :]
                    peri, t_peri = integrate_peri(t0=snap_times[idx_peri + 1], t1=snap_times[idx_peri - 1], xv0=xv0, xv1=xv1, nsnap=n, halo=halo)

                tperis.append(t_peri)
                tapos.append(t_apo)
                peris.append(peri)
                apos.append(apo)
            except:
                print('problem, check %i' % j)
                tperis.append(np.nan)
                tapos.append(np.nan)
                peris.append(np.nan)
                apos.append(np.nan)

        peris = np.asarray(peris)
        apos = np.asarray(apos)
        tperis = np.asarray(tperis)
        tapos = np.asarray(tapos)

        fname = '%s_%i' % (halo, i)
        if 'elvis' in halo:
            fname = '%s_host%i_%i' % (halo, elvis_host, i)

        np.save('output/streams_v%i/peris_apos/%s/snap_peris_%s.npy' % (stream_version, method, fname), peris)
        np.save('output/streams_v%i/peris_apos/%s/snap_apos_%s.npy' % (stream_version, method, fname), apos)
        np.save('output/streams_v%i/peris_apos/%s/snap_tperis_%s.npy' % (stream_version, method, fname), tperis)
        np.save('output/streams_v%i/peris_apos/%s/snap_tapos_%s.npy' % (stream_version, method, fname), tapos)
        p_peri = np.nanpercentile(peris, [16, 50, 84])
        p_apo = np.nanpercentile(apos, [16, 50, 84])
        print('r_peri, r_apo = %.2f, %.2f' % (p_peri[1], p_apo[1]))

        if plotting:
            plt.figure()
            idx = np.random.choice(np.arange(len(inds[i])), 10)

            for j in idx:
                plt.plot(snap_times, r_array[:, j], lw=0.5, alpha=0.4, zorder=0)

            plt.plot(tperis[idx], peris[idx], 'k.', ms=3)
            plt.plot(tapos[idx], apos[idx], 'b.', ms=3)

            plt.axhline(p_peri[1], c='C0', lw=2, ls='--', zorder=901)
            plt.axhline(p_apo[1], c='C0', lw=2, ls='--', zorder=901)
            plt.axhspan(p_peri[0], p_peri[2], color='C0', alpha=0.4, zorder=900)
            plt.axhspan(p_apo[0], p_apo[2], color='C0', alpha=0.4, zorder=900)

            tmin = snap_times[np.argmin(np.abs(n - nsnaps))]
            plt.xlim(tmin, np.max(snap_times))
            xmin, xmax, ymin, ymax = plt.axis()
            plt.ylim(0, ymax)
            try:
                halo_rads = get_halo_radius(nsnaps, halo=halo)
                plt.plot(time_sp(nsnaps), halo_rads, c='C3', lw=2, alpha=0.5, zorder=0)
            except:
                pass
            plt.axvline(tmins[i], c='C3', lw=2, alpha=0.5, ls='--', zorder=0)

            plt.xlabel('t (Gyr)')
            plt.ylabel('r (kpc)')
            plt.title('Halo %s, Stream %i' % (halo, i))
            if 'elvis' in halo:
                plt.savefig('../plots/snap_dists/snap_dists_%s_host%i_v%i_%i.png' % (halo, elvis_host, stream_version, i))
            else:
                plt.savefig('../plots/snap_dists/snap_dists_%s_v%i_%i.png' % (halo, stream_version, i))


def get_prog(pos, q=95):
    density = gaussian_kde(pos.T)(pos.T)
    cutoff = np.percentile(density, q)
    return density > cutoff


# def get_prog_orbit(halo='m12i_res7100', elvis_host=1, snap=None, high_mem=True, stream_version=3, star_sel='tcut', integrate=False, method='last', prog_q=95, plotting=True):
#     masses, inds = read_fire_streams.get_star_indices(halo, elvis_host=elvis_host, sorting=True, stream_version=stream_version, star_sel=star_sel, snap=snap)
#     n_streams = len(inds)

#     snap_times = np.loadtxt('/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/%s/snapshot_times.txt' % halo)
#     time_sp = UnivariateSpline(snap_times[:, 0], snap_times[:, 3], s=0)
#     snap_times = snap_times[:, 3][::-1]

#     tmins = get_tmins(halo=halo, elvis_host=elvis_host, high_mem=high_mem, stream_version=stream_version, star_sel=star_sel)
#     tmaxs = np.full_like(tmins, 14)

#     if snap is None:
#         snap = read_fire_streams.load_halo(halo)

#     pos_stars = snap['star'].prop('host%i.distance' % elvis_host)

#     nsnaps = np.arange(600, 0 - 1, -1)

#     for i in range(n_streams):
#         prog_idx = get_prog(pos[inds[i]], q=prog_q)

#     if 'elvis' in h:
#         f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_host%i_*.npy' % (stream_version, method, h, elvis_host))
#         f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_host%i_*.npy' % (stream_version, method, h, elvis_host))
#         f_peris.sort()
#         f_apos.sort()
#     else:
#         f_peris = glob.glob('output/streams_v%i/peris_apos/%s/snap_peris_%s_*.npy' % (stream_version, method, h))
#         f_apos = glob.glob('output/streams_v%i/peris_apos/%s/snap_apos_%s_*.npy' % (stream_version, method, h))
#         f_peris.sort()
#         f_apos.sort()


def plot_E_L():
    pass
