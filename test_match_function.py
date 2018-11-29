# from astropy.table import Table, join, unique
# from find_streams_plots import *
# from find_streams_analysis import *
# from find_streams_analysis_functions import *
# from velocity_transformations import *

import numpy as np
import matplotlib.pyplot as plt


# from scipy.special import erf
# def prob_e(v1, e1, v2, e2):
#     P = 0.5 * (1. + erf(np.abs(v1 - v2)/np.sqrt(2*(e1**2 + e2**2))))
#     return P
#     # return -1.*np.log10(1.-P)


def prob_e(vr, er, v1, e1, v2, e2):
    en1 = np.sqrt(er**2 + e1**2)
    en2 = np.sqrt(er**2 + e2**2)
    P = -0.5 * ((vr - v1)**2/en1**2 + (vr - v2)**2/en2**2 + np.log(2*np.pi*en1**2) + np.log(2*np.pi*en2**2))
    return P


d = np.linspace(0, 5, 2000)
p = prob_e(1., 2, 1.+d, 0.01, 1.+d, 0.01)
plt.plot(d, p, label='0.01', c='C0')
p = prob_e(1., 2, 1.+d, 0.1, 1.+d, 0.1)
plt.plot(d, p, label='0.1 ', c='C0')
p = prob_e(1., 2, 1.+d, 0.2, 1.+d, 0.2)
plt.plot(d, p, label='2.0 ', c='C0')
p = prob_e(1., 2, 1.+d, 5, 1.+d, 5)
plt.plot(d, p, label='5.0 ', c='C0')

p = prob_e(1., 0.2, 1.+d, 0.01, 1.+d, 0.01)
plt.plot(d, p, label='0.01', c='C1')
p = prob_e(1., 0.2, 1.+d, 0.1, 1.+d, 0.1)
plt.plot(d, p, label='0.1 ', c='C1')
p = prob_e(1., 0.2, 1.+d, 0.2, 1.+d, 0.2)
plt.plot(d, p, label='2.0 ', c='C1')
p = prob_e(1., 0.2, 1.+d, 5, 1.+d, 5)
plt.plot(d, p, label='5.0 ', c='C1')

p = prob_e(1., 0.01, 1.+d, 0.01, 1.+d, 0.01)
plt.plot(d, p, label='0.01', c='C2')
p = prob_e(1., 0.01, 1.+d, 0.1, 1.+d, 0.1)
plt.plot(d, p, label='0.1 ', c='C2')
p = prob_e(1., 0.01, 1.+d, 0.2, 1.+d, 0.2)
plt.plot(d, p, label='2.0 ', c='C2')
p = prob_e(1., 0.01, 1.+d, 5, 1.+d, 5)
plt.plot(d, p, label='5.0 ', c='C2')

plt.legend()
plt.ylim(-102, 20)
plt.show()
plt.close()

raise SystemExit

data_dir = '/data4/cotar/'
tgas_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')
tgas_data = tgas_data[tgas_data['parallax_error']/tgas_data['parallax'] < 0.2]
tgas_data = tgas_data[tgas_data['parallax'] > 0.]

obj_ra = np.deg2rad(tgas_data['ra'])
obj_dec = np.deg2rad(tgas_data['dec'])

GLOBAL_stream = compute_xyz_vel(np.deg2rad(164.), np.deg2rad(13.), 20.)


parallax_pmra_predicted = compute_distance_pmra(obj_ra, obj_dec, tgas_data['pmra'], GLOBAL_stream, parallax=True)
parallax_pmdec_predicted = compute_distance_pmdec(obj_ra, obj_dec, tgas_data['pmdec'], GLOBAL_stream, parallax=True)
rv_stream_predicted = compute_rv(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']), GLOBAL_stream)

obj_parsec = 1e3 / tgas_data['parallax']
pmra_stream_predicted = compute_pmra(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
pmdec_stream_predicted = compute_pmdec(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)

ok4 = match_values_probability(tgas_data['pmra'], tgas_data['pmra_error'], pmra_stream_predicted, log_prob=-8)
ok5 = match_values_probability(tgas_data['pmdec'], tgas_data['pmdec_error'], pmdec_stream_predicted, log_prob=-8)

ok1 = match_values_within_std(tgas_data['parallax'], tgas_data['parallax_error'], parallax_pmra_predicted, std=1.5)
ok2 = match_values_within_std(tgas_data['parallax'], tgas_data['parallax_error'], parallax_pmdec_predicted, std=1.5)

idx_all1 = np.logical_and(ok1, ok2)
print np.sum(idx_all1)

n_MC = 250
tgas_data = tgas_data[idx_all1][:50]
for g_d in tgas_data:
    pmra_MC = MC_values([g_d['pmra']], [g_d['pmra_error']], n_MC)[0]
    pmdec_MC = MC_values([g_d['pmdec']], [g_d['pmdec_error']], n_MC)[0]
    parallax_MC = MC_values([g_d['parallax']], [g_d['parallax_error']], n_MC)[0]

    pmra_stream_predicted = compute_pmra(np.deg2rad(g_d['ra']), np.deg2rad(g_d['dec']), 1e3 / parallax_MC, GLOBAL_stream)
    pmdec_stream_predicted = compute_pmdec(np.deg2rad(g_d['ra']), np.deg2rad(g_d['dec']), 1e3 / parallax_MC, GLOBAL_stream)

    parallax_pmra_predicted = compute_distance_pmra(np.deg2rad(g_d['ra']), np.deg2rad(g_d['dec']), pmra_MC, GLOBAL_stream, parallax=True)
    parallax_pmdec_predicted = compute_distance_pmdec(np.deg2rad(g_d['ra']), np.deg2rad(g_d['dec']), pmdec_MC, GLOBAL_stream, parallax=True)

    p1 = prob_e(g_d['pmra'], g_d['pmra_error'], np.median(pmra_stream_predicted), np.std(pmra_stream_predicted))
    p2 = prob_e(g_d['pmdec'], g_d['pmdec_error'], np.median(pmdec_stream_predicted), np.std(pmdec_stream_predicted))
    p3 = prob_e(g_d['parallax'], g_d['parallax_error'], np.median(parallax_pmra_predicted), np.std(parallax_pmra_predicted))
    p4 = prob_e(g_d['parallax'], g_d['parallax_error'], np.median(parallax_pmdec_predicted), np.std(parallax_pmdec_predicted))

    print 'PMRA: ', g_d['pmra'], g_d['pmra_error'], np.median(pmra_stream_predicted), np.std(pmra_stream_predicted), p1
    print 'PMDEC:', g_d['pmdec'], g_d['pmdec_error'], np.median(pmdec_stream_predicted), np.std(pmdec_stream_predicted), p2
    print 'PLXRA:', g_d['parallax'], g_d['parallax_error'], np.median(parallax_pmra_predicted), np.std(parallax_pmra_predicted), p3
    print 'PLXDE:', g_d['parallax'], g_d['parallax_error'], np.median(parallax_pmdec_predicted), np.std(parallax_pmdec_predicted), p4
    print
    print


raise SystemExit

idx_all2 = np.logical_and(ok4, ok5)
idx_all = np.logical_and(idx_all1, ok3)
idx_all_p = np.logical_and(idx_all2, ok3)
print np.sum(ok1), np.sum(ok2), np.sum(np.logical_and(ok1, ok2)), np.sum(idx_all), np.sum(idx_all_p)

tgas_data['p1'] = parallax_pmra_predicted
tgas_data['p2'] = parallax_pmdec_predicted

print tgas_data['p1', 'p2', 'parallax', 'parallax_error'][idx_all]
print tgas_data['pmra', 'pmdec', 'rv', 'source_id', 'ra', 'dec'][idx_all]

n_MC = 100
# parallax_MC = MC_values(tgas_data['parallax'], tgas_data['parallax_error'], n_MC)
pmra_MC = MC_values(tgas_data['pmra'][idx_all], tgas_data['pmra_error'][idx_all], n_MC)
pmdec_MC = MC_values(tgas_data['pmdec'][idx_all], tgas_data['pmdec_error'][idx_all], n_MC)

pmra_stream_predicted = compute_pmra(obj_ra[idx_all], obj_dec[idx_all], 1e3/tgas_data['parallax'][idx_all], GLOBAL_stream)
pmdec_stream_predicted = compute_pmdec(obj_ra[idx_all], obj_dec[idx_all], 1e3/tgas_data['parallax'][idx_all], GLOBAL_stream)

print pmra_stream_predicted
print pmdec_stream_predicted
print compute_pmra(obj_ra[idx_all], obj_dec[idx_all], 1e3/tgas_data['p1'][idx_all], GLOBAL_stream)
print compute_pmdec(obj_ra[idx_all], obj_dec[idx_all], 1e3/tgas_data['p2'][idx_all], GLOBAL_stream)

# print pmra_MC[0]
# d_pm = compute_distance_pmra(obj_ra[idx_all][0], obj_dec[idx_all][0], pmra_MC[0], GLOBAL_stream, parallax=True)
# print d_pm
# print match_values_within_std(tgas_data['parallax'][idx_all][0], tgas_data['parallax_error'][idx_all][0], d_pm, std=1)
# # print pmdec_MC[0]
