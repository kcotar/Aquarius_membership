from lmfit import minimize, Parameters, report_fit, Minimizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt, argrelextrema, medfilt, cspline1d, cspline1d_eval, savgol_filter


# peaks_max = 1.*np.array([11,31,41,49,68,83,91,108,133,157,173])
# phase_hist = 1.*np.array([119,108,109,96,93,78,68,65,65,60,76,84,72,55,65,78,68,86,62,89,91,106,140,147,153,215,197,249,342,357,293,475,445,458,268,302,426,352,317,296,356,437,370,330,289,307,332,328,359,410,338,316,312,298,338,362,362,526,409,424,466,629,552,609,848,737,797,979,1038,961,979,886,904,830,805,827,817,844,943,957,1021,1015,984,1096,835,805,774,718,769,899,1044,1359,1321,1204,1336,1262,891,773,917,838,702,701,827,852,733,965,830,959,1117,810,675,712,487,490,372,325,377,366,224,208,140,140,148,124,101,109,101,97,90,90,111,198,321,337,184,95,83,50,46,37,44,35,42,31,42,36,39,41,40,35,34,45,40,41,44,46,45,53,48,45,51,51,49,58,60,59,85,80,91,80,91,110,88,114,79,81,99,102,119,96])
# phase_hist_pos = 1.*np.array([1.,3.,5.,7.,9.,11.,13.,15.,17.,19.,21.,23.,25.,27.,29.,31.,33.,35.,37.,39.,41.,43.,45.,47.,49.,51.,53.,55.,57.,59.,61.,63.,65.,67.,69.,71.,73.,75.,77.,79.,81.,83.,85.,87.,89.,91.,93.,95.,97.,99.,101.,103.,105.,107.,109.,111.,113.,115.,117.,119.,121.,123.,125.,127.,129.,131.,133.,135.,137.,139.,141.,143.,145.,147.,149.,151.,153.,155.,157.,159.,161.,163.,165.,167.,169.,171.,173.,175.,177.,179.,181.,183.,185.,187.,189.,191.,193.,195.,197.,199.,201.,203.,205.,207.,209.,211.,213.,215.,217.,219.,221.,223.,225.,227.,229.,231.,233.,235.,237.,239.,241.,243.,245.,247.,249.,251.,253.,255.,257.,259.,261.,263.,265.,267.,269.,271.,273.,275.,277.,279.,281.,283.,285.,287.,289.,291.,293.,295.,297.,299.,301.,303.,305.,307.,309.,311.,313.,315.,317.,319.,321.,323.,325.,327.,329.,331.,333.,335.,337.,339.,341.,343.,345.,347.,349.,351.,353.,355.,357.,359.])
# baseline_data = 1.*np.array([157.83903119,132.58650305,112.75275134,97.42121723,85.76613225,77.05251824,70.63618743,65.96374237,62.57257598,60.0908715,58.23760254,56.82253304,55.74621732,55.,54.66601609,54.91719093,56.01680376,58.28531811,62.,67.36174835,74.49465849,83.4460222,94.1863277,106.60925962,120.53169907,135.69372354,151.75860699,168.31793987,184.94794956,201.26582096,216.93481644,231.66427591,245.20961679,257.37233401,268.,276.98626473,284.27085567,289.83957296,293.72392181,296.,296.78813023,296.25285531,294.60291365,292.08937516,289.,285.65737456,282.41888688,279.67672668,277.85788537,277.42385205,278.84751263,282.54323912,288.84378876,298.,310.18079256,325.47316738,343.88220666,365.33107381,389.66101352,416.6313517,445.91949549,477.1209333,509.74923475,543.23605073,576.93153137,610.13609502,642.1965712,672.53796956,700.66389788,726.15656212,748.67676636,767.96391282,783.83600187,796.18963205,805.,810.32090054,812.28472663,811.10246936,807.06371797,800.53665986,791.96808056,781.88298347,770.85568898,759.42237128,748.05215748,737.14674731,727.04041314,718.,710.22492553,703.84718004,698.93132644,695.47450031,693.40640985,692.58933591,692.81813197,693.82384265,695.31350704,697.0099621,698.65546111,700.01167373,700.85968592,701.,700.25253464,698.45662484,695.47102194,691.17389363,685.46282394,678.25481323,669.48627822,659.11305196,647.11038384,633.47293959,618.2148013,601.36946739,582.9898526,563.14828804,541.93652117,519.46571575,495.86645192,471.28872614,445.90195122,419.89495632,393.47598693,366.8712259,340.30852475,314.00113494,288.14622888,262.92489996,238.50216254,215.02695194,192.63212445,171.43445732,151.53464877,133.01731799,115.95100513,100.38817132,86.36519865,73.90239016,63.00396988,53.65808279,45.83679486,39.496093,34.57588509,31.,28.67618754,27.49611851,27.33538465,28.05356746,29.49946513,31.52691089,34.,36.79315849,39.79114317,42.88904163,45.99227224,49.01658417,51.88805735,54.54310249,56.9284611,59.00120545,60.7287386,62.0887944,63.06943747,63.6690632,63.8963978,63.77049821,63.3207522,62.58687828,61.61892578,60.47727476,59.23263612,57.9660515,56.76889333,55.74286483,55.,54.66266361,54.86355122,55.74568916,57.46243457,60.17747534])


phase_hist = 1.*np.array([34, 55, 44, 56, 62, 64, 69, 70, 66, 98, 92, 103, 95, 98, 107, 102, 113, 140, 140, 152, 155, 163, 130, 147, 108, 110, 132, 124, 135, 105, 86, 210, 112, 228, 111, 99, 140, 113, 93, 133, 143, 169, 302, 215, 156, 227, 261, 199, 318, 329, 269, 233, 235, 325, 351, 365, 353, 522, 530, 485, 565, 378, 374, 337, 287, 297, 365, 374, 291, 305, 314, 322, 275, 313, 318, 360, 359, 370, 295, 298, 307, 323, 351, 366, 383, 368, 354, 327, 305, 254, 238, 224, 219, 250, 213, 185, 166, 176, 145, 140, 254, 150, 138, 150, 159, 143, 136, 135, 120, 108, 109, 121, 91, 94, 118, 137, 128, 101, 128, 88, 101, 74, 83, 68, 65, 65, 60, 47, 59, 62, 61, 52, 72, 54, 59, 55, 58, 51, 56, 43, 52, 50, 31, 32, 36, 37, 31, 37, 32, 61, 69, 65, 59, 57, 49, 40, 53, 48, 54, 45, 49, 50, 46, 44, 34, 33, 27, 30, 40, 29, 30, 35, 42, 43, 36, 21, 37, 29, 42, 40])
phase_hist_pos = 1.*np.array([1.,3.,5.,7.,9.,11.,13.,15.,17.,19.,21.,23.,25.,27.,29.,31.,33.,35.,37.,39.,41.,43.,45.,47.,49.,51.,53.,55.,57.,59.,61.,63.,65.,67.,69.,71.,73.,75.,77.,79.,81.,83.,85.,87.,89.,91.,93.,95.,97.,99.,101.,103.,105.,107.,109.,111.,113.,115.,117.,119.,121.,123.,125.,127.,129.,131.,133.,135.,137.,139.,141.,143.,145.,147.,149.,151.,153.,155.,157.,159.,161.,163.,165.,167.,169.,171.,173.,175.,177.,179.,181.,183.,185.,187.,189.,191.,193.,195.,197.,199.,201.,203.,205.,207.,209.,211.,213.,215.,217.,219.,221.,223.,225.,227.,229.,231.,233.,235.,237.,239.,241.,243.,245.,247.,249.,251.,253.,255.,257.,259.,261.,263.,265.,267.,269.,271.,273.,275.,277.,279.,281.,283.,285.,287.,289.,291.,293.,295.,297.,299.,301.,303.,305.,307.,309.,311.,313.,315.,317.,319.,321.,323.,325.,327.,329.,331.,333.,335.,337.,339.,341.,343.,345.,347.,349.,351.,353.,355.,357.,359.])


# print len(phase_hist)
# phase_hist = phase_hist[0:100]
# phase_hist_pos = phase_hist_pos[0:100]
# baseline_data = baseline_data[0:100]

# phase_hist -= baseline_data
# baseline_data.fill(0)


def fit_baseline_cheb(x, y, c_deg, n_step=2):
    # initial fit before any point removal from the fit
    chb_coef = np.polynomial.chebyshev.chebfit(x, y, c_deg)
    base = np.polynomial.chebyshev.chebval(x, chb_coef)
    for i_s in range(n_step):
        y_diff = y - base
        y_diff_std = np.nanstd(y_diff)
        # removal of upper outliers
        idx_up_use = np.logical_and(y_diff > 0, np.abs(y_diff) < 0.5 * y_diff_std)
        # removal of outliers below baseline
        idx_low_use = np.logical_and(y_diff < 0, np.abs(y_diff) < 2. * y_diff_std)
        # final idx to be used
        idx_use = np.logical_or(idx_up_use, idx_low_use)
        # refit
        chb_coef = np.polynomial.chebyshev.chebfit(phase_hist_pos[idx_use], phase_hist[idx_use], 15)
        base = np.polynomial.chebyshev.chebval(phase_hist_pos, chb_coef)
    return base


# fit gaussian function(s) to the hist-baseline function
def gaussian_fit(parameters, data, phs, ref_data, evaluate=True):
    n_keys = (len(parameters)) / 3
    function_val = np.array(ref_data)
    for i_k in range(n_keys):
        function_val += parameters['amp' + str(i_k)] * np.exp(-0.5 * (parameters['phs' + str(i_k)] - phs) ** 2. / parameters['std' + str(i_k)] ** 2.)
    if evaluate:
        likelihood = np.power(data - function_val, 2)
        return likelihood
    else:
        return function_val

filtered_curve = savgol_filter(phase_hist, window_length=7, polyorder=5)
baseline = fit_baseline_cheb(phase_hist_pos, phase_hist, 25, n_step=4)
# replace curve with baseline points
idx_replace = (filtered_curve - baseline) < 0
filtered_curve[idx_replace] = baseline[idx_replace]
peaks_max = argrelextrema(filtered_curve, np.greater, order=4, mode='wrap')[0]

residual = filtered_curve - baseline

# remove peaks bellow baseline - there should be no one like this when replacing filtered with baseline for low values
idx_peaks_ok = residual[peaks_max] > 0
peaks_max = peaks_max[idx_peaks_ok]

# remove insignificant peaks or based on their height above the baseline
idx_peaks_ok = baseline[peaks_max]*0.1 < residual[peaks_max]
peaks_max = peaks_max[idx_peaks_ok]

# Parameters class with the values to be fitted to the filtered function
fit_param = Parameters()
fit_keys = list([])
for i_p in range(len(peaks_max)):
    key_std = 'std' + str(i_p)
    fit_param.add(key_std, value=4., min=0.5, max=10., vary=True)
    fit_keys.append(key_std)
    key_amp = 'amp' + str(i_p)
    fit_param.add(key_amp, value=500., min=1., max=2000., vary=True)
    fit_keys.append(key_amp)
    key_wvl = 'phs' + str(i_p)
    peak_loc = phase_hist_pos[peaks_max[i_p]]
    fit_param.add(key_wvl, value=peak_loc, min=peak_loc - 5., max=peak_loc + 5., vary=True)
    fit_keys.append(key_wvl)

# perform the actual fit itself
fit_res = minimize(gaussian_fit, fit_param, args=(filtered_curve, phase_hist_pos, baseline), method='leastsq')
fit_res.params.pretty_print()
report_fit(fit_res)
fitted_curve = gaussian_fit(fit_res.params, 0., phase_hist_pos, baseline, evaluate=False)

# extract phase results
final_phase_peaks = list([])
res_param = fit_res.params
for i_k in range(len(res_param) / 3):
    p_phs = res_param['phs' + str(i_k)].value
    p_std = res_param['std' + str(i_k)].value
    final_phase_peaks.append([p_phs, p_std])

fig, ax = plt.subplots(1, 1)
ax.plot(phase_hist_pos, phase_hist, lw=0.5, color='black')
ax.plot(phase_hist_pos, baseline, lw=1, color='green')
ax.plot(phase_hist_pos, filtered_curve, lw=1, color='blue')
for i_p in peaks_max:
    ax.axvline(x=phase_hist_pos[i_p], lw=0.5, color='green')
ax.plot(phase_hist_pos, fitted_curve, lw=1, color='red')
ax.set(ylim=(0., np.max(phase_hist)*1.05), xlim=(0., 360.))

# plot fitted peak (center and std) to the graph
for i_k in range(len(final_phase_peaks)):
    p_phs = final_phase_peaks[i_k][0]
    p_std = 2.*final_phase_peaks[i_k][1]
    ax.axvline(x=p_phs, lw=1, color='black')
    ax.axvspan(p_phs-p_std, p_phs+p_std, facecolor='black', alpha=0.05)

fig.tight_layout()
plt.show()
plt.close()

