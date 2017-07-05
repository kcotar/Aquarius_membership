import numpy as np

from scipy.stats import norm
from find_streams_analysis_functions import match_values_within_std

# mc distribution
imaginary_data = np.array([1,2,2,3,3,4,4,4,4,5,5,5,5,5,6,6,6,7,8,8,9,10])
# test distributions
mean_d1 = 5
std_d1 = .1
mean_d2 = 10
std_d2 = 5

# results old - VERY WRONG
print np.sum(match_values_within_std(mean_d1, std_d1, imaginary_data))
print np.sum(match_values_within_std(mean_d2, std_d2, imaginary_data))

# results new matching functions
print np.sum(norm.pdf(imaginary_data, loc=mean_d1, scale=std_d1))
print np.sum(norm.pdf(imaginary_data, loc=mean_d2, scale=std_d2))
