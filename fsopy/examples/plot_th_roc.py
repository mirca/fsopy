from fsopy.receiver_operating_characteristic import th_roc_glq, th_roc_num
from matplotlib import pyplot as plt

# ook modulation
mod_order = 2

# signal to noise ratio in dB
snr_db = 10

# number of transmitted symbols
n_samples = 20

# number of points to make the ROC
n_thresh = 1000

# number of terms for the GL quadrature
n_terms = 90

# fading type
fading = 'gamma_gamma'

# fading parameters
alpha = 1
eta = 1
beta = 1

Pf, Pm1 = th_roc_glq(mod_order, snr_db, n_samples, n_thresh, n_terms, fading,
                     beta, alpha)

Pf, Pm2 = th_roc_num(mod_order, snr_db, n_samples, n_thresh, fading, beta, alpha)

plt.figure()
plt.loglog(Pf, Pm1)
plt.figure()
plt.semilogx(Pf, Pm1-Pm2)
plt.show()

fading = 'exp_weibull'

Pf, Pm1 = th_roc_glq(mod_order, snr_db, n_samples, n_thresh, n_terms, fading,
                     beta, alpha, eta)

Pf, Pm2 = th_roc_num(mod_order, snr_db, n_samples, n_thresh, fading, beta,
                     alpha, eta)

plt.figure()
plt.loglog(Pf, Pm1)
plt.figure()
plt.semilogx(Pf, Pm1-Pm2)
plt.show()
