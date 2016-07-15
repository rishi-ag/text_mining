
# coding: utf-8

import functions as fn
import numpy as np
import collections
import pandas as pd
import h5py
np.random.seed(1)


#get parameter dictionary
M = 100
K = 20
V = 600
N = 500
alpha_scalar = 0.5
beta_scalar = 0.5


param_dict = fn.lda_parameters(M, K,V, N, alpha_scalar, beta_scalar)
model = fn.generate_lda(param_dict)


no_iter = 12000
c = 10
M_fin = 5000

seq = 4

theta_post_arr, phi_post_arr, topics_posterior_arr, perp = fn.uncollapsed_gibbs(param_dict, model, seq, no_iter)

print "first save"
#save theta
for m in range(0, M):
    file_name = './u_giibs/theta/' + 'theta_doc_' + str(m+1) + '.h5'
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('dataset_1', data=theta_post_arr[:, m, :])
        
#save phi
for k in range(0, K):
    file_name = './u_giibs/phi/' + 'topic_phi_' + str(k+1) + '.h5'
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('dataset_1', data=phi_post_arr[:, k, :])

#save topics

for m in range(0, M):
    file_name = './u_giibs/topics/' + 'doc_topic' + str(m+1) + '.h5'
    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('dataset_1', data=topics_posterior_arr[:, m, :])

np.savetxt('./u_giibs/u_perp.csv', perp)
print "first save end"



# In[4]:
print "start MCMC diagnostics"
tau_int_topics, tau_int_theta, tau_int_phi, var_topics, var_theta, var_phi, avg_topics, avg_theta, avg_phi = fn.MCMC_analysis(topics_posterior_arr, theta_post_arr, phi_post_arr, param_dict, no_iter, c,  M_fin)
print "finish MCMC diagnostics"

print "saving everything else"
pd.DataFrame(tau_int_topics).to_csv('./u_giibs/tau_int_topics.csv')
pd.DataFrame(tau_int_theta).to_csv('./u_giibs/tau_int_theta.csv')
pd.DataFrame(tau_int_phi).to_csv('./u_giibs/tau_int_phi.csv')
pd.DataFrame(var_topics).to_csv('./u_giibs/var_topics.csv')
pd.DataFrame(var_theta).to_csv('./u_giibs/var_theta.csv')
pd.DataFrame(var_phi).to_csv('./u_giibs/var_phi.csv')
pd.DataFrame(avg_topics).to_csv('./u_giibs/avg_topics.csv')
pd.DataFrame(avg_theta).to_csv('./u_giibs/avg_theta.csv')
pd.DataFrame(avg_phi).to_csv('./u_giibs/avg_phi.csv')
