import numpy as np
import collections

def lda_parameters(M, K, V, N, alpha_scalar, beta_scalar, uniform = True):
    """
    returns a dictionary of parameters
    
    INPUTS:
    M - number of documents
    K - no of topics
    V - vocab size
    N - document size
    alpha_scalar - hyperparameter for distribution of topics
    beta_scalar - hyperparameter for distribution of vocab
    
    OUPUT
    dictionary of parameters
    
    DEPENDENCY:
    import numpy as np
    """
    if uniform:
        alpha = np.empty(K); alpha.fill(alpha_scalar)
        beta = np.empty(V); beta.fill(beta_scalar)
    else:
        alpha = np.random.uniform(0.5, 2, size = K)
        beta = np.random.uniform(0.5, 2, size = V)
        
    param_dict = ({"M":M, "K":K, "V":V, "alpha":alpha,
                   "beta":beta, "N":N, "alpha_scalar": alpha_scalar, "beta_scalar": beta_scalar})
    return(param_dict)


def vocab_distribution(beta, K):
    """
    returns a K by V matrix where each row is a dirichlet distributed vector  
    
    DEPENDENCY:
    import numpy as np
    """
    
    phi = np.zeros((K, np.size(beta)))
    for k in range(0,K):
        phi[k,:] = np.random.dirichlet(beta)
    return(phi)


def topic_distribution(alpha, M):
    """
    returns a M by K matrix of topic distribution for every doc m
    
    DEPENDENCY:
    import numpy as np
    """
    theta = np.zeros((M, np.size(alpha)))
        
    for m in range(0, M):
        theta[m,:] = np.random.dirichlet(alpha)
    return(theta)
   


def generate_lda(param_dict):
    
    """
    Returns 2 lists of topics and word assigned to every word of every document 
    
    INPUTS
    param_dict - A dictionary of parameters which is the putpur of the fucnction lda_parameters
    
    OUTPUTS
    a dict with the following keys:
    
    corpus_topics - A list where each element is a numpy array of length (N[m]). 
                    Each element contains an array of topics assigned to every word in each document.

    corpus_words - A list where each element is a numpy array of length (N[m]). 
                    Each element contains an array of word index assigned to every word in each document.
                    
    phi - K by V matix of distributions over vocab for every tpic
    
    theta - M by K matrix of distribution over topics for very document 
    
    DEPENDENCY:
    import numpy as np

    """
    
    #initialise topic distribution
    phi = vocab_distribution(param_dict["beta"], param_dict["K"])
    
    #initialise topic distribution
    theta = topic_distribution(param_dict["alpha"], param_dict["M"])
    
    #initialise a list to store topics and words chosen for every word in every document
    #leach elemnt of the list would be a numpy array
    corpus_topics = np.zeros((param_dict["M"], param_dict["N"]))
    corpus_words = np.zeros((param_dict["M"], param_dict["N"]))
    
    for m in range(0,param_dict["M"]):
        #get topic distribution for every document
        theta_m = theta[m, :]
        
        document_topic = np.zeros(param_dict["N"])
        document_word = np.zeros(param_dict["N"])
        
        for n in range(0, param_dict["N"]):
            #for every word in the document choose a topic from the randomly
            #drawn topic distribution. 
            #z topic number - 1
            #z_one_hot is a K by 1 bit vector
            z_one_hot = np.random.multinomial(1, theta_m)
            z = np.nonzero(z_one_hot)[0][0]
            
            phi_z = phi[z, :]
            
            
            #for every word from the chosen topic sample a word randomly and assign it   
            w_one_hot = np.random.multinomial(1, phi_z)
            w = np.nonzero(w_one_hot)[0][0]
            
            
            document_topic[n] = z + 1
            document_word[n] = w + 1
         
        corpus_topics[m, :] = document_topic
        corpus_words[m, :] = document_word
        
        model = {"topics": corpus_topics.astype(int), "words": corpus_words.astype(int), "phi": phi, "theta": theta}
            
    return(model)

###################################################################################
#functins for both gibbs
###################################################################################
def doc_topic_ass_counter(topics):
    """
    Returns a dict like object of the frequency of topics assigned to the given document  
    
    INPUTS
    topics - array of topics
    
    OUTPUTS
    doc_topic_count - a dict like object of the frequency of topics assigned to the given document

    """
    
    M = topics.shape[0]
    
    doc_topic_count = [0] * M
    
    for m in range(0, M):
        
        #for every document m calculate the number of occurances of every topc k allocation
        cnt = collections.Counter(topics[m,:])
        
        doc_topic_count[m] = cnt
        
    return doc_topic_count


def doc_word_ass_counter(words):
    """
    Returns a dict like object of the frequency of words assigned to the given document  
    
    INPUTS
    words - array of words
    
    OUTPUTS
    doc_word_count - a dict like object of the frequency of words assigned to the given document

    """

    M = words.shape[0]
    
    doc_word_count = [0] * M
    
    for m in range(0, M):
        
        #for every document m calculate the number of occurances of every topc k allocation
        cnt = collections.Counter(words[m,:])
        
        doc_word_count[m] = cnt
        
    return doc_word_count


def topic_ass_counter(topics):
    """
    Returns a dict like object of the frequency of topics assigned to a corpus  
    
    INPUTS
    topics - array of topics
    
    OUTPUTS
    topics_ass - a dict like object of the frequency of topics assigned to a corpus

    """
    

    topics_ass = topics.flatten()
    
    return collections.Counter(topics_ass)


def topic_word_ass_counter(words, topics, K):
    """
    Returns a list of the frequency of words assigned to every topic  
    
    INPUTS
    words - array of words
    topics - array of topics
    K - number of topics
    
    OUTPUTS
    topics_ass - a dict like object of the frequency of topics assigned to a corpus

    """
    
    #create a list of lists of all words assigned to a topic
    M = words.shape[0]
    topics = topics.flatten()
    words = words.flatten()
    topic_words = []
    
    for k in range(0, K):
        # boolean array of presence of topic
        topic_present_k = np.in1d(topics, k +1)
        #array or words associated with the given word 
        topic_words_k = np.multiply(topic_present_k, words)
        #index of nonzero elements in topic_words_k 
        non_zero_ind = np.nonzero(topic_words_k)
        #array of nonzero words 
        non_zero_words = list(topic_words_k[non_zero_ind])
        
        #add list of words of topic k to a list
        topic_words.append(non_zero_words)
        
        #ensure length of topic_words = no of topics
    assert len(topic_words) == K
    
    #create a Counter object for counts of words(vocab) for every topic
    word_topic_cnts = [collections.Counter(word_list) for word_list in topic_words]
    
        
    return word_topic_cnts



def perplexity(theta, phi, doc_word_cnt, M, N, V):
    temp = 0
    mat_prod = np.log(theta.dot(phi))
    
    for m in range(0, M):
        for v in range(0, V):
            temp1 = 0
            temp += doc_word_cnt[m][v + 1] * mat_prod[m, v]
            
    return np.exp(- temp / (M*N))


def pred_distributions(doc_topic_ass_cnt, topic_word_cnt, topic_ass_cnt, param_dict):
    """
    Returns the posterior topics and words probability distribution  
    
    INPUTS
    doc_topic_ass_cnt - count of topics assigned to document 
    topic_word_cnt    - count of all words assigned to every topic
    topic_ass_cnt     - frequency of topics
    
    OUTPUTS
    theta, phi - posterior topics and words probability distribution

    """
    
    
    theta = np.zeros((param_dict["M"], param_dict["K"]))
    
    for m in range(0, param_dict["M"]):
        for k in range(0, param_dict["K"]):
            theta[m, k] = ((doc_topic_ass_cnt[m][k + 1] + param_dict["alpha_scalar"]) /
                           (param_dict["N"] + param_dict["K"] * param_dict["alpha_scalar"]))
    
    phi = np.zeros((param_dict["K"], param_dict["V"]))
    
    
    for k in range(0, param_dict["K"]):
        for v in range(0, param_dict["V"]):
            phi[k, v] = ((topic_word_cnt[k][v +1] + param_dict["beta_scalar"]) 
                         / (topic_ass_cnt[k +1] + param_dict["V"] * param_dict["beta_scalar"]))
            
    return theta , phi

###################################################################################
#functins for uncollapsed gibbs
###################################################################################
#ingredients for gibbs sampling update

def topic_update(words, theta, phi,M, N, K, topic_choice):
    """
    updates topics assigned given theta
    """
    
    topics = np.zeros((M, N))
    k_prob = np.zeros(K)
    
    for m in range(0, M):
        for n in range(0, N):
            for k in range(0, K):
                k_prob[k] = theta[m, k] * phi[k, words[m, n] - 1]
            
            k_prob = k_prob / k_prob.sum()
            topics[m, n] = np.random.choice(topic_choice, 1, p = k_prob)[0]
    topics = topics.astype(int)
                
    return topics



def theta_update(topics, alpha):
    
    """
    updates theta given topics and alpha
    """
    
    theta = np.zeros((topics.shape[0], alpha.size))
    
    
    for m in range(0, topics.shape[0]):
        
        #initialise dirichlet parameters to alpha
        dir_param = np.zeros(alpha.size)
        #for every document m calculate the number of occurances of every topc k allocation
        cnt = collections.Counter(topics[m,:])
        
        #update diricklet param
        for k in range(0, alpha.size):
            dir_param[k] = alpha[k] + cnt[k + 1]
        
        #dirichlt draw from updated parameters 
        theta[m, :] = np.random.dirichlet(dir_param)
        
    return theta


def phi_update(words, topics, beta, K, V):
    """
    updates phi given other random variables
    """
    
    phi = np.zeros((K, V))
    word_topic_cnt = topic_word_ass_counter(words, topics, K)
    
    #update phi
    for k, word_cnts in enumerate(word_topic_cnt):
        phi_param = np.zeros(beta.size)
    
        for v in range(0, V):
            phi_param[v] = beta[v] + word_cnts[v + 1]
        
        #draw from dirichlet with updated parameters
        #print phi_param, V, len(phi_param)
        phi[k, :] = np.random.dirichlet(phi_param)
        
    return phi

###################################################################################
#Uncollapsed gibbs sampler
###################################################################################

def gibbs_seq(words, topics, theta_posterior, phi_posterior, param_dict, topic_choice, seq):
    
    if seq == 1:
        topics           = topic_update(words, theta_posterior, phi_posterior, param_dict["M"], param_dict["N"], param_dict["K"], topic_choice)
        phi_posterior    = phi_update(words, topics, param_dict["beta"], param_dict["K"], param_dict["V"])
        theta_posterior  = theta_update(topics, param_dict["alpha"])
        
    elif seq == 2:
        phi_posterior    = phi_update(words, topics, param_dict["beta"], param_dict["K"], param_dict["V"])
        theta_posterior  = theta_update(topics, param_dict["alpha"])
        topics           = topic_update(words, theta_posterior, phi_posterior, param_dict["M"], param_dict["N"], param_dict["K"], topic_choice)
        
    elif seq == 3:
        phi_posterior    = phi_update(words, topics, param_dict["beta"], param_dict["K"], param_dict["V"])
        topics           = topic_update(words, theta_posterior, phi_posterior, param_dict["M"], param_dict["N"], param_dict["K"], topic_choice)
        theta_posterior  = theta_update(topics, param_dict["alpha"])

    elif seq == 4:
        theta_posterior  = theta_update(topics, param_dict["alpha"])
        topics           = topic_update(words, theta_posterior, phi_posterior, param_dict["M"], param_dict["N"], param_dict["K"], topic_choice)
        phi_posterior    = phi_update(words, topics, param_dict["beta"], param_dict["K"], param_dict["V"])

    return topics, theta_posterior, phi_posterior

def uncollapsed_gibbs(param_dict, model, seq = 1, no_iter = 8000):
    """
    Ret
    """
    #starting values of randon variables. Keep topics and words the same as generated from model
    #but change the thetas and betas
    phi_posterior = model["phi"]
    theta_posterior = model["theta"]
    topics = model["topics"]
    words = model["words"]
    
    #for calculating perplexity
    doc_word_cnt = doc_word_ass_counter(words)
    
    theta_posterior_array    = np.zeros((no_iter + 1, param_dict["M"] , param_dict["K"]))
    phi_posterior_array      = np.zeros((no_iter + 1, param_dict["K"] , param_dict["V"]))
    topics_posterior_array   = np.zeros((no_iter + 1, param_dict["M"] , param_dict["N"]))
    
    theta_posterior_array[0, :, :] = theta_posterior
    phi_posterior_array[0, :, :] = phi_posterior
    topics_posterior_array[0, :, :] = model["topics"]
    #set up list for perplexity
    perp_list = np.zeros(no_iter)
    
    #count for samples collected
    topic_choice         = range(1, (param_dict["K"] + 1))
    
    for i in range(1, no_iter + 1):
        topics           = topic_update(words, theta_posterior, phi_posterior, param_dict["M"], param_dict["N"], param_dict["K"], topic_choice)
        phi_posterior    = phi_update(words, topics, param_dict["beta"], param_dict["K"], param_dict["V"])
        theta_posterior  = theta_update(topics, param_dict["alpha"])

        theta_posterior_array[i, :, :]  = theta_posterior
        phi_posterior_array[i, :, :]    = phi_posterior
        topics_posterior_array[i, :, :] = topics
    
        print i
        
    return theta_posterior_array, phi_posterior_array, topics_posterior_array

###################################################################################
#collapsed gibbs sampler
###################################################################################

def collapsed_gibbs(param_dict, model,  no_iter = 8000):
    
    
    theta_posterior_array    = np.zeros((no_iter + 1, param_dict["M"] , param_dict["K"]))
    phi_posterior_array      = np.zeros((no_iter + 1, param_dict["K"] , param_dict["V"]))
    topics_posterior_array   = np.zeros((no_iter + 1, param_dict["M"] , param_dict["N"]))
    perp                     = np.zeros(no_iter)
    
    perp                  = np.zeros(no_iter)
                                                                                                                                                                                                                                                                                        
    topic_posterior       = model["topics"]
    words                 = model["words"]
    
    theta_posterior_array[0, :, :]  = model["theta"]
    phi_posterior_array[0, :, :]    = model["phi"]
    topics_posterior_array[0, :, :] = model["topics"]
    

    topic_choice          = range(1, (param_dict["K"] + 1))
    
    #initialise counts
    doc_topic_ass_cnt     = doc_topic_ass_counter(topic_posterior)
    topic_ass_cnt         = topic_ass_counter(topic_posterior)
    topic_word_cnt        = topic_word_ass_counter(words, topic_posterior, param_dict["K"])  
    doc_word_cnt          = doc_word_ass_counter(words)
    
    prob_k                = np.zeros(param_dict["K"])
                
    
    for i in range(0, no_iter):
        for m in range(0, param_dict["M"]):
            for n in range(0, param_dict["N"]):
                word = words[m, n]
                topic = topic_posterior[m, n]

                #update counts
                doc_topic_ass_cnt[m][topic] -= 1
                topic_word_cnt[topic - 1][word] -= 1
                topic_ass_cnt[topic] -= 1

                
                for k in range(0, param_dict["K"]):
                    left  = doc_topic_ass_cnt[m][k + 1] + param_dict["alpha_scalar"]
                    right = ((topic_word_cnt[k][word] + param_dict["beta_scalar"]) / 
                             (topic_ass_cnt[k+1] + param_dict["V"] * param_dict["beta_scalar"]))
                    prob_k[k] = left * right
                
                #renormalise probs
                prob_k = prob_k/ prob_k.sum()
                #assign topic
                topic = np.random.choice(topic_choice,1, p = prob_k)[0]
                topic_posterior[m, n] = topic
                                
                #change counts
                doc_topic_ass_cnt[m][topic] += 1
                topic_word_cnt[topic - 1][word] += 1
                topic_ass_cnt[topic] += 1

        #predictive phis and predictive beta's
        topics_posterior_array[i, : , :] = topic_posterior
        theta_posterior, phi_posterior = pred_distributions(doc_topic_ass_cnt, topic_word_cnt, param_dict)
        theta_posterior_array[i, : , :] = theta_posterior
        phi_posterior_array[i, :, :] = phi_posterior
        print i
    return topics_posterior_array.astype(int), theta_posterior_array, phi_posterior_array


###################################################################################
#gibbs sampler diagnostics
###################################################################################

def integrated_autocorrelation(x, c, no_iter, M_fin):
    """
    Returns integrated autocorrelation given a time series  
    
    INPUTS
    x      - time series of function
    c      - factor that determines bias variance tradeoff
    no_iter- number of iterations
    M_fin  - window factor to get reasonable error estimates
    
    OUTPUTS
    topics_ass - a dict like object of the frequency of topics assigned to a corpus

    """

    n = len(x)
    samp_mean = x.mean()
    samp_var = x.var()
    x = x - samp_mean

    acf = np.zeros(M_fin + 1)
    
    for lag in range(0, M_fin + 1):
        acf[lag] = np.dot(x[0:(n - lag )], x[lag :] )/(samp_var * n) 

    acf_sum = -0.5
    for m in range(0, M_fin):
        acf_sum += acf[m]

        if m >= int( c * acf_sum):
                break
    return acf_sum

def function_variance(ts, tau_int, n):
    return (2 * tau_int * ts.var()) / n              
    
def MCMC_analysis(topics_ts, theta_ts, phi_ts, param_dict, no_iter, c, M_fin):
    # function calculates all different metrics for gibbs sampling time series like mean, variance, integrated autocorrelations, exponential
    #autocorrelations etc.

    M = param_dict["M"]
    N = param_dict["N"]
    K = param_dict["K"]
    V = param_dict["V"]
    
    #INTEGRATED AUTO
    tau_int_topics = np.zeros([M, N])
    tau_int_theta  = np.zeros([M, K])
    tau_int_phi    = np.zeros([K, V])

    #VARIANCE
    var_topics = np.zeros([M, N])
    var_theta  = np.zeros([M, K])
    var_phi    = np.zeros([K, V])

    #average of the variables
    avg_topics = np.zeros([M, N])
    avg_theta  = np.zeros([M, K])
    avg_phi    = np.zeros([K, V])
    
        
    #calculate integrated autocorrelation
    for m in range(0, M):
        for n in range(0, N):
            tau_int_topics[m, n] = integrated_autocorrelation(topics_ts[1:, m, n], c, no_iter, M_fin)


    for m in range(0, M):
        for k in range(0, K):
            tau_int_theta[m, k]  = integrated_autocorrelation(theta_ts[1:, m, k], c, no_iter, M_fin)

    for k in range(0, K):
        for v in range(0, V):
            tau_int_phi[k, v]    = integrated_autocorrelation(phi_ts[1:, k, v], c, no_iter, M_fin)
            
            
    #calculate mean variance of functions

    for m in range(0, M):
        for n in range(0, N):
            burn_in = int(tau_int_topics[m, n] * 20)
            var_topics[m, n] = (function_variance(topics_ts[(burn_in + 1) :, m, n], 
                                                  tau_int_topics[m, n],
                                                  no_iter - burn_in -1))
            avg_topics[m, n] = topics_ts[(burn_in + 1) :, m, n].mean()

    for m in range(0, M):
        for k in range(0, K):
            burn_in = int(tau_int_theta[m, k] * 20)
            var_theta[m, k] = (function_variance(theta_ts[(burn_in + 1) :, m, k],
                                                 tau_int_theta[m, k],
                                                 no_iter - burn_in -1))
            avg_theta[m, k] = theta_ts[(burn_in + 1 ) :, m, k].mean()

    for k in range(0, K):
        for v in range(0, V):
            burn_in = int(tau_int_phi[k, v] * 20)
            var_phi[k, v] = (function_variance(phi_ts[(burn_in + 1) :, k, v],
                                               tau_int_phi[k, v],
                                               no_iter - burn_in -1))
            avg_phi[k, v] = phi_ts[(burn_in + 1 ) :, k, v].mean()

    return tau_int_topics, tau_int_theta, tau_int_phi, var_topics, var_theta, var_phi, avg_topics, avg_theta, avg_phi

