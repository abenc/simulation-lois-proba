import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
from scipy import stats
import statsmodels.api as sm
from tabulate import tabulate
from math import trunc
sns.set_style('whitegrid')
%matplotlib inline

data_c = pd.read_csv("fic_exo_9_3_17c.csv",sep=";")
data_d = pd.read_csv("fic_exo_9_3_17d.csv",sep=";")

data_c = data_c.drop("V10",axis=1)
data_d = data_d.drop("V10",axis=1)

vec_c_init = pd.melt(data_c).value
vec_d_init = pd.melt(data_d).value

vec_c = np.sort(vec_c_init)
vec_d = np.sort(vec_d_init)



# question A
histo_fichier_c = plt.hist(vec_c,bins=60)
histo_fichier_d = plt.hist(vec_d,bins=60)
histo_fichier_d = plt.hist(vec_d[0:19200],bins=60)
# question B
        # moyenne
        # ecart type
        # min
        # max
        # etendue quartile
dataset_description = pd.DataFrame(vec_d_init.describe()).T.append(pd.DataFrame(vec_c_init.describe()).T)
dataset_description["echantillon"] = ["d","c"]
print (tabulate(dataset_description, headers='keys', tablefmt='psql'))

# question C

# Courbe de concentration
def CourbeConcentration(vec):
    vec_x = [x/(len(vec)+1) for x in range(len(vec))]
    vec_y = np.cumsum(vec)/sum(vec)
    return vec_x,vec_y
# courbe echantillon 1

x1,y1 = CourbeConcentration(vec_c)
sns.plt.scatter(x1,y1)

# courbe echantillon 2
x2,y2 = CourbeConcentration(vec_d)
sns.plt.scatter(x2,y2)

# indice de Gini
# difference de l'aire inferieur du rectangle, et la fonction de repartition de l'echantillon
def gini(vec):
    n       = len(vec)
    vec     = np.sort(vec)
    num     = np.sum(vec * [x for x in range(1,n+1)])
    denom   = n * np.sum(vec)
    Gini    = (2*num/denom)-((n+1)/n)
    return Gini

    # indice de Gini echantillon C = 0.112998772434
print(gini(vec_c))

    # indice de Gini echantillon D = 0.509403954113
print(gini(vec_d))

# question D
# diagramme de probabilités de retour
##### ECHANTILLON C
k                   = 200
vec_c_quantiles     = np.split(vec_c_init,k)
max_par_quantile    = np.array(list(map(lambda x : max(x), vec_c_quantiles)))
u                   = np.array(list(range(1,k)))/(k+1)
Yu                  = np.array(list(map(lambda x: -1* np.log(1-x) , u)))

echantillon_loi_de_proba = np.array(list(map(lambda x: np.percentile(max_par_quantile,x),u*100)))
plt.scatter(np.log(Yu),echantillon_loi_de_proba)
########

####### ECHANTILLON D
k                   = 200
vec_d_quantiles     = np.split(vec_d_init,k)
max_par_quantile    = np.array(list(map(lambda x : max(x), vec_d_quantiles)))
u                   = np.array(list(range(1,k)))/(k+1)
Yu                  = np.array(list(map(lambda x: -1* np.log(1-x) , u)))

echantillon_loi_de_proba = np.array(list(map(lambda x: np.percentile(max_par_quantile,x),u*100)))
plt.scatter(np.log(Yu),echantillon_loi_de_proba)


########
# nous donne une idée si la distribution suit du weibull, du frechet ou alors du gumbell
# question E
# estimation de "eta" de la loi gumbel
def pickandVec(vec):
    R       = np.array([])
    cst     = 1/np.log(2)
    n       = len(vec)
    indice  = n-1
    for i in range(1,int(n/4)+1):
        numerateur      = vec[indice-i+1] - vec[indice-2*i+1]
        denominateur    = vec[indice-2*i+1] - vec[indice-4*i+1]
        #print( vec_c[indice-i+1] )
        #print(vec_c[indice-2*i+1])
        #print(np.log(numerateur/denominateur))
        R = np.append(R,np.log(numerateur/denominateur)*cst)
    return R

R_ech_c = pickandVec(vec_c)
R_ech_d = pickandVec(vec_d)


plt.plot(R_ech_c)
plt.plot(R_ech_d)
eta1 = np.median(R_ech_c[2000:len(R_ech_c)])
eta2 = np.median(R_ech_d[1000:len(R_ech_d)])
#eta1 = -0.969
#eta2 = 0.709
################################################### QUESTION 2 ##############################################################

# question A
esp_c   = np.mean(vec_c)
esp_d   = np.mean(vec_d)
sigma_c = np.std(vec_c)
sigma_d = np.std(vec_d)
var_c   = sigma_c**2
var_d   = sigma_d**2

### ECHANTILLON C
param_loi_expo = 1/esp_c

param_logn_u        = np.log(esp_c) - 1/2*np.log(1+ ( var_c/(esp_c**2) ) )
# param_logn_u = 6.3706477793788316
param_logn_sigma    = np.log(1+ ( var_c/(esp_c**2) ) )
# param_logn_sigma**1/2 = 0.018
param_logn_sigma
##pareto solving

from sympy import Eq, Symbol, solve

k       = Symbol('k')
eqn     = Eq( (esp_c**2)+ (2*var_c*k)-(var_c*k**2) ,0)
k_root  = solve(eqn)[1]
xm      = Symbol('xm')
eqn2    = Eq( esp_c - k_root*xm/(k_root-1) )
xm_root = solve(eqn2)[0]
# xm_root = 499.55021
# mode = xm_root*((k_root-1)/k_root)**1/k_root

exp_c          = np.random.exponential(scale=esp_c,size=len(vec_c))
count, bins, _ = plt.hist(exp_c, 200, normed=True)
sm.qqplot_2samples(exp_c,vec_c,line='r').suptitle('echantillon C ~> expo (KS_dist = 0.49)', fontsize=20)
stats.ks_2samp(np.sort(exp_c),vec_c)
#Ks_2sampResult(statistic=0.49, pvalue=0.0

norm_c = np.random.normal(loc=esp_c,scale=sigma_c,size=len(vec_c))
count, bins, _ = plt.hist(norm_c, 200, normed=True)
sm.qqplot_2samples(norm_c,np.log(vec_c),line='r').suptitle('echantillon C ~> normal (KS_dist = 0.07)', fontsize=20)
stats.ks_2samp(np.sort(norm_c),vec_c)
#Ks_2sampResult(statistic=0.068199999999999927, pvalue=6.3527863052483843e-41)

par_c = (np.random.pareto(k_root,len(vec_c))+1) * float(xm_root)
count, bins, _ = plt.hist(par_c, 200, normed=True)
sm.qqplot_2samples(par_c,vec_c,line='r').suptitle('echantillon C ~> pareto (KS_dist = 0.27)', fontsize=20)
stats.ks_2samp(np.sort(par_c),vec_c)
#Ks_2sampResult(statistic=0.26915, pvalue=0.0)

logn_c = np.random.lognormal(mean=param_logn_u,sigma=param_logn_sigma**(1/2),size=len(vec_c))
count, bins, _ = plt.hist(logn_c, 200, normed=True)
sm.qqplot_2samples(logn_c,vec_c,line='r').suptitle('echantillon C ~> logn (KS_dist = 0.386)', fontsize=20)
stats.ks_2samp(np.sort(logn_c),vec_c)
#Ks_2sampResult(statistic=0.077200000000000046, pvalue=0.0)


#np.corrcoef(np.sort(logn_c),vec_c)
#np.corrcoef(np.sort(par_c),vec_c)
#np.corrcoef(np.sort(exp_c),vec_c)
#np.corrcoef(np.sort(norm_c),vec_c)
#estimation de alpha pareto .. n / sum(np.log(vec_c/np.min(vec_c)))

### ECHANTILLON D
param_loi_expo   = 1/esp_d
param_logn_u     = np.log(esp_d) - 1/2*np.log(1+ ( var_d/(esp_d**2) ) )
# param_logn_u = 4.4980912003571554
param_logn_sigma = np.log(1+ ( var_d/(esp_d**2) ) )
# param_logn_sigma**1/2 = 1.635
##pareto solving



k       = Symbol('k')
eqn     = Eq( (esp_d**2)+ (2*var_d*k)-(var_d*k**2) ,0)
k_root  = solve(eqn)[1]

xm      = Symbol('xm')
eqn2    = Eq( esp_d - k_root*xm/(k_root-1) )
xm_root = solve(eqn2)[0]
# xm_root = 232.819758724083

mode = xm_root*((k_root-1)/k_root)**1/k_root

exp_d = np.array([np.random.exponential(scale=esp_d) for x in range(len(vec_d))])
count, bins, _ = plt.hist(exp_d, 200, normed=True)
sm.qqplot_2samples(exp_d,vec_d,line='r').suptitle('echantillon D ~> expo (KS_dist = 0.276)', fontsize=20)
stats.ks_2samp(np.sort(exp_d),vec_d)
#Ks_2sampResult(statistic=0.276, pvalue=0.0

norm_d = np.random.normal(loc=esp_d,scale=sigma_d,size=(len(vec_d)))
count, bins, _ = plt.hist(norm_d, 200, normed=True)
sm.qqplot_2samples(norm_d,np.log(vec_d),line='r').suptitle('echantillon D ~> normal (KS_dist = 0.44)', fontsize=20)
stats.ks_2samp(np.sort(norm_d),vec_d)
#Ks_2sampResult(statistic=0.44124999999999998, pvalue=0.0)

par_d = (np.random.pareto(k_root,len(vec_d))+1) * float(xm_root)
count, bins, _ = plt.hist(par_d, 200, normed=True)
sm.qqplot_2samples(par_d,vec_d,line='r').suptitle('echantillon D ~> pareto (KS_dist = 0.48)', fontsize=20)
stats.ks_2samp(np.sort(par_d),vec_d)
#Ks_2sampResult(statistic=0.47994999999999999, pvalue=0.0)

logn_d = np.random.lognormal(mean=param_logn_u,sigma=param_logn_sigma,size=len(vec_d))
count, bins, _ = plt.hist(logn_d, 200, normed=True)
sm.qqplot_2samples(logn_d,vec_d,line='r').suptitle('echantillon D ~> lognormal (KS_dist = 0.56)', fontsize=20)
stats.ks_2samp(np.sort(logn_d),vec_d)
#Ks_2sampResult(statistic=0.56030000000000002, pvalue=0.0)



#np.corrcoef(np.sort(logn_d),vec_d)
#np.corrcoef(np.sort(par_d),vec_d)
#np.corrcoef(np.sort(exp_d),vec_d)
#np.corrcoef(np.sort(norm_d),vec_d)
