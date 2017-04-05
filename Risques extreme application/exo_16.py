import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
from scipy import stats
from sympy import Eq, Symbol, solve
import statsmodels.api as sm
sns.set_style('whitegrid')
from itertools import repeat
%matplotlib inline

# EXERCICE 1 :

# QUESTION 1
# question a
# determination de la loi de coût associée


data = pd.read_csv("cout_cyclones.csv",sep=";")
vec = data.cout
histo_cout = plt.hist(vec,bins=20)
vec.describe()
#count      55.000000
#mean      689.532727
#std      1311.824256
#min         5.210000
#25%        48.020000
#50%       145.030000
#75%       672.970000
#max      7717.340000

esp     = np.mean(vec)
sigma   = np.std(vec)
var     = sigma **2
med     = np.median(vec)

param_logn_u   = np.log(esp) - 1/2*np.log(1+ ( var/(esp**2) ) )
# param_logn_u = 5.7780514356479884
param_logn_sigma = np.log(1+ ( var/(esp**2) ) )
param_logn_sigma**(1/2)
# param_logn_sigma = 1.51592545136812

k       = Symbol('k')
eqn     = Eq( (esp**2)+ (2*var*k)-(var*k**2) ,0)
k_roots = solve(eqn)
k_par   = k_roots[1]
#k_par  = 2.13199
xm      = Symbol('xm')
eqn2    = Eq( esp - k_par*xm/(k_par-1) )
xm_root = solve(eqn2)[0]
#xm_root= 366.11
#mode = xm_root*((k_par-1)/k_par)**1/k_par

theta_gam = var/esp
k_gam     = esp**2/var
theta_gam
k_gam
#from scipy.special import gamma as Gamma
#
#def f1(x):
#    return Gamma(x)
#eqn3 = Eq( med/(np.log(2)**(1/k))*f1((1+1/k))-esp )
k_wei = float(0.4758754)
lambda_wei = med/(np.log(2))**(1/k_wei)
lambda_wei

vec=np.sort(vec)

# Log(vec) -> Normal
norm = np.array([np.random.normal(loc=esp,scale=sigma) for x in range(len(vec))])
count, bins, _ = plt.hist(norm, 30, normed=True)
sm.qqplot_2samples(norm,np.log(vec),line='r').suptitle('Log(echantillon) -> Normal (KS_dist = 0.7)', fontsize=20)
np.corrcoef(np.sort(norm),np.log(vec))
stats.ks_2samp(np.sort(norm),np.log(vec))
#Ks_2sampResult(statistic=0.69090909090909092, pvalue=1.8946637774700268e-12)

# vec -> LogNormal
lognorm = np.random.lognormal(mean=param_logn_u,sigma=param_logn_sigma**(1/2),size=len(vec))
count,bins,_ = plt.hist(lognorm,30,normed=True)
sm.qqplot_2samples(lognorm,vec,line='r').suptitle('echantillon -> LogNorm (KS_dist = 0.14)', fontsize=20)
np.corrcoef(np.sort(lognorm),vec)
stats.ks_2samp(np.sort(lognorm),vec)
#Ks_2sampResult(statistic=0.145, pvalue=0.00784630338162055)


# vec -> exp
exp = np.random.exponential(scale=esp,size=len(vec))
count,bins,_ = plt.hist(exp,30,normed=True)
sm.qqplot_2samples(exp,vec,line='r').suptitle('echantillon -> expo (KS_dist = 0.35)',fontsize=20)
np.corrcoef(np.sort(exp),vec)
stats.ks_2samp(np.sort(exp),vec)
#Ks_2sampResult(statistic=0.34545454545, pvalue=1.2980779947219388e-25)

# vec -> pareto
par = (np.random.pareto(k_par,len(vec)) +1) * float(xm_root)
count, bins, _ = plt.hist(par, 30, normed=True)
sm.qqplot_2samples(par,vec,line='r').suptitle('echantillon -> pareto (KS_dist = 0.54)',fontsize=20)
np.corrcoef(np.sort(par),vec)
stats.ks_2samp(np.sort(par),vec)
#Ks_2sampResult(statistic=0.54, pvalue=4.7377321421555687e-10)

#vec -> gamma
gamma = np.random.gamma(shape=k_gam,scale=theta_gam,size=len(vec))
count, bins, _ = plt.hist(gamma, 30, normed=True)
sm.qqplot_2samples(gamma,vec,line='r').suptitle('echantillon -> Gamma (KS_dist = 0.16)',fontsize=20)
np.corrcoef(np.sort(gamma),vec)
stats.ks_2samp(np.sort(gamma),vec)
#Ks_2sampResult(statistic=0.16363636363636364, pvalue=0.41922597905560005)

#vec -> Weibull
weib = (np.random.weibull(k_wei,len(vec))+lambda_wei)
count, bins, _ = plt.hist(weib, 30, normed=True)
sm.qqplot_2samples(weib,vec,line='r').suptitle('echantillon -> Weibull (KS_dist = 0.6)',fontsize=20)
np.corrcoef(np.sort(weib),vec)
stats.ks_2samp(np.sort(weib),vec)
#Ks_2sampResult(statistic=0.59999999999999998, pvalue=1.7117781590138689e-09)

#vec -> Normal
norm_simple = np.random.normal(loc=esp,scale=sigma,size=len(vec))
count, bins, _ = plt.hist(norm_simple, 30, normed=True)
sm.qqplot_2samples(norm,(vec),line='r').suptitle('echantillon -> Normal (KS_dist = 0.27)',fontsize=20)
np.corrcoef(np.sort(norm),(vec))
stats.ks_2samp(np.sort(norm),vec)
#Ks_2sampResult(statistic=0.27272727, pvalue=0.026)

# QUESTION 2
# simulation d'un nb de sinistre / année via loi de poisson, puis simulation du cout du sinistre loi gamma (loi qui presente la plus petite distance du test KS)
vec_nb_sin = np.random.poisson(lam=1.7,size=10)
nb_sin_tot = sum(vec_nb_sin)
#vec_couts_sin = np.random.pareto(k_par,nb_sin_tot) * float(xm_root)
vec_couts_sin = gamma = np.random.gamma(shape=k_gam,scale=theta_gam,size=nb_sin_tot)

vec_couts_sin_cum = np.array([])
vec_annees = np.array(list(map(lambda x : 1950+x,range(10))))

def cumulSin(vec,index_deb,index_fin):
    return np.sum(vec[index_deb:index_fin+1])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

index_debut = 0
index_fin = 0
for i in (np.cumsum(vec_nb_sin)-1):
    index_fin = i
    cumul  = cumulSin(vec_couts_sin,index_debut,index_fin)
    vec_couts_sin_cum = np.append(vec_couts_sin_cum,cumul)
    index_debut = index_fin+1

vec_couts_sin_cum_extended = []
vec_annees_extended = []
idx = 0
for i in vec_nb_sin:
    vec_couts_sin_cum_extended += list(repeat(vec_couts_sin_cum[idx],i))
    vec_annees_extended += list(repeat(vec_annees[idx],i))
    idx+=1


df_final                = pd.DataFrame([])
df_final["annee"]       = vec_annees_extended
df_final["cout_unit"]   = vec_couts_sin
df_final["cout cumule"] = vec_couts_sin_cum_extended
print (tabulate(df_final, headers='keys', tablefmt='psql'))
df_final
#       annee    cout_unit  cout cumule
#   0    1950   161.494810   342.579749
#   1    1950   181.084939   342.579749
#   2    1951   420.324314   420.324510
#   3    1951     0.000196   420.324510
#   4    1952   239.153243   954.861653
#   5    1952   588.551284   954.861653
#   6    1952   127.157127   954.861653
#   7    1953  1406.373972  6394.798505
#   8    1953  2027.439976  6394.798505
#   9    1953  2960.984557  6394.798505
#   10   1954     0.400953     0.400953
#   11   1956  5049.271694  5418.772598
#   12   1956   369.500905  5418.772598
#   13   1957   267.692205  5202.877948
#   14   1957  1517.482207  5202.877948
#   15   1957  3417.703537  5202.877948
#   16   1959   371.471077   371.472574
#   17   1959     0.001497   371.472574
########################################## EXERCICE 2
######## question 1
N = 10000
vec_sin_geom = np.random.geometric(0.1,N)
sin_geom_total = np.sum(vec_sin_geom)

vec_cout_X_cum = np.array(list(map(lambda x: np.sum(np.random.exponential(scale=1/0.01,size=x)),vec_sin_geom)))

esp_theo = 1/(0.01*0.1)
sigma_theo = esp_theo

esp_empirique_X   = np.mean(vec_cout_X_cum)
sigma_empirique_X = np.std(vec_cout_X_cum)

err_esp_abs = abs(esp_empirique_X - esp_theo)
err_esp_rel = (esp_empirique_X - esp_theo) / abs(esp_theo)

err_sigma_abs = abs(sigma_empirique_X - sigma_theo)
err_sigma_rel = (sigma_empirique_X - sigma_theo) / abs(sigma_theo)

moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_cout_X_cum)
IC =[str(borne_inf),str(borne_sup)]


recap_array = []

recap_array.append([
    "X ~> expo ",N,err_esp_abs,err_esp_rel,err_sigma_abs,err_sigma_rel,IC
    ])

recap_df = pd.DataFrame(recap_array,columns=["Loi de X","N","err abs esperance","err rel esperance","erreur abs sigma","err rel sigma","confiance"])
print (tabulate(recap_df, headers='keys', tablefmt='psql'))

########## question 2

def getQuantileP(vec_,p):
    if p <=1:
        return float(pd.DataFrame(vec_).quantile(p)[0])

C = getQuantileP(vec_cout_X_cum,0.95)
C
### on prend le max(0 , (Xi - c)  )

vec_stop_loss   = np.array(list(map(lambda x: max(0,x-C),vec_cout_X_cum)))
esp_stop_loss   = np.mean(vec_stop_loss)
sigma_stop_loss = np.std(vec_stop_loss)

esp_stop_loss

moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_stop_loss)
IC = [str(borne_inf),str(borne_sup)]
recap_array = []
recap_array.append([
    esp_stop_loss,sigma_stop_loss,IC
    ])
recap_df = pd.DataFrame(recap_array,columns=["esperance","écart type","confiance"])

print (tabulate(recap_df, headers='keys', tablefmt='psql'))

#   esperance  écart type                       confiance
#  49.246769  294.413546  [55.0181554453, 43.4753821855]
#################################################################


########### question 3 : couts extremes
# on simule n*sinistre, puis on prend le max de cout de sinistre (on prend le max de chaque quantile)
def f_temp(x):
    if x>0:
        return np.max(np.random.exponential(scale=1./0.01,size=x)) # on prend le max du vecteur simulé avec n sinistres
    return 0

vec_cout_extremes = np.array(list(map(lambda x:f_temp(x),vec_sin_geom)))
sigma_cout_extremes = np.std(vec_cout_extremes)
moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_cout_extremes)
IC = [str(borne_inf),str(borne_sup)]
recap_array = []
recap_array.append([
    moyenne,sigma_cout_extremes,IC
    ])
recap_df = pd.DataFrame(recap_array,columns=["esperance","écart type","confiance"])

print (tabulate(recap_df, headers='keys', tablefmt='psql'))

#    esperance  écart type                       confiance
#  255.709692   149.98895  [258.649923828, 252.769459227]
###############################################################################


############ question 4 : cout individuel ~> pareto x0 = 80 & shape = 5


#N = 10000
#vec_sin_geom = np.random.geometric(0.1,N)
#sin_geom_total = np.sum(vec_sin_geom)
x0 = 80
vec_cout_X_cum = np.array(list(map(lambda x: np.sum( ((np.random.pareto(a=5,size=x))+1)*x0 ),vec_sin_geom)))

esp_N   = 1/0.1
esp_Cb  = 5*80/4

var_N   = 0.9/(0.1**2)
var_Cb  = (80**2)*5/((4**2)*3)

esp_theo    = esp_N * esp_Cb
sigma_theo  = (esp_N*var_Cb + var_N*(esp_Cb**2))**(1/2)
esp_empirique_X

esp_empirique_X   = np.mean(vec_cout_X_cum)
sigma_empirique_X = np.std(vec_cout_X_cum)

err_esp_abs = abs(esp_empirique_X - esp_theo)
err_esp_rel = (esp_empirique_X - esp_theo) / abs(esp_theo)

err_sigma_abs = abs(sigma_empirique_X - sigma_theo)
err_sigma_rel = (sigma_empirique_X - sigma_theo) / abs(sigma_theo)

moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_cout_X_cum)
IC =[str(borne_inf),str(borne_sup)]


recap_array = []

recap_array.append([
    "X ~> pareto(a=5,x0=80) ",N,err_esp_abs,err_esp_rel,err_sigma_abs,err_sigma_rel,IC
    ])

recap_df = pd.DataFrame(recap_array,columns=["Loi de X","N","err abs esperance","err rel esperance","erreur abs sigma","err rel sigma","confiance"])
recap_df
print (tabulate(recap_df, headers='keys', tablefmt='psql'))
#      Loi de X              N          err abs esperance  err rel esperance   erreur abs sigma           err rel sigma                      confiance
#  X ~> pareto(a=5,x0=80)   10000            7.67018            0.00767         1.287431                    -0.001352                     [1026.3107252, 989.029634808]
#########################################################################################################################################################################

########## ETAPE 2

def getQuantileP(vec_,p):
    if p <=1:
        return float(pd.DataFrame(vec_).quantile(p)[0])

C = getQuantileP(vec_cout_X_cum,0.95)
C
### on prend le max(0 , (Xi - c)  )

vec_stop_loss   = np.array(list(map(lambda x: max(0,x-C),vec_cout_X_cum)))
esp_stop_loss   = np.mean(vec_stop_loss)
sigma_stop_loss = np.std(vec_stop_loss)

moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_stop_loss)
IC = [str(borne_inf),str(borne_sup)]
recap_array = []
recap_array.append([
    esp_stop_loss,sigma_stop_loss,IC
    ])
recap_df = pd.DataFrame(recap_array,columns=["esperance","écart type","confiance"])
print (tabulate(recap_df, headers='keys', tablefmt='psql'))
#   esperance  écart type                       confiance
#  45.558553  273.201068  [50.9141116824, 40.2029945198]
##############################################################

########### ETAPE 3 : couts extremes
# on simule n*sinistre, puis on prend le max de cout de sinistre (on prend le max de chaque quantile)
def f_temp(x):
    if x>0:
        return np.max(np.random.exponential(scale=1./0.01,size=x))
    return 0

vec_cout_extremes = np.array(list(map(lambda x:f_temp(x),vec_sin_geom)))
sigma_cout_extremes = np.std(vec_cout_extremes)
moyenne,borne_sup,borne_inf = mean_confidence_interval(vec_cout_extremes)
IC = [str(borne_inf),str(borne_sup)]
recap_array = []
recap_array.append([
    moyenne,sigma_cout_extremes,IC
    ])
recap_df = pd.DataFrame(recap_array,columns=["esperance","écart type","confiance"])
print (tabulate(recap_df, headers='keys', tablefmt='psql'))
#   esperance  écart type                       confiance
#  255.37593  150.223166  [258.320754098, 252.431106807]
#####################################################################
