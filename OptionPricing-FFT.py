""" 
System Specifications:
	OS:  macOS Catalina (Version 10.15.7, 64-bit)
	CPU: 2.4 GHz Quad-Core Intel Core i5
	RAM: 16GB 21333 MHz LPDDR3
	Python-Version 3.9.2 64-bit
"""

#%%
import math
import numpy as np
from numpy.fft import fft
from scipy.stats import norm
import matplotlib.pyplot as plt
import cmath            
import timeit
#import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     "figure.figsize" : [6.10356, 3.5], 
#     "pgf.preamble" : r"\usepackage{siunitx} \usepackage[T1]{fontenc} \usepackage[utf8x]{inputenc}"
# }) 
M_crr = 500
M_mc = 100000
S0 = 100.00             
K = 100.00              
T = 1.                  
r = 0.05                
sigma = 0.3             
i = complex(0, 1)       
strikes = np.linspace(S0*0.6, S0*1.4, 50)

np.random.seed(1993)

def BSM_analytical(S0, K, T, r, sigma):
    d_1 = (np.log(S0/K)+(r+sigma**2 /2)*T)/(sigma*np.sqrt(T))
    d_2 = (np.log(S0/K)+(r-sigma**2 /2)*T)/(sigma*np.sqrt(T))
    return S0*norm.cdf(d_1)-K*np.exp(-r*T)*norm.cdf(d_2)

def BSM_CF(v, s, T, r, sigma):
    return np.exp((s+(r-sigma**2 /2)*T)*i*v-sigma**2 *v**2 *T/2)

def Carr_Madan_FFT(S0, K, T, r, sigma):
    k = np.log(K/S0) 
    s = np.log(S0/S0)
    N = 2**12
    lam = 0.00613
    eta = 2*np.pi/(N*lam)
    b = 0.5*N*lam-k
    u = np.arange(1, N+1)
    v_j = eta*(u-1)
    alpha = 3
    v = v_j-(alpha+1)*i
    
    psi = np.exp(-r*T)*(BSM_CF(v, s, T, r, sigma)/(alpha**2 +alpha-v_j**2 +i*(2*alpha+1)*v_j))

    Kronecker = np.zeros(N)
    Kronecker[0] = 1
    j = np.arange(1, N+1)
    Simpson = (3+(-1)**j -Kronecker)/3

    FFT_function = np.exp(i*b*v_j)*psi*eta*Simpson
    C_T = np.exp(-alpha*k)/np.pi*(fft(FFT_function)).real

    return C_T[int((k+b)/lam)]*S0
    
tic = timeit.default_timer()
analytical_values = BSM_analytical(S0, strikes, T, r, sigma)
toc = timeit.default_timer()
time_analytical = toc-tic


tic = timeit.default_timer()
fft_values = np.array([Carr_Madan_FFT(S0, K, T, r, sigma) for K in strikes])
toc = timeit.default_timer()
time_fft = toc-tic

error_fft = fft_values - analytical_values

mse_fft = np.mean(error_fft**2)

plt.clf()
plt.plot(strikes, analytical_values, 'b', label='Analytical', linewidth=2)
plt.plot(strikes, fft_values, 'r.', label='FFT', linewidth=2)
plt.ylabel(r'$C$')
plt.xlabel(r'$K$')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('fft_prices.pgf')
plt.show()

plt.clf()
plt.plot(strikes, error_fft, 'black', label='FFT Error', linewidth=2)
plt.ylabel('Error')
plt.xlabel(r'$K$')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('fft_error.pgf')
plt.show()


def CRR(S0, K, T, r, sigma, M):
    dt = T/M
    beta = (math.exp(-r*dt) + math.exp((r+sigma**2)*dt))/2
    u = beta+math.sqrt(beta**2 -1)
    d = 1/u
    q = (math.exp(r*dt)-d)/(u-d)

    S = np.zeros((M+1, M+1))
    V = np.zeros((M+1, M+1))

    for i in range(1, M+2):
        for j in range(1, i+1):
            S[j-1, i-1] = S0*u**(j-1)*d**(i-j)

    def payoff_call(x):
        return np.maximum(x-K, 0)

    V[:, M] = payoff_call(S[:, M])

    for i in range(M-1, -1, -1):
        for j in range(0, i+1):
            V[j, i] = np.exp(-r*dt)*(q*V[j+1, i+1]+(1-q)*V[j, i+1])

    return V[0, 0]


def MC(S0, K, T, r, sigma, M):
    S = np.empty((M+1))
    V = np.empty((M+1))
    X = np.random.normal(0, 1, M+1)

    def payoff_call(x):
        return np.maximum(x-K, 0)

    for i in range(0, M+1):
        S[i] = S0*math.exp((r-sigma**2/2)*T+sigma*math.sqrt(T)*X[i])
        V[i] = payoff_call(S[i])

    return math.exp(-r*T)*np.mean(V)



tic = timeit.default_timer()
crr_values = np.array([CRR(S0, K, T, r, sigma, M_crr) for K in strikes])
toc = timeit.default_timer()
time_crr = toc-tic

tic = timeit.default_timer()
mc_values = np.array([MC(S0, K, T, r, sigma, M_mc) for K in strikes])
toc = timeit.default_timer()
time_mc = toc-tic

error_crr = crr_values - analytical_values
error_mc = mc_values - analytical_values
mse_crr = np.mean(error_crr**2)
mse_mc = np.mean(error_mc**2)


plt.clf()
plt.plot(strikes, analytical_values, 'b', label='Analytical', linewidth=2)
plt.plot(strikes, crr_values, 'r.', label='CRR', linewidth=0.2)
plt.plot(strikes, mc_values, 'g.', label='MC', linewidth=2)
plt.ylabel(r'$C$')
plt.xlabel(r'$K$')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('other_prices.pgf')
plt.show()

plt.clf()
plt.plot(strikes, error_crr, 'r', label='CRR Error', linewidth=2)
plt.plot(strikes, error_mc, 'g', label='MC Error', linewidth=2)
plt.ylabel('Error')
plt.xlabel(r'$K$')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('other_error.pgf')
plt.show()



# %%
