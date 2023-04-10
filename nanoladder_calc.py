import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants

#################################################################################################
# Global functions and function to generate the data (obtained from the javascript of the experiment website)






# intrinsic experimental parameters [hidden from operator]
# resonator properties
m = 5.5e-15; # resonator mass in units of kg

f0 = 5.5e3; # resonance frequency in units of Hz
f0_sqr = f0**2

w0 = 2 * np.pi * f0; # angular resonance frequency in units of rad / s

Q = 4.5e4; # mechanical quality factor, no units
# other parameters
f0_const = f0_sqr/Q**2

kB = 1.38e-23; # Boltzmann constant in units of J / K

h = 6.63e-34; # Plack constant in units of Js

G = 1e7; # conversion gain of the interferometer in units of V / m
G_sqr = G**2

s_n = 1e-6; # voltage noise PSD in units of V ^ 2 / Hz

tau = Q / (np.pi * f0); # initialize PSD arrays
inv_tau = np.pi * f0 / Q
chi_sqr_const = 1/(16*m*m*(np.pi**4))

#bookkeeper
#those variables are used as storage to prevent calculating the freq array and S_f_mult array
#if sampling rate and acquisition time doesnt change between function calls
#probably only minimal improvement
S_last = -1
t_meas_last = -1
freq_last =np.array(1)
S_f_mult_last = np.array(1)

def gendata(temperature, samplingRate, acquisitionTime, noise) :
    global S_last
    global t_meas_last
    global freq_last
    global S_f_mult_last

    # experimental parameters [visible to operator]
    T = temperature; # temperature in units of K
  
    S = samplingRate; # sampling rate of ADC in units of Hz
  
    t_meas = acquisitionTime; # data acquisition time in units of s
  
    meas_noise = noise; #measurement noiese on or off

    s_f = 4 * kB * T * m * w0 / Q; # mean value of thermomechanical noise PSD(single - sided) in units of N ^ 2 / Hz
  
    df = 1 / t_meas; # frequency resolution
  
    N = t_meas * S / 2; # number of point in PSD arrays
    N_exact = np.floor(N).astype(int) + 1
    n = t_meas * inv_tau; # number of uncorrelated variance estimations
    freq = freq_last
    S_f_mult = S_f_mult_last
    if S_last !=S or t_meas_last != t_meas :
        freq = df*(1+np.arange(N_exact))
        S_f_mult = chi_sqr_const/ ((f0_sqr - freq**2)**2 + f0_const * freq**2) *G_sqr

    S_v = np.abs(np.random.normal(loc = s_f, scale = s_f / np.sqrt(n),size = N_exact)) *S_f_mult
    if meas_noise:
        S_v += np.abs(np.random.normal(loc = s_n, scale =  s_n/N , size=N_exact))

    S_last = S
    t_meas_last = t_meas
    freq_last = freq
    S_f_mult_last = S_f_mult
    return freq,S_v




def background_PSD(S_v,f):
    '''
    function to calculate background noise and subtract from displacement variance
    S_v: voltage PSD
    f: frequency
    return: displacement variance without background noise
    '''
    G = 10**7 #detection gain [V/m]
    
    #transform S_v into S_x
    S_x = S_v/(G**2)

    f_max = np.max(f)
    f_min = np.min(f)
    df = f[1:] - f[:-1]
    S_x_mean = np.mean(S_x[0:1000])
    disp_var = np.sum(S_x[0:-1]*df)

    disp_var = disp_var - S_x_mean*f_max

    return disp_var




def displacement_variance(S_v,f):
    '''
    function to calculate displacement variance using parseval theorem
    S_v: voltage PSD
    f: frequency
    return: displacement_variance
    '''
    G = 10**7 #detection gain [V/m]
    
    #transform S_v into S_x
    S_x = S_v/(G**2)
    
    #use parseval theorem to calculate displacement variance
    df = f[1:] - f[:-1]
    #integrate S_x over [0,f_max], dropping the last value of S_x with S_x[0:-1] so that dimensions of df and S_x match for integration
    disp_var = np.sum(S_x[0:-1]*df)

    return disp_var


def equip_theorem(disp_var):
    '''
    function to verify that displacement variance is correct using the equipartition theorem
    disp_var: displacement variance
    return: Temperature that has been used for doing the simulation
    '''
    m = 5.5e-15 #resonator mass in [kg]
    f_c = 5.5*1000 #resonance frequency in [Hz]
    w_c = 2*np.pi*f_c #angular resonance frequency

    T = (m*(w_c**2)*disp_var)/constants.k

    return T



###################################################################################
#Task 1:
def task_1():
    f, S_v = gendata(300, 15e3, 30, False)

    print(np.size(f))
    print(np.size(S_v))

    plt.figure(figsize=(16,12))
    plt.loglog(f, S_v)

    plt.xlabel('f [HZ]',fontsize=16)
    plt.ylabel(r'$V_{PSD}$ $[V^{2}/Hz]$',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.show()
    plt.savefig("task_1.png")

    sig_x_2 = displacement_variance(S_v,f)
    print(sig_x_2)
    T = equip_theorem(sig_x_2)
    print(T)


###################################################################################
#Task 2: vary sampling rate and measurement time
def task_2():
    rate = [15e3,12e3,11.5e3,11e3,10.5e3,10.4e3,10.3e3,10.2e3,10e3,9.8e3,9.6e3,9.4e3,9e3,8e3]
    T_vals1 = []
    
    for i in rate:
        f, S_v = gendata(300, i, 30, False)
        sig_x_2 = displacement_variance(S_v,f)
        T = equip_theorem(sig_x_2)
        T_vals1.append(T)

    
    plt.plot(rate,T_vals1)
    plt.xlabel('sampling rate [Hz]')
    plt.ylabel('T [K]')
    #plt.savefig('task_211.png')
    #plt.show()

    x = np.linspace(10000,12000,60)
    T_vals3 = []

    
    for i in x:
        f, S_v = gendata(300, i, 30, False)
        sig_x_2 = displacement_variance(S_v,f)
        T = equip_theorem(sig_x_2)
        T_vals3.append(T)

    plt.figure(figsize=(12,8))
    plt.plot(x,T_vals3)
    plt.xlabel('sampling rate [Hz]',fontsize=16)
    plt.ylabel('T [K]',fontsize=16)
    plt.savefig('task_21.png')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


    time = [2,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
    T_vals2 = []
    
    for i in time:
        f, S_v = gendata(300, 15e3, i, False)
        sig_x_2 = displacement_variance(S_v,f)
        T = equip_theorem(sig_x_2)
        T_vals2.append(T)

    plt.figure(figsize=(12,8))
    plt.plot(time,T_vals2)
    plt.xlabel('aquisition time [s]',fontsize=16)
    plt.ylabel('T [K]',fontsize=16)
    plt.savefig('task_22.png')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    

###################################################################################
#Task 3: vary sampling rate and measurement time
def task_3():
    f, S_v = gendata(300, 15e3, 60, True)

    print(np.size(f))
    print(np.size(S_v))

    plt.figure(figsize=(16,12))
    plt.loglog(f, S_v)
    plt.xlabel('f [HZ]',fontsize=16)
    plt.ylabel(r'$V_{PSD}$ $[V^{2}/Hz]$',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("task_3.png")



    sig_x_2_noise = displacement_variance(S_v,f)
    print('displacement_variance_noise: ' + str(sig_x_2_noise))
    T_noise = equip_theorem(sig_x_2_noise)
    print('T_noise: ' + str(T_noise))
    sig_x_2 = background_PSD(S_v,f)
    print('displacement_variance: ' + str(sig_x_2))
    T = equip_theorem(sig_x_2)
    print('T: ' + str(T))



###################################################################################
#Task 4:
from scipy.optimize import curve_fit
    
def task_4():

    T_list = np.array([300,0.1,4.2,70,1000])
    T_mean = np.zeros(5)
    T_std = np.zeros(5)

    for k,T in enumerate(T_list):
        print('For T = '+str(T),':')
        T_vals = np.zeros(16)
        
        for i in range(16):
            f, S_v = gendata(T, 15e3, 60, True)
            sig_x_2 = background_PSD(S_v,f)
            T_val = equip_theorem(sig_x_2)
            T_vals[i] = T_val

        #print(len(T_vals))
        #print(T_vals)
        T_mean[k] = np.mean(T_vals)
        T_std[k] = np.std(T_vals)
        
        T_std_rel = T_std/T_mean
        
        print('T_mean: ',np.mean(T_vals))
        print('std: ',np.std(T_vals))


        
    #performing the linear fit
    def linear(x,a,b):
        return a + b*x

    popt, pcov = curve_fit(linear, T_list, T_mean, sigma = T_std, absolute_sigma=True)
    std = np.sqrt(np.diag(pcov))

    print(popt)
    print(std)
    print(r'a = %.4f $\pm$ %.3f' %(round(popt[0], 3), round(std[0], 2)))
    print(r'b = %.4f $\pm$ %.4f' %(round(popt[1], 3), round(std[1], 2)))
    

    #ploting the data with the fit
    plt.figure(figsize=(6,6))
    plt.title('data with analytical fit \n $a=%.1f \pm %.2f$ \n $b=%.1f \pm %.2f$' % (popt[0], std[0], popt[1], std[1] ))
    plt.plot(T_list,T_mean,marker='o', label='data')
    plt.plot(T_list, popt[0] + popt[1]*T_list, label='fit: y = a + bx')
    plt.errorbar(T_list, T_mean, yerr=T_std,linestyle='', capsize=3, label='')
    plt.xlabel('T set [K]')
    plt.ylabel('T mean [K]')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.5)
    plt.savefig('task_4.png')
    plt.show()
    print(T_std_rel)
    print(T_std)









if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()


