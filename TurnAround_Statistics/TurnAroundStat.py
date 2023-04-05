####################################################################################################################################
#Turn Around statistical analysis 
####################################################################################################################################

import numpy as np
import LoadData as ld
import CreatingVariableBins as cvb
import Models as mod

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('default')

from lmfit import Model
from scipy import integrate
import scipy.optimize
from scipy.stats import ks_2samp
#_____________________________________________
plt.close("all")

#Loading Data
data1, data2, data3 = ld.Create_DataSet()
data_16, data_17, data_18 = ld.DataToLists(data1, data2, data3)
array16, array17, array18 = ld.FromListsToArrays(data_16, data_17, data_18)
data_tot, dataTot, array_tot = ld.TotalDataSet(data_16, data_17, data_18)
data_tot_A, data_tot_B, data_tot_C, array_totA, array_totB, array_totC = ld.PartialDataSets(data_16, data_17, data_18)
bi16, bi17, bi18, biT, biTA, biTB, biTC = cvb.CreateBins(array16, array17, array18, array_tot, array_totA, array_totB, array_totC)

#Fitting Models
NEl_mod = mod.N_ExpLaw_Model() 
NTpl_mod = mod.N_TruncLaw_Model()

#2016

#fit parameters
A = np.min(array16)
B = np.max(array16)

#fitting models
def HighNormalized_PowerLaw (x, Off, n):
        return ((1-n)/(((B-Off)**(1-n))-(A-Off)**(n-1)))*(1/((x-Off)**n))
HnPl_mod = Model(HighNormalized_PowerLaw) 

#print(A, B)

#Plotting 
#Fits with 3 different fitting functions:
#Power Law Fit;
#Truncated Power Law Fit;
#Exponential Law Fit;
#Residuals 
#Average value of the turn around time

fig1, ax1 = plt.subplots()
n1, bins1, patches1 = ax1.hist(data_16, bins=bi16, facecolor='darkseagreen', alpha=0.5, density=True, label="2016 Data")

binscenters = np.array([0.5 * (bins1[i] + bins1[i+1]) for i in range(len(bins1)-1)])
x = binscenters
y = n1

hnpl_result = HnPl_mod.fit(y, x=x, Off=1.8, n=0.5) 
ax1.plot(binscenters, hnpl_result.best_fit, 'r--', label="Power law")
ax1.plot([], [], 'rx ', label='Reduced Chi-Square ={:.5f}'.format(hnpl_result.redchi))
#ax1.plot([], [], 'rx', label='Offset = {:.3f} +/- {:.3f}'.format(hnpl_result.params['Off'].value, hnpl_result.params['Off'].stderr ))
#ax1.plot([], [], 'rx', label='Exponent = {:.3f} +/- {:.3f}'.format(hnpl_result.params['n'].value, hnpl_result.params['n'].stderr ))

NTpl_mod.set_param_hint('off', value=1.7, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.7, amp=1, lam=0.2, n=0.9) 
ax1.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax1.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
#ax1.plot([], [], 'bx', label='Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['off'].value, ntpl_result.params['off'].stderr ))
#ax1.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['n'].value, ntpl_result.params['n'].stderr ))
#ax1.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
#ax1.plot([], [], 'bx', label='\u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['lam'].value, ntpl_result.params['lam'].stderr ))



nel_result = NEl_mod.fit(y, x=x, off=2, lam=0) 
ax1.plot(binscenters, nel_result.best_fit, 'k:', label="Exponential Law")
ax1.plot([], [], 'kx ', label='Reduced Chi-Square ={:.5f}'.format(nel_result.redchi))
#ax1.plot([], [], 'kx', label='\u03BB = {:.3f} +/- {:.3f}'.format(nel_result.params['lam'].value, nel_result.params['lam'].stderr ))
#ax1.plot([], [], 'kx', label='Offset = {:.3f} +/- {:.3f}'.format(nel_result.params['off'].value, nel_result.params['off'].stderr ))

plt.title('Turn Around Times 2016')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig1.savefig("Img/Turn Around Times 2016.pdf")

#Residuals
plt.close("all")
fig2, (ax2A, ax2B, ax2C) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(10, 5))
hnpl_result.plot_residuals(ax2A, datafmt='r*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax2A.set_ylabel("Residuals")
ax2A.set_title("Residuals for the Power Law - 2016")
ntpl_result.plot_residuals(ax2B, datafmt='b*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax2B.set_ylabel("Residuals")
ax2B.set_title("Residuals for the Truncated Power Law")
nel_result.plot_residuals(ax2C, datafmt='k*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
plt.xlabel("Bincenters")
ax2C.set_ylabel("Residuals")
ax2C.set_title("Residuals for the Exponential Law")
#plt.show()
fig2.savefig("Img/Residuals - 2016.pdf")

#Truncated Power law Fit
plt.close("all")

fig1, ax1 = plt.subplots()
n1, bins1, patches1 = ax1.hist(data_16, bins=bi16, facecolor='darkseagreen', alpha=0.5, density=True, label="2016 Data")

binscenters = np.array([0.5 * (bins1[i] + bins1[i+1]) for i in range(len(bins1)-1)])
x = binscenters
y = n1

NTpl_mod.set_param_hint('off', value=1.7, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.7, amp=1, lam=0.2, n=0.9) 
ax1.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax1.plot([], [], 'b. ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax1.plot([], [], 'bx', label='Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['off'].value, ntpl_result.params['off'].stderr ))
ax1.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['n'].value, ntpl_result.params['n'].stderr ))
ax1.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax1.plot([], [], 'bx', label='\u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['lam'].value, ntpl_result.params['lam'].stderr ))


plt.title('Turn Around Times 2016')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig1.savefig("Img/Turn Around Times 2016 Fit.pdf")

#èrinting results
print('____________________2016_______________________________')
print('RESULTs OF THE POWER LAW FIT')
print('-------------------------------------------------------')
print(hnpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE TRUNCATED POWER LAW FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE EXPONENTIAL FIT')
print('-------------------------------------------------------')
print(nel_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')

#Removing correlation
def TPL_corr(x, uoff, ulam, un, amp):
     return (amp/(x-(-0.957*amp+uoff))**((amp-un)/0.887))*(np.exp(-(-(amp-ulam)/0.692)*(x)))

Corr_mod = Model(TPL_corr)

#Uncorrelated Turn Around Times
plt.close("all")
fig1, ax1 = plt.subplots()
n1, bins1, patches1 = ax1.hist(data_16, bins=bi16, facecolor='darkseagreen', alpha=0.5, density=True, label="2016 Data")

binscenters = np.array([0.5 * (bins1[i] + bins1[i+1]) for i in range(len(bins1)-1)])
x = binscenters
y = n1

#Corr_mod.set_param_hint('n', value=0.5, min=0, max=2.0)
#Corr_mod.set_param_hint('uamp', value=0.0, min=0, max=2.0)
Corr_mod.set_param_hint('ulam', value=0.2, min=0, max=3.0)
Corr_mod.set_param_hint('uoff', value=1.9, min=0.0, max=3.0)
ntpl_result = Corr_mod.fit(y, x=x, uoff=1.0,  ulam=0.2, un=0.5, amp=0) 
ax1.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax1.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax1.plot([], [], 'bx', label='Uncorrelated Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['uoff'].value, ntpl_result.params['uoff'].stderr ))
ax1.plot([], [], 'bx', label='uncorrelated Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['un'].value, ntpl_result.params['un'].stderr ))
ax1.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax1.plot([], [], 'bx', label='Uncorrelated \u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['ulam'].value, ntpl_result.params['ulam'].stderr ))

plt.title('Uncorrelated Turn Around Times 2016')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()

print('_______________________________________________________')
print('RESULTs OF THE Uncorrelated Truncated Power Law FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())

#Uncorrelated Residual
plt.close("all")
ntpl_result.plot_residuals(datafmt='b*')
plt.xlabel("Bincenters")
plt.ylabel("Residuals")
plt.title("Uncorrelated Residual Plot 2016")
#plt.show()

#Average value of the turn around time
#### Test for the interval of the integrand - 2016
x1 = np.linspace(2, 150, 1000)
a1 = 0.37082720*x1*(np.power((x1-2),(-0.73454371)))*(np.exp(-0.10000000*x1))
plt.close("all")
fig1, ax1 = plt.subplots()
ax1.plot(x1, a1, "b-")
ax1.set_title("f_trunc16(t)")
#plt.show()
                      
#####2016 Epectation Value
def f_trunc16(t):
   a = 0.37082720
   b = 2
   d = 0.10000000
   n = 0.73454371
   return t*((a/(np.power(t-b, n)))*np.exp(-d*t))
val16, err16 = integrate.quad(f_trunc16, 2, 250)
                      
#èrinting results
print("_____________________________________________________")
print("2016: E[t_ta]=",val16,"+-", err16)
print("_____________________________________________________")  

#2017

#fit parameters
A = np.min(array17)
B = np.max(array17)

#fitting models
def HighNormalized_PowerLaw (x, Off, n):
        return ((1-n)/(((B-Off)**(1-n))-(A-Off)**(n-1)))*(1/((x-Off)**n))
HnPl_mod = Model(HighNormalized_PowerLaw) 

#print(A, B)

#Plotting 
#Fits with 3 different fitting functions:
#Power Law Fit;
#Truncated Power Law Fit;
#Exponential Law Fit;
#Residuals 
#Average value of the turn around time
plt.close("all")
fig3, ax3 = plt.subplots()
n3, bins3, patches3 = ax3.hist(data_17, bins=bi17, facecolor='steelblue', alpha=0.4, density=True, label="2017 Data")

binscenters = np.array([0.5 * (bins3[i] + bins3[i+1]) for i in range(len(bins3)-1)])
x = binscenters
y = n3

hnpl_result = HnPl_mod.fit(y, x=x, Off=1.8, n=0.5) 
ax3.plot(binscenters, hnpl_result.best_fit, 'r--', label="Power law")
ax3.plot([], [], 'rx ', label='Reduced Chi-Square ={:.5f}'.format(hnpl_result.redchi))

NTpl_mod.set_param_hint('off', value=1.7, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.7, amp=0.1, lam=0.2, n=1) 
ax3.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax3.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))

nel_result = NEl_mod.fit(y, x=x, off=2, lam=0) 
plt.plot(binscenters, nel_result.best_fit, 'k:', label="Exponential Power law")
ax3.plot([], [], 'kx ', label='Reduced Chi-Square ={:.5f}'.format(nel_result.redchi))
plt.title('Turn Around Times 2017')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig3.savefig("Img/Turn Around Times 2017.pdf")

#Residuals
plt.close("all")
fig4, (ax4A, ax4B, ax4C) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(10, 5))
hnpl_result.plot_residuals(ax4A, datafmt='r*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax4A.set_ylabel("Residuals")
ax4A.set_title("Residuals for the Power Law - 2017")
ntpl_result.plot_residuals(ax4B, datafmt='b*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax4B.set_ylabel("Residuals")
ax4B.set_title("Residuals for the Truncated Power Law")
nel_result.plot_residuals(ax4C, datafmt='k*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
plt.xlabel("Bincenters")
ax4C.set_ylabel("Residuals")
ax4C.set_title("Residuals for the Exponential Law")
#plt.show()
fig4.savefig("Img/Residuals - 2017.pdf")

#Truncated Power law Fit
plt.close("all")
fig5, ax5 = plt.subplots()
n3, bins3, patches3 = ax5.hist(data_17, bins=bi17, facecolor='steelblue', alpha=0.4, density=True, label="2017 Data")

binscenters = np.array([0.5 * (bins3[i] + bins3[i+1]) for i in range(len(bins3)-1)])
x = binscenters
y = n3

NTpl_mod.set_param_hint('off', value=1.7, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.7, amp=0.1, lam=0.2, n=1) 
ax5.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax5.plot([], [], 'b. ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax5.plot([], [], 'bx', label='Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['off'].value, ntpl_result.params['off'].stderr ))
ax5.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['n'].value, ntpl_result.params['n'].stderr ))
ax5.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax5.plot([], [], 'bx', label='\u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['lam'].value, ntpl_result.params['lam'].stderr ))

ax5.set_title('Turn Around Times 2017')
ax5.set_xlabel('Turn Around Times [h]')
ax5.set_ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig5.savefig("Img/Turn Around Times 2017 Fit.pdf")

print('__________________2017_________________________________')
print('RESULTs OF THE POWER LAW FIT')
print('-------------------------------------------------------')
print(hnpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE TRUNCATED POWER LAW FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE EXPONENTIAL FIT')
print('-------------------------------------------------------')
print(nel_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')

#Average value of the turn around time
#### Test for the interval of the integrand - 2017
x2 = np.linspace(2, 150, 1000)
a2 = 0.29687182*x2*(np.power((x2-2),(-0.50729741)))*(np.exp(-0.10422466*x2))
#plt.close("all")
fig2, ax2 = plt.subplots()
ax2.plot(x2, a2, "b-")
ax2.set_title("f_trunc17(t)")
#plt.show()
                      
#####2017 Epectation Value
def f_trunc17(t):
   a = 0.29687182
   b = 2
   d = 0.10422466
   n = 0.50729741
   return t*((a/(np.power(t-b, n)))*np.exp(-d*t))
val17, err17 = integrate.quad(f_trunc17, 2, 250)
                      
#èrinting results
print("_____________________________________________________")
print("2017: E[t_ta]=",val17,"+-", err17)
print("_____________________________________________________") 

#2018

#Fit parameters
A = np.min(array18)
B = np.max(array18)

#Fitting Models
def HighNormalized_PowerLaw (x, Off, n):
        return ((1-n)/(((B-Off)**(1-n))-(A-Off)**(n-1)))*(1/((x-Off)**n))
HnPl_mod = Model(HighNormalized_PowerLaw)

#print(A, B)

#Plotting 
#Fits with 3 different fitting functions:
#Power Law Fit;
#Truncated Power Law Fit;
#Exponential Law Fit;
#Residuals 
#Average value of the turn around time
plt.close("all")
fig6, ax6 = plt.subplots()
n3, bins3, patches3 = ax6.hist(data_18, bins=bi18, facecolor='r', alpha=0.2, density=True, label="2018 Data")

binscenters = np.array([0.5 * (bins3[i] + bins3[i+1]) for i in range(len(bins3)-1)])
x = binscenters
y = n3

hnpl_result = HnPl_mod.fit(y, x=x, Off=1, n=0.2) 
ax6.plot(binscenters, hnpl_result.best_fit, 'r--', label="Power law")
ax6.plot([], [], 'rx ', label='Reduced Chi-Square ={:.5f}'.format(hnpl_result.redchi))

#NTpl_mod.set_param_hint('off', value=1.0, min=0, max=2.0)
#ntpl_result = NTpl_mod.fit(y, x=x, off=1, amp=1, lam=0.2, n=0.9) 

NTpl_mod.set_param_hint('off', value=1.9, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.9, amp=0.5, lam=0.2, n=0.5) 
ax6.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax6.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))

nel_result = NEl_mod.fit(y, x=x, off=2, lam=0) 
ax6.plot(binscenters, nel_result.best_fit, 'k:', label="Exponential Power law")
ax6.plot([], [], 'kx ', label='Reduced Chi-Square ={:.5f}'.format(nel_result.redchi))
plt.title('Turn Around Times 2018')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig6.savefig("Img/Turn Around Times 2018.pdf")

#Residuals
plt.close("all")
fig7, (ax7A, ax7B, ax7C) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(10, 5))
hnpl_result.plot_residuals(ax7A, datafmt='r*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax7A.set_ylabel("Residuals")
ax7A.set_title("Residuals for the Power Law - 2018")
ntpl_result.plot_residuals(ax7B, datafmt='b*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax7B.set_ylabel("Residuals")
ax7B.set_title("Residuals for the Truncated Power Law")
nel_result.plot_residuals(ax7C, datafmt='k*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
plt.xlabel("Bincenters")
ax7C.set_ylabel("Residuals")
ax7C.set_title("Residuals for the Exponential Law")
#plt.show()
fig7.savefig("Img/Residuals - 2018.pdf")

#Truncated Power law Fit
plt.close("all")

fig8, ax8 = plt.subplots()
n3, bins3, patches3 = ax8.hist(data_18, bins=bi18, facecolor='r', alpha=0.2, density=True, label="2018 Data")

binscenters = np.array([0.5 * (bins3[i] + bins3[i+1]) for i in range(len(bins3)-1)])
x = binscenters
y = n3

NTpl_mod.set_param_hint('off', value=1.9, min=1.5, max=2.0)
NTpl_mod.set_param_hint('lam', value=0.2, min=0.1, max=0.4)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.9, amp=0.5, lam=0.2, n=0.5) 
ax8.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax8.plot([], [], 'b. ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax8.plot([], [], 'bx', label='Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['off'].value, ntpl_result.params['off'].stderr ))
ax8.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['n'].value, ntpl_result.params['n'].stderr ))
ax8.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax8.plot([], [], 'bx', label='\u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['lam'].value, ntpl_result.params['lam'].stderr ))




plt.title('Turn Around Times 2018')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig8.savefig("Img/Turn Around Times 2018 Fit.pdf")

print('__________________2018_________________________________')
print('RESULTs OF THE POWER LAW FIT')
print('-------------------------------------------------------')
print(hnpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE TRUNCATED POWER LAW FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE EXPONENTIAL FIT')
print('-------------------------------------------------------')
print(nel_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')

#Average value of the turn around time
#### Test for the interval of the integrand - 2018
x3 = np.linspace(1.50000008, 150, 1000)
a3 = 0.316*x3*(np.power((x3-1.901),(-0.964)))*(np.exp(-0.017*x3))
#plt.close("all")
fig3, ax3 = plt.subplots()
ax3.plot(x3, a3, "b-")
ax3.set_title("f_trunc18(t)")
#plt.show()
                      
#####2018 Epectation Value
def f_trunc18(t):
   a = 0.33245703
   b = 1.50000008
   d = 0.1642191
   n = 0.27151320
   return t*((a/(np.power(t-b, n)))*np.exp(-d*t))
val18, err18 = integrate.quad(f_trunc18, 1.50000008, 250)
                      
#èrinting results
print("_____________________________________________________")
print("2018: E[t_ta]=",val18,"+-", err18)
print("_____________________________________________________") 

#TOTAL RUN

#Fit parameters
A = np.min(array_tot)
B = np.max(array_tot)

#Fitting Models
def HighNormalized_PowerLaw (x, Off, n):
        return ((1-n)/(((B-Off)**(1-n))-(A-Off)**(n-1)))*(1/((x-Off)**n))
HnPl_mod = Model(HighNormalized_PowerLaw)

#print(A, B)

#Plotting 
#Fits with 3 different fitting functions:
#Power Law Fit;
#Truncated Power Law Fit;
#Exponential Law Fit;
#Residuals 
#Average value of the turn around time
plt.close("all")
fig9, ax9 = plt.subplots()
n4, bins4, patches4 = ax9.hist(dataTot, bins=biT, facecolor='gold', alpha=0.5, density=True, label="Run 2 Data")

binscenters = np.array([0.5 * (bins4[i] + bins4[i+1]) for i in range(len(bins4)-1)])
x = binscenters
y = n4


hnpl_result = HnPl_mod.fit(y, x=x, Off=1.8, n=0.5) 
ax9.plot(binscenters, hnpl_result.best_fit, 'r--', label="Power law")
ax9.plot([], [], 'rx ', label='Reduced Chi-Square ={:.5f}'.format(hnpl_result.redchi))

NTpl_mod.set_param_hint('off', value=1.9, min=1.5, max=2.0)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.9, amp=1, lam=0, n=0.9) 
ax9.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax9.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))

nel_result = NEl_mod.fit(y, x=x, off=2, lam=0) 
ax9.plot(binscenters, nel_result.best_fit, 'k:', label="Exponential law")
ax9.plot([], [], 'kx ', label='Reduced Chi-Square ={:.5f}'.format(nel_result.redchi))

plt.title('Turn Around Times Total')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig9.savefig("Img/Turn Around Times Run 2.pdf")

#Residuals
plt.close("all")
fig10, (ax10A, ax10B, ax10C) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(10, 5))
hnpl_result.plot_residuals(ax10A, datafmt='r*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax10A.set_ylabel("Residuals")
ax10A.set_title("Residuals for the Power Law - RUN 2")
ntpl_result.plot_residuals(ax10B, datafmt='b*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
ax10B.set_ylabel("Residuals")
ax10B.set_title("Residuals for the Truncated Power Law")
nel_result.plot_residuals(ax10C, datafmt='k*', yerr=None, data_kws=None, fit_kws=None, ax_kws=None, parse_complex='abs')
plt.xlabel("Bincenters")
ax10C.set_ylabel("Residuals")
ax10C.set_title("Residuals for the Exponential Law")
#plt.show()
fig10.savefig("Img/Residuals - Run 2.pdf")

#Truncated Power law Fit
plt.close("all")
fig11, ax11 = plt.subplots()
n4, bins4, patches4 = ax11.hist(dataTot, bins=biT, facecolor='gold', alpha=0.5, density=True, label="Run 2 Data")

binscenters = np.array([0.5 * (bins4[i] + bins4[i+1]) for i in range(len(bins4)-1)])
x = binscenters
y = n4


NTpl_mod.set_param_hint('off', value=1.9, min=1.5, max=2.0)
ntpl_result = NTpl_mod.fit(y, x=x, off=1.9, amp=1, lam=0, n=0.9) 
ax11.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax11.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax11.plot([], [], 'bx', label='Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['off'].value, ntpl_result.params['off'].stderr ))
ax11.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['n'].value, ntpl_result.params['n'].stderr ))
ax11.plot([], [], 'bx', label='Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax11.plot([], [], 'bx', label='\u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['lam'].value, ntpl_result.params['lam'].stderr ))

plt.title('Turn Around Times Total')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()
fig11.savefig("Img/Turn Around Times Run 2 Fit.pdf")

print('__________________RUN 2_________________________________')
print('RESULTs OF THE POWER LAW FIT')
print('-------------------------------------------------------')
print(hnpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE TRUNCATED POWER LAW FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')
print('_______________________________________________________')
print('RESULTs OF THE EXPONENTIAL FIT')
print('-------------------------------------------------------')
print(nel_result.fit_report())
print('_______________________________________________________')
print('_______________________________________________________')

#Removing Correlation

def TPL_corr(x, uoff, ulam, un, amp):
     return (amp/(x-(0.938*amp+uoff))**((amp-un)/0.858))*(np.exp(-(-(amp-ulam)/0.683)*(x)))

Corr_mod = Model(TPL_corr)

#Uncorrelated Turn Around Times Total
plt.close("all")
fig11, ax11 = plt.subplots()
n4, bins4, patches4 = ax11.hist(dataTot, bins=biT, facecolor='gold', alpha=0.5, density=True, label="Histogram")

binscenters = np.array([0.5 * (bins4[i] + bins4[i+1]) for i in range(len(bins4)-1)])
x = binscenters
y = n4

#Corr_mod.set_param_hint('n', value=0.5, min=0, max=2.0)
#Corr_mod.set_param_hint('uamp', value=0.0, min=0, max=2.0)
Corr_mod.set_param_hint('ulam', value=0.2, min=0, max=3.0)
Corr_mod.set_param_hint('uoff', value=1.9, min=0.0, max=3.0)
ntpl_result = Corr_mod.fit(y, x=x, uoff=1.0,  ulam=0.2, un=0.5, amp=0) 
ax11.plot(binscenters, ntpl_result.best_fit, 'b--', label="Truncated Power law")
ax11.plot([], [], 'bx ', label='Reduced Chi-Square ={:.5f}'.format(ntpl_result.redchi))
ax11.plot([], [], 'bx', label='Uncorrelated Offset = {:.3f} +/- {:.3f}'.format(ntpl_result.params['uoff'].value, ntpl_result.params['uoff'].stderr ))
ax11.plot([], [], 'bx', label='Exponent = {:.3f} +/- {:.3f}'.format(ntpl_result.params['un'].value, ntpl_result.params['un'].stderr ))
ax11.plot([], [], 'bx', label='Uncorrelated Amplitude = {:.3f} +/- {:.3f}'.format(ntpl_result.params['amp'].value, ntpl_result.params['amp'].stderr ))
ax11.plot([], [], 'bx', label='Uncorrelated \u03BB = {:.3f} +/- {:.3f}'.format(ntpl_result.params['ulam'].value, ntpl_result.params['ulam'].stderr ))

plt.title('Uncorrelated Turn Around Times Total')
plt.xlabel('Turn Around Times [h]')
plt.ylabel('Normalized Frequency')


plt.legend(loc='best')
#plt.show()

print('_______________________________________________________')
print('RESULTs OF THE Exponential Law FIT')
print('-------------------------------------------------------')
print(ntpl_result.fit_report())

#Uncorrelated Residuals
plt.close("all")
ntpl_result.plot_residuals(datafmt='b*')
plt.xlabel("Bincenters")
plt.ylabel("Residuals")
plt.title("Uncorrelated Residual Plot RUN 2")
#plt.show()


#Kolmogorov-Smirnov Test

#Performing the tests
test1 = ks_2samp(data_16, data_17)
test2 = ks_2samp(data_16, data_18)
test3 = ks_2samp(data_17, data_18)

print(test1)
print(test2)
print(test3)
print('Lokking at the K-S results t is possible to say that\n\
    data coming from 2016 and 2017 seems to be distributed according\n\
    to the same distribution, as data coming from 2017 and 2018.\n\
    However, data for 2016 and 2018 samples it is not possible to accept\n\
    the hyothesis of equal distributions. This last conslusion can be linked\n\
    to the consistent difference in the number of objects in the two samples.\n\
    Furthermore, taking into account the results as a whole, it is possible to state,\n\
    albeit with a certain limit of uncertainty, that the turn-around times of different\n\
    years are distributed following the same distribution.') 
