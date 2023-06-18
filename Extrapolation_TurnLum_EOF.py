#################################################################################################################################################################
##
######## Thiziri Amezza 23-02-2023 #############################################################################################################################
##
#################################################################################################################################################################
import numpy as np
import LoadData as ld
import pandas as pd
import LuminosityOptimization as lo
import matplotlib.pyplot as plt
import time as t
#from lmfit import Model
from scipy import integrate
import scipy.optimize
from scipy.optimize import curve_fit
#_____________________________________________
plt.close("all")

#Font setting
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  "font.size": 12
})

#defining the start time of the program
start=t.time()

#Selecting Current Year
year=18
tmp = 12

#plotting
plot=True

#model parameters 
if year==16:
   n_i, k_b, B_s, E_s, B_r, G_r, S_int, n_c, N_i, T_hc, T_ph, S_z, S_s, Fe, f_rev, Xi, Eps= lo.Parameters2016()
elif year==17:
    n_i, k_b, B_s, E_s, B_r, G_r, S_int, n_c, N_i, T_hc, T_ph, S_z, S_s, Fe, f_rev, Xi, Eps= lo.Parameters2017()
elif year==18:
    n_i, k_b, B_s, E_s, B_r, G_r, S_int, n_c, N_i, T_hc, T_ph, S_z, S_s, Fe, f_rev, Xi, Eps= lo.Parameters2018()

Lmes16, Lmes17, Lmes18=ld.MeasuredLuminosity() #femtobarn^-1
L_int_summary_16 = Lmes16*1e9
L_int_summary_17 = Lmes17*1e9
L_int_summary_18 = Lmes18*1e9
L_int_summary_16 = np.array(L_int_summary_16)
L_int_summary_17 = np.array(L_int_summary_17)
L_int_summary_18 = np.array(L_int_summary_18)

#loading fill number
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()

#load turnaround times and fill times 
data_ta16, data_tf16, data_ta17, data_tf17, data_ta18, data_tf18 = ld.loadFill()
data_ta16_sec = data_ta16*3600 
data_tf16_sec = data_tf16*3600  
data_ta17_sec = data_ta17*3600 
data_tf17_sec = data_tf17*3600
data_ta18_sec = data_ta18*3600 
data_tf18_sec = data_tf18*3600

if year==16:
    FillNumber=FillNumber16
    ta=data_ta16_sec
    tf=data_tf16_sec
    L_int_summary = L_int_summary_16
elif year==17:
    FillNumber=FillNumber17
    FillNumber_Prev=FillNumber16
    previous_year=16
    ta=data_ta17_sec
    tf=data_tf17_sec
    L_int_summary = L_int_summary_17
elif year==18:
    FillNumber=FillNumber18
    FillNumber_Prev=FillNumber17
    previous_year=17
    ta=data_ta18_sec
    tf=data_tf18_sec
    L_int_summary = L_int_summary_18
#print(FillNumber)

#Fill to delete [6638,6666,6174,7061,7065,7087,7124,7127]
skip16=[5097,5112,5117,5264,5406,5427,5439]
skip17=[5837,5840,6019,6055,6060,6082,6084,6089,6104,6106,6110,6116,6142,6143,6152,6156,6158,6160,6167,6168,6169,6193,6258,6268,6271,6272,6275,6283,6285,6287,6288,6291,6298,6303,6304,6305,6308,6311,6312,6314,6317,6324,6325,6337,6346,6356,6360,6362,6364,6371]
skip18=[6654,6659,6666,6688,6690,6694,6696,6706,6729,6741,6747,6752,6782,6778,6882,6890,6892,6901,6929,6939,6961,7036,7039,7118,7120,7135,7145,7217,7245,7259,7271,7324]
skip = skip16 + skip17 + skip18

for fill in skip:
    FillNumber=np.delete(FillNumber, (np.where(FillNumber==(fill))[0]))
for i in range(len(FillNumber)):
    if FillNumber[i] in skip:
        continue

#defining the double-exponential decay fit function
def fit(x, a, b, c, d):
    return (a*np.exp((-b)*x))+(c*np.exp((-d)*x))

#defining the DA MODEL-4 fit function
def L_model2(x, eps_ni, B, k):
        x = np.where(x>1, x, 2 ) 
        np.nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None) 
        #temp = (np.power(np.log(x), k))
        #temp[np.isnan(temp)] = 0.0
        #D = B / (np.power(np.log(x), k))
        #D = B / (np.log(x) ** k)
        #D = B *np.exp(-k*np.log((np.log(x))))
        
        D = B * np.power(k / (2 * np.exp(1)), k) / (np.power(np.log(x), k))
        #D = B *np.exp(-k*np.log((np.log(x))))
        L = (1/(1 + eps_ni * (x-1))**2) - ((1 + D**2)*np.exp(-(D**2))) * ((2 - (1 + D**2)*np.exp(-(D**2)/2)))
        return L 

def compute_r_squared(y, y_fit):
    ss_tot = ((y - np.mean(y))**2).sum()
    ss_res = ((y - y_fit)**2).sum()
    r_squared = 1 - ss_res / ss_tot
    return r_squared

###########################################################  Define function for computing quality of fit ##################################################################

def Cut_Fit(year, text):
    """Function that performs the necessary cut on the current fill

    Args:
        year (int): current year
        text (str): current fill

    Returns:
        L_fit: cutted data
        T_fit_real: times in second for the cutted data fit
        Y: Luminosity evolution form the fit
        a: fitting parameter
        b: fitting parameter
        c: fitting parameter
        d: fitting parameter
        chi: reduced chi square of the fit
        L_evol: raw luminosity data
        Times: raw Unix time
    """
    year=str(year)
    f=open('ATLAS/ATLAS_fill_20{}/{}_lumi_ATLAS.txt'.format(year, text),"r")
    lines=f.readlines()
    L_evolx=[]
    times=[]
    for x in lines:
        times.append(int(x.split(' ')[0]))  
        L_evolx.append(float(x.split(' ')[2]))
        
    f.close()
    Times = np.array(times)
    L_evol = np.array(L_evolx)

    #deleting the null values of the luminosity
    zero=np.where(L_evol<100)
    L_zero=np.delete(L_evol, zero)
    T_zero=np.delete(Times, zero)
        
    #check for enough points
    if len(L_zero)<10:
        zero=np.where(L_evol<5)
        L_zero=np.delete(L_evol, zero)
        T_zero=np.delete(Times, zero)

    #defining the derivative 
    dy = np.zeros(L_zero.shape)
    dy[0:-1] = np.diff(L_zero)/np.diff(T_zero)


    #start to slim down the fit interval       
    L_tofit=[]
    T_tofit=[]
    for idx in range(len(L_zero)):
        #cancelling too strong derivative points
        if dy[idx]<0 and dy[idx]>-1.5:
            L_tofit.append(L_zero[idx])
            T_tofit.append(T_zero[idx])
        if dy[idx]>0 or dy[idx]<-1.5:
            continue     
        
    #evaluating the differences between two subsequent points
    diff=np.diff(L_tofit)
        
    #deleting the discrepancies
    thr=np.max(abs(diff))*0.05
    idx_diff= np.where(abs(diff)>thr)[0]+1
        
    #new slim down of data
    L_tofit2=np.delete(L_tofit, idx_diff)
    T_tofit2=np.delete(T_tofit, idx_diff)
        
    #check for enough points
    if len(L_tofit2) < 30:
        L_tofit2=L_tofit
        T_tofit2=T_tofit
        
    L_fit=L_tofit2
    T_fit=T_tofit2     

    L_fit=np.array(L_fit)
    T_fit=np.array(T_fit)

    #transforming the times from unix in seconds
    T_fit_real=T_fit-np.amin(T_fit)


    return T_fit_real,L_fit,L_evol, Times

L_int_opt=[]

for i in range(len(FillNumber)):

    text = str(int(FillNumber[i])) #number of current fill
    T_fit_real,L_fit,L_evol, Times= Cut_Fit(year, text)

    #defining the new time variable representing the number of turns
    Turn=[] 
    Turn=np.array(Turn)
    for el in T_fit_real:
        tau=(f_rev*el+1)
        Turn=np.append(Turn, tau)

    L_min = min(L_fit)
    L_max = max(L_fit)
    #
    # Normlise L_fit
    L_Tau = L_fit/L_max 


    try:
        idx = np.where(Turn >= (18000*f_rev+1))[0][0] #1h
        
    except IndexError:
        # handle the case where no indices satisfy the condition
        print("Error: No indices satisfy the condition.")
        idx = len(Turn)
        L_norm = L_Tau[:idx]
        Tau_fit = Turn[:idx]
        #Tau_fit = Turn[:idx]
        continue
    else:
        # Select data from start up to 5 hour
        L_norm = L_Tau[:idx]
        Tau = Turn[:idx]

    Tau = np.array(Tau)
    L_norm = np.array(L_norm)

    #ta = data_ta16_sec[int(i)]
    ######tau_opt = lo.tau_opt_eval(N_i, Eps, (f_rev*ta[int(i)]+1))

    #compare the optimal time which in our case here the last value of the EOF time array Turn[-1] with the last value of the Tau array (Spanned data)
    ##tau_opt = Turn[0]+tau_opt
    ######## maybe Turn = (tau[0]-1)+Turn
    Turn = Tau[0]+Turn
    print("tau[0]+Turn=", Turn)
  
    control = Turn>Tau[len(Tau)-1] 
    print("tau[0]+Turn=", Turn, "Extrapolation needed:", control)       
    limit=Tau[len(Tau)-1] #last value of the Tau array

    #plotting results 
    plt.close("all")
    fig,  ax = plt.subplots()
    
    ######################################################################## Need to extrapolate Tau until the EOF TURN  #########################################################################
    
    if int(Turn[-1])>limit:
      
       #X = np.arange(limit+(f_rev*60+1), int(tau_opt), (f_rev*60+1)) #definition of the extrapolation interval   if int(tau_opt)>limit:
       X = np.arange(limit+(f_rev*60+1), int(Turn[-1]),(f_rev*60+1))

       if np.size(X) == 0:
           # Handle the case where X is empty
           print('X is empty iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', 'FillNumber',FillNumber[i])
           continue
       else:
           # Perform the calculation using X
           X = np.arange(limit+(f_rev*60+1), int(Turn[-1]),(f_rev*60+1))

       #defining the fit intervals
       j=0
       L_fit=np.array([])
       Tau_fit=np.array([])
       L_norm=np.array(L_norm)
       Tau_t=np.array(Tau)

       #filling the fit interval until it encounters a big step in the data given by the derivatives
       for a in range(len(L_norm)):
           L_fit=np.append(L_fit, L_norm[len(L_norm)-1-j])
           Tau_fit=np.append(Tau_fit, Tau_t[len(Tau_t)-1-j])
           j=j+1

       Tau_norm = Tau_fit/(np.amax(X)-np.amin(Tau_fit))
       

       norm_X=[]
       norm_X=np.array(norm_X)
       for element in X:
           z=(element-np.amin(Tau_fit))/(np.amax(X)-np.amin(Tau_fit))
           norm_X=np.append(norm_X, z)

        ################ Set up arrays for plotting the quality of the fit #############################################################################

        # Set up arrays for plotting
       end_times = []
       model2_r_squareds = []
       double_gaussian_r_squareds = []

        #### performing fit of last segments of data ########## Using DA Model-2 #############################################################################
       #p0 = (1e-10, 10 , 1) (4.5e-10, 155, 0.9) (5e-10, 155, 0.9) (9.5e-10, 155, 0.9)
       p0 = (9.5e-10, 155, 0.9)
       popt, pcov = curve_fit(L_model2, Tau_fit, L_fit,p0,bounds=([1e-10, 150., 0.85], [2.e-9, 300., 2]),maxfev=500000)

       eps_ni, B, k = popt
       print('length ppppppppppppppppp', popt)

       model2_fit = L_model2(Tau_fit, *popt)
       model2_r_squared = compute_r_squared(L_fit, model2_fit) 
       
       print('model2_r_squared model2_r_squared', model2_r_squared)
       

       Y = L_model2(X, eps_ni, B, k)     

       
       #### performing fit of last segments of data ########### Using Double-Exponential ####################################################################
       popt, pcov = curve_fit(fit, Tau_norm, L_fit,bounds=(0, [1,50,1,50]),maxfev=5000)

       a, b, c,d = popt


       double_gaussian_fit = fit(Tau_norm, *popt)
       double_gaussian_r_squared = compute_r_squared(L_fit, double_gaussian_fit)

       print('double_gaussian_r_squared double_gaussian_r_squared', double_gaussian_r_squared)


       #YY = fit(norm_X, a, b/(np.amax(X)-np.amin(Tau_fit)), c,d/(np.amax(X)-np.amin(Tau_fit)))
       YY = fit(norm_X, a, b, c,d)
           
       print('length YYYYYYYYYYY', len(Y))
       print('length XXXXXXXXXXX', len(X))

       ########################################    Save results for plotting     ####################################################################
       # 
       end_times.append(Tau_fit[-1])
       model2_r_squareds.append(model2_r_squared)
       double_gaussian_r_squareds.append(double_gaussian_r_squared)

       ################################## I use DA Model-2  #################################################################################################

       k1=0  # We start to count for extrapolation until the optimal fill reached if needed and we add the extrapolated value Y and X to the existed data
       L=np.append(L_norm, Y) 
       T=np.append(Tau_t, X)
       Lumi_evol=np.array([]) 
       Time=np.array([])
       for m in T: #for element in extrapolated time k = k+1 until the t_opt smaller than the limit
           if int(Turn[-1])>=m:
               Lumi_evol=np.append(Lumi_evol, L[k1])
               Time=np.append(Time, T[k1])
               k1=k1+1

        ################################## I use Double-Exponential  ########################################################################################
    
       k0=0  # We start to count for extrapolation until the optimal fill reached if needed and we add the extrapolated value Y and X to the existed data
       LL=np.append(L_norm, YY) 
       T=np.append(Tau_t, X)
       Lumi_ev=np.array([]) 
       Tim=np.array([])
       for m in T: #for element in extrapolated time k = k+1 until the t_opt smaller than the limit
           if int(Turn[-1])>=m:
               Lumi_ev=np.append(Lumi_ev, LL[k0])
               Tim=np.append(Tim, T[k0])
               k0=k0+1


    ########################## Here what should be plotted in case the extrapolation needed, all the cutting steps considered #########################
    #### Fit interval = Selected data means 
       
       ax.plot(Turn/1e9, L_Tau, "b.", label='Fill data', markersize=8)
       #ax.plot(Tau_fit/1e9, L_fit,color='lime', label='Fit interval', markersize=8)
       #ax.plot(Tau_t/1e9, L_norm, "y.", label='Selected data', markersize=8)
       ax.plot(Tau_fit/1e9, L_fit,"y.", label='Fit interval', markersize=8) #The Selected data point
       
       ax.plot(X/1e9,Y, "g.", label='Extrapolated Interval Mod-2', markersize=6)
       ax.plot(X/1e9,YY, "c.", label='Extrapolated Interval Double-Exp', markersize=6) #ADDED
       ax.plot(Time/1e9, Lumi_evol, "r.", label='Luminosity Evolution Mod-2', markersize=2)
       ax.plot(Tim/1e9, Lumi_ev, "m.", label='Luminosity Evolution Double-Exp', markersize=2) #ADDED
       ax.set_xlabel('Normlised Time in Turns 'r'$\tau$(10$^9$)')
       ax.set_ylabel("'${L}/{L_i}$'") 
       ax.set_title('Extrapolated Luminosity evolution {}'.format(text))
       plt.legend(loc='best')
       plt.savefig('Extrapolation/20{}/{}_Opt_Lumi_Evol_mod2_Exp.pdf'.format(str(year),text))  
       #plt.show()  
       # 
       plt.close("all")
       fig0,  ax0 = plt.subplots()
       ax0.plot(Turn/1e9, L_Tau, "b.", label='Fill data', markersize=8)
       ax0.plot(Tau_fit/1e9, L_fit,"y.", label='Fit interval', markersize=8) #The Selected data point       
       #ax0.plot(X/1e9,Y, "r.", label='Extrapolated Interval Mod-2', markersize=4)
       #ax0.plot(X/1e9,YY, "m.", label='Extrapolated Interval Double-Exp', markersize=4) #ADDED
       ax0.plot(Time/1e9, Lumi_evol, "r-", label='Luminosity Evolution Mod-2', markersize=2)
       ax0.plot(Tim/1e9, Lumi_ev, "m-", label='Luminosity Evolution Double-Exp', markersize=2) #ADDED
       ax0.set_xlabel('Normlised Time in Turns 'r'$\tau$(10$^9$)')
       ax0.set_ylabel("'${L}/{L_i}$'") 
       ax0.set_title('Extrapolated Fit for Luminosity evolution {}'.format(text))
       plt.legend(loc='best')
       plt.savefig('Extrapolation/20{}/{}_Extrapolated_Fit_mod2_Exp.pdf'.format(str(year),text))  
       #plt.show()"""  
       # 
       plt.close("all")
       fig1,  ax1 = plt.subplots()
       ax1.plot(Turn/1e9, L_Tau, "b.", label='Fill data', markersize=8)
       ax1.plot(Tau_fit/1e9, L_fit,"y.", label='Fit interval', markersize=8) #The Selected data point       
       ax1.plot(X/1e9,Y, "r.", label='Extrapolated Interval Mod-2', markersize=4)
       ax1.plot(X/1e9,YY, "m.", label='Extrapolated Interval Double-Exp', markersize=4) #ADDED
       ax1.set_xlabel('Normlised Time in Turns 'r'$\tau$(10$^9$)')
       ax1.set_ylabel("'${L}/{L_i}$'") 
       ax1.set_title('The Extrapolated Luminosity evolution {}'.format(text))
       plt.legend(loc='best')
       plt.savefig('Extrapolation/20{}/{}_Extrapolation_mod2_Exp.pdf'.format(str(year),text))  
        #plt.show()"""    

    ########################################################################################################################################################### 


       with open('MOD2/Mod2SmoothCoeff/{}_mod2_FitCoeff{}.txt'.format(str(tmp),str(year)), 'w') as f:
          f.write('')
          f.close()   

       with open('MOD2/Mod2SmoothCoeff/{}_mod2_Rajd_FitCoeff{}.txt'.format(str(tmp),str(year)), 'w') as f:
          f.write('')
          f.close()

       with open('Double/DoubleSmoothCoeff/{}_Double_FitCoeff{}.txt'.format(str(tmp),str(year)), 'w') as f:
          f.write('')
          f.close()   

       with open('Double/DoubleSmoothCoeff/{}_Double_Rajd_FitCoeff{}.txt'.format(str(tmp),str(year)), 'w') as f:
          f.write('')
          f.close()

       for i in range(len(FillNumber)):
           text = str(int(FillNumber[i])) 
       
           with open('MOD2/Mod2SmoothCoeff/{}_mod2_FitCoeff{}.txt'.format(str(tmp),str(year)), 'a') as f:
               f.write(text)
               f.write(' ')
               f.write(str(eps_ni))
               f.write(' ')
               f.write(str(B))
               f.write(' ')
               f.write(str(k))
               f.write('\n')

           with open('MOD2/Mod2SmoothCoeff/{}_mod2_Rajd_FitCoeff{}.txt'.format(str(tmp),str(year)), 'a') as f:
               f.write(text)
               f.write(' ')
               f.write(str(model2_r_squared))
               f.write('\n')
 

           with open('Double/DoubleSmoothCoeff/{}_Double_FitCoeff{}.txt'.format(str(tmp),str(year)), 'a') as f:
                f.write(text)
                f.write(' ')
                f.write(str(a))
                f.write(' ')
                f.write(str(b))
                f.write(' ')
                f.write(str(c))
                f.write(' ')
                f.write(str(d))
                f.write('\n')
  
           with open('Double/DoubleSmoothCoeff/{}_Double_Rajd_FitCoeff{}.txt'.format(str(tmp),str(year)), 'a') as f:
                f.write(text)
                f.write(' ')
                f.write(str(double_gaussian_r_squared))
                f.write('\n')
 

    #### if exttapolation not needed ########################################################################################################################
    elif int(Turn[-1])<=limit:       
       k2=0 
       Lumi_evol=[]
       Time=[]
       L=L_norm
       Tau_t=Tau
       T=Tau_t
       for m in Turn:
           if int(Turn[-1])>=m:
              Lumi_evol.append(L_norm[k2])
              Time.append(Turn[k2])
              k2=k2+1
            
       #Lumi_evol = np.array(Lumi_evol).flatten('F')
       #Time = np.array(Time).flatten('F')
       Lumi_evol = np.array(Lumi_evol)
       Time = np.array(Time)

       Tau_norm = (Time)/(np.amax(Time)) 
       #performing fit of last segments of data using: DA Model-2
       p0 = (1e-10, 10 , 1)    
       popt, pcov = curve_fit(L_model, Time, Lumi_evol,p0,bounds=(0, [1e-9, 100., 10]),maxfev=500000)
   
       eps_ni, B, k = popt
       Y = L_model(Time, eps_ni, B, k)


       #performing fit of last segments of data Using: Double-Exponential
       popt, pcov = curve_fit(fit, Tau_norm, Lumi_evol,bounds=(0, [1,50,1,50]),maxfev=5000)

       a, b, c,d = popt
       YY = fit(Time, a, b/(np.max(Time)), c,d/(np.max(Time)))  

       ########################## Here what should be plotted in case the extrapolation not needed #########################

       ax.plot(Turn/1e9, L_Tau, "b.", label='Fill data', markersize=8) 
       #ax.plot(Tau_t/1e9, L_norm, "g.", label='Selected data', markersize=8)
       ax.plot(Time/1e9, Lumi_evol, "y.", label='Luminosity Evolution', markersize=2)
       ax.plot(Time/1e9, Y, "r-", label='MOD-2', markersize=4)
       ax.plot(Time/1e9, YY, color='plum', label='Double-Exponential', markersize=4)
       ax.set_xlabel('Normlised Time in Turns 'r'$\tau$(10$^9$)')
       ax.set_ylabel("'${L}/{L_i}$'")
       ax.set_title('Normalized Luminosity evolution for optimal fill {}'.format(text))
       plt.legend(loc='best')
       plt.savefig('No_Extrapolation/20{}/{}_Opt_Lumi_Evol_mod2.pdf'.format(str(year),text))
       #plt.show()

       #evaluating the fill integral luminosity    
    L_integrate_opt = integrate.simps(Lumi_evol*1e30, Time)
    #L_integrate_opt_f = (L_max*L_integrate_opt)/(N_i*Eps*f_rev*1e9)
    #L_int_opt.append(L_integrate_opt_f)
    L_int_opt.append(L_integrate_opt)

#defining the dataframes 
df=pd.DataFrame(L_int_opt, columns=['Opt. Measured Int. Lum.']) 

#evaluating the total luminosities   
Lum_tot=[np.sum(L_int_opt)]
L_summary_tot=[np.sum(L_int_summary)]

dftot=pd.DataFrame(Lum_tot, columns=['Opt. Measured tot. Lum.']) 

#write dataframes on an excel file
with pd.ExcelWriter('Optimized Measured Luminosity 20{}.xlsx'.format(str(year))) as writer:
        df.to_excel(writer, sheet_name='20{} Integrated Luminosity'.format(str(year)))
        dftot.to_excel(writer, sheet_name='20{} Total Luminosity'.format(str(year)))

#L_int_opt_scaled = [x/1e30 for x in L_int_opt]  # divide each element by 1e30
Lopt = [x/1e39 for x in L_int_opt]
Lmes_tot = [y/1e9 for y in L_summary_tot]
L_tot = [z/1e9 for z in Lum_tot]

#comparison between Actual and Optimised integrated luminosity
fig, ax1= plt.subplots()
ax1.hist(L_int_summary/1e9,  facecolor="steelblue", alpha=0.4, density=True, label="Measured Integrated Luminosities" )
#ax1.hist(L_int_opt/1e30, histtype='step', density=True, color='red', label="Optimised Integrated Luminosity")
ax1.hist(Lopt, histtype='step', density=True, color='red', label="Optimised Integrated Luminosity")
#plt.plot([],[], "k.", label=r'Initial $L_{\mathrm{tot}}$='+'{:.2f}'.format((Lmes_tot))+r' [$\mathrm{fb}^{-1}$]')
#plt.plot([],[], "k.", label=r'Optimized $L_{\mathrm{tot}}$='+'{:.2f}'.format((L_tot))+r' [$\mathrm{fb}^{-1}$]')
ax1.set_xlabel(r'Integrated Luminosity [$\mathrm{fb}^{-1}$]')
ax1.set_ylabel('Normalised Frequencies')
ax1.set_title('20{}'.format(year))
plt.legend(loc='upper left')
plt.savefig('20{}_lumi.pdf'.format(year))





#defining the stop time of the program      
stop=t.time()
#evaluating the run time 
runtime_seq=stop-start
print('The runtime is:', runtime_seq, '[s]')      
