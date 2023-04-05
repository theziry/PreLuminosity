import numpy as np
import LoadData as ld
import matplotlib.pyplot as plt

# Selecting Current Year
years = [16, 17, 18]

#loading fill number
FillNumber16, FillNumber17, FillNumber18 = ld.FillNumber()

# create empty arrays to store fill numbers and initial luminosities
fill_nums = []
initial_lums = []
max_lums = []
max_time = []

# DEFINE THE FUNCTION FOR DATA SELECTION IN TIME SCALE

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


# loop over each year and fill number and compute initial luminosity
for year in years:
    if year==16:
        #FillNumber=ld.FillNumber()[0]
        FillNumber=FillNumber16
    elif year==17:
        #FillNumber=ld.FillNumber()[1]
        FillNumber=FillNumber17
        FillNumber_Prev=FillNumber16
        previous_year=16
    elif year==18:
        #FillNumber=ld.FillNumber()[2]
        FillNumber=FillNumber18
        FillNumber_Prev=FillNumber17
        previous_year=17 
    
    with open('SELECTION/20{}_InitialBeam_Fill.txt'.format(str(year)), 'w') as f:
        f.write('')
        f.close() 
    
    with open('SELECTION/20{}_Fill_to_Check.txt'.format(str(year)), 'w') as f:
        f.write('')
        f.close() 
    
    with open('SELECTION/20{}_Length_Fill.txt'.format(str(year)), 'w') as f:
        f.write('')
        f.close() 
               
        
    for i in range(len(FillNumber)):
        text = str(int(FillNumber[i])) #number of current fill
        T_fit_real,L_fit,L_evol, Times= Cut_Fit(year, text)
        
        Time = T_fit_real
        
        L_i = L_fit[0]
        L_max = max(L_fit)
        T_max = max(Time)/3600           
            
        if year==16:
            if L_max <= 7500:
              print("FillNumber: ", FillNumber[i], " The peak to remove L_max: ", L_max)
              #print("FillNumber: ", FillNumber[i], " L_i: ", L_i)
              with open('SELECTION/20{}_InitialBeam_Fill.txt'.format(str(year)), 'a') as f:
                   f.write(str(FillNumber[i]))
                   f.write(' ')
                   f.write(str(L_max))
                   f.write('\n')
        elif year==17:
            if L_max <= 9500:
               print("FillNumber: ", FillNumber[i], " The peak to remove L_max: ", L_max)
               #print("FillNumber: ", FillNumber[i], " L_i: ", L_i)
               with open('SELECTION/20{}_InitialBeam_Fill.txt'.format(str(year)), 'a') as f:
                   f.write(str(FillNumber[i]))
                   f.write(' ')
                   f.write(str(L_max))
                   f.write('\n')
        elif year==18:
            if L_max <= 10000:
               print("FillNumber: ", FillNumber[i], " The peak to remove L_max: ", L_max)
               #print("FillNumber: ", FillNumber[i], " L_i: ", L_i)
               with open('SELECTION/20{}_InitialBeam_Fill.txt'.format(str(year)), 'a') as f:
                   f.write(str(FillNumber[i]))
                   f.write(' ')
                   f.write(str(L_max))
                   f.write('\n')
                   

        if L_i < 7000:
           print("FillNumber: ", FillNumber[i], " L_i: ", L_i)
           #print("FillNumber: ", FillNumber[i], " L_i: ", L_i)
           with open('SELECTION/20{}_Fill_to_Check.txt'.format(str(year)), 'a') as f:
              f.write(str(FillNumber[i]))
              f.write(' ')
              f.write(str(L_i))
              f.write(' ')
              f.write(str(L_max))
              f.write('\n')

        if T_max < 2:
            print("FillNumber: ", FillNumber[i], " Fill Length is: ", T_max,"h")
            with open('SELECTION/20{}_Length_Fill.txt'.format(str(year)), 'a') as f:
              f.write(str(FillNumber[i]))
              f.write(' ')
              f.write(str(T_max))
              f.write('\n')

        
        fill_nums.append(FillNumber[i])
        initial_lums.append(L_i)
        max_lums.append(L_max)
        max_time.append(T_max)
        peak_l = []
        for l in max_lums:
            peak_l.append(l/1e4)
        peak_l = np.array(peak_l)
        
# plot the initial luminosity for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], max_lums[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], max_lums[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], max_lums[len(FillNumber18):], 'g.', label='2018')
plt.xlabel('FillNumber')
plt.ylabel('Initial Luminosity ($10^{30} cm^{-2}s^{-1}$)')
plt.legend()
plt.savefig('SELECTION/INITIAL_BEAM_Luminosity_Fill.pdf')
plt.show()

# plot the Peak luminosity for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], peak_l[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], peak_l[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], peak_l[len(FillNumber18):], 'g.', label='2018')
plt.xlabel('FillNumber')
plt.ylabel('Peak Luminosity $L$ ($10^{34} cm^{-2}s^{-1}$)')
plt.legend()
plt.savefig('SELECTION/PEAK_Luminosity_Fill.pdf')
plt.show()

# plot the initial and max luminosity for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], initial_lums[:len(FillNumber16)], 'k.')
plt.plot(fill_nums[:len(FillNumber16)], max_lums[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], initial_lums[len(FillNumber16):len(FillNumber17)], 'k.')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], max_lums[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], initial_lums[len(FillNumber18):], 'k.')
plt.plot(fill_nums[len(FillNumber18):], max_lums[len(FillNumber18):], 'g.', label='2018')
#plt.axhline(8000, color='magenta', linestyle='dashed', linewidth=2, label='8000 ($10^{30} cm^{-2}s^{-1}$)')
plt.xlabel('FillNumber')
plt.ylabel('Initial Luminosity ($10^{30} cm^{-2}s^{-1}$)')
plt.legend()
plt.savefig('SELECTION/Max_Initial_BEAM_LuminosityFill.pdf')
plt.show()


# plot the Fill length for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], max_lums[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], max_lums[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], max_lums[len(FillNumber18):],  'g.', label='2018')
plt.xlabel('FillNumber')
plt.ylabel('Time [h]')
plt.legend()
plt.savefig('SELECTION/Fill_Lenght_LuminosityFill.pdf')
plt.show()


# plot the Cutted Initial/Max luminosity for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], max_lums[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], max_lums[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], max_lums[len(FillNumber18):], 'g.', label='2018')
# Plot for 2016 data
y16_line = np.full_like(fill_nums[:len(FillNumber16)],7500)
plt.plot(fill_nums[:len(FillNumber16)], y16_line, color='pink',linestyle='dashed', linewidth=1.5, label='0.75($10^{34} cm^{-2}s^{-1}$)')
# Plot for 2017 data
y17_line = np.full_like(fill_nums[len(FillNumber16):len(FillNumber17)],9500)
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], y17_line, color='tomato',linestyle='dashed', linewidth=1.5, label='0.95($10^{34} cm^{-2}s^{-1}$)')
#plt.axhline(y=9500, xmin=FillNumber17[0], xmax=FillNumber17[-1], color='yellow', linestyle='dashed', linewidth=2, label='0.95($10^{34} cm^{-2}s^{-1}$)')
# Plot for 2018 data
y18_line = np.full_like(fill_nums[len(FillNumber18):],10000)
plt.plot(fill_nums[len(FillNumber18):], y18_line, color='magenta',linestyle='dashed', linewidth=1.5, label='1($10^{34} cm^{-2}s^{-1}$)')
#plt.axhline(y=10000, xmin=FillNumber18[0], xmax=FillNumber18[-1], color='pink', linestyle='dashed', linewidth=2, label='1($10^{34} cm^{-2}s^{-1}$)')

plt.xlabel('FillNumber')
plt.ylabel('Initial Luminosity ($10^{30} cm^{-2}s^{-1}$)')
#plt.legend()
plt.legend(loc='best')
plt.savefig('SELECTION/Cut_Initial_BEAM_LuminosityFill.pdf')
plt.show()


# plot the Fill length for each fill number over three years
plt.plot(fill_nums[:len(FillNumber16)], max_lums[:len(FillNumber16)], 'r.', label='2016')
plt.plot(fill_nums[len(FillNumber16):len(FillNumber17)], max_lums[len(FillNumber16):len(FillNumber17)], 'b.', label='2017')
plt.plot(fill_nums[len(FillNumber18):], max_lums[len(FillNumber18):], 'g.', label='2018')
plt.axhline(2, color='magenta', linestyle='dashed', linewidth=1.5, label='2 [h]')
plt.xlabel('FillNumber')
plt.ylabel('Time [h]')
plt.legend()
plt.savefig('SELECTION/Cut_Fill_Lenght_LuminosityFill.pdf')
plt.show()








