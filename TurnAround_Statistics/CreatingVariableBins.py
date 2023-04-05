#####################################################################################################################################
#Creatig Variable Bins--- How TO Use:
#####################################################################################################################################
#import CreatingVariableBins as cvb
#bi16, bi17, bi18, biT, biTA, biTB, biTC = cvb.CreateBins(array16, array17, array18, array_tot, array_totA, array_totB, array_totC)
#####################################################################################################################################

import numpy as np

def CreateBins(array16, array17, array18, array_tot, array_totA, array_totB, array_totC):
     """Defines the Variable Binning and returns the correct bin arrays.
    
     Parameters
     ----------
     array16, array17, array18, array_tot, array_totA, array_totB, array_totC: arrays
    
    
     Returns
     -------
     bi16, bi17, bi18, biT, biTA, biTB, biTC: ndarrays"""
    
     def bin_array1(array):
        bin_lim = array[0]
        dbin = 0.6 #### 0.6
        fine = np.max(array)
   
        A = [bin_lim] 
        o=0
        while bin_lim <= fine:
             i=0
             o+=1
             for a in array:
                 if a >= bin_lim and a < (bin_lim + dbin):
                         i = i+1 
                
             if o==500:
                break
            
             if i<8: ###### 8
                 dbin = dbin + 0.2 
             else:
                 bin_lim = bin_lim + dbin
                 A.append(bin_lim) 
        A.append(fine) 
        bi = np.array(A, dtype=float)
        return bi 

     bi16 = bin_array1(array16)
     bi17 = bin_array1(array17)
     bi18 = bin_array1(array18) 

     #total dataset bins
     biT = bin_array1(array_tot)

     #partial datasets bins
     biTA = bin_array1(array_totA)
     biTB = bin_array1(array_totB) 
     biTC = bin_array1(array_totC)
     return (bi16, bi17, bi18, biT, biTA, biTB, biTC)