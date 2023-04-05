####################################################################################################################################
#How TO Use:
#Models to be used with lmfit function.
####################################################################################################################################

import numpy as np
from lmfit import Model


#Power Laws
def N_PowLaw_Model():
    """Defines the fitting model of the normalized Power Law.
    
    Returns
    -------
    NPl_mod: Model"""
    
    def Normalized_PowerLaw (x, amp, Offset, Exponent):
              return amp/((x-Offset)**Exponent) 
    
    NPl_mod = Model(Normalized_PowerLaw) 
    return NPl_mod  


#Exponential Laws
def N_ExpLaw_Model():
    """Defines the fitting model of of the normalized Exponential Law.
    
    Returns
    -------
    NEl_mod: Model"""
    
    def Normalized_ExponentialLaw (x, lam, off):
              return (lam*np.exp(-lam*(x-off))) 
    
    NEl_mod = Model(Normalized_ExponentialLaw) 
    return NEl_mod  



#Truncated Power Laws
def N_TruncLaw_Model():
    """Defines the fitting model of of the normalized Truncated Power Law.
    
    Returns
    -------
    NNTpl_mod: Model"""
    
    def Normalized_TruncatedPowerLaw (x, off, amp, lam, n):
        return (amp/((x-off)**(n)))*(np.exp(-lam*(x)))
    NTpl_mod = Model(Normalized_TruncatedPowerLaw) 
    return NTpl_mod 

