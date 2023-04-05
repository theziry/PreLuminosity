import numpy as np
import pandas as pd

def Create_DataSet():
     """Reads the Excel file and creates the corret datasets.    
    Returns
    -------
    data1, data2, data3: DataFrame"""

     #Importing datas from the excel file
     data = pd.read_excel(r'TurnAroundData.xlsx')

     df1 = pd.DataFrame(data, columns=['sample16'])
     df2 = pd.DataFrame(data, columns=['sample17'])
     df3 = pd.DataFrame(data, columns=['sample18'])

     #Setting the correct different data sets
     data1 = df1.dropna()#loc[:156]
     data2 = df2.dropna()#loc[:194]
     data3 = df3.dropna()#loc[:217]

     return data1, data2, data3

def DataToLists(data1, data2, data3):
     """Transfroms Pandas Dataframes into Lists.
    
    Parameters
    ----------
    data1: dataframe
    data2: dataframe
    data3: dataframe
    
    
    Returns
    -------
    data_16, data_17, data_18: lists"""
     
     #Converting data into python lists
     data_16 = data1['sample16'].values.tolist()
     data_17 = data2['sample17'].values.tolist()
     data_18 = data3['sample18'].values.tolist()
     
     return data_16, data_17, data_18
 
def FromListsToArrays(data_16, data_17, data_18):
     """Transfroms Lists into arrays.
    
    Parameters
    ----------
    data_16: list
    data_17: list
    data_18: list
    
    
    Returns
    -------
    array16, array17, array18: arrays"""
     
     #to array
     array16 = np.array(data_16)
     array17 = np.array(data_17)
     array18 = np.array(data_18)
     
     return array16, array17, array18
 
def TotalDataSet(data_16, data_17, data_18):
     """Creates the total dataset and transforms the inital list into a dataframe and an array.
    
    Parameters
    ----------
    data_16: list
    data_17: list
    data_18: list
    
    
    Returns
    -------
    dataTot: dataframe
    array_tot: array"""
     
     #Summing lists for the total dataset 
     data_tot = data_16 + data_17 + data_18
     dataTot = pd.DataFrame(data_tot)
     array_tot = dataTot.to_numpy()
     
     return data_tot, dataTot, array_tot
 
def PartialDataSets(data_16, data_17, data_18):
      """Creates the partial dataseta and transforms the inital lists into dataframea and arrays.
    
    Parameters
    ----------
    data_16: list
    data_17: list
    data_18: list
    
    
    Returns
    -------
    data_tot_A, data_tot_B, data_tot_C: dataframes
    array_totA, array_totB, array_totC: arrays"""
      
      #Summing lists for the partial datasets 
      data_tot_A = data_16 + data_17
      data_tot_B = data_17 + data_18
      data_tot_C = data_16 + data_18
      array_totA = np.array(data_tot_A)
      array_totB = np.array(data_tot_B)
      array_totC = np.array(data_tot_C)
      
      return data_tot_A, data_tot_B, data_tot_C, array_totA, array_totB, array_totC
 
def Data():
     """Generate the whole set of data sample needed.

     Returns:
         data_16, data_17, data_18: list
         array16, array17, array18: array
     """
     #Importing datas from the excel file
     data = pd.read_excel(r'TurnAroundData.xlsx')

     df1 = pd.DataFrame(data, columns=['sample16'])
     df2 = pd.DataFrame(data, columns=['sample17'])
     df3 = pd.DataFrame(data, columns=['sample18'])

     #Setting the correct different data sets
     data1 = df1.dropna()#loc[:156]
     data2 = df2.dropna()#loc[:194]
     data3 = df3.dropna()#loc[:217]
     
     #Converting data into python lists
     data_16 = data1['sample16'].values.tolist()
     data_17 = data2['sample17'].values.tolist()
     data_18 = data3['sample18'].values.tolist()
     
     #to array
     array16 = np.array(data_16)
     array17 = np.array(data_17)
     array18 = np.array(data_18)
     
     return data_16, data_17, data_18, array16, array17, array18
     
def loadFill():
     """Reads the Excel file and creates the corret dataset lists.    
     Returns
     -------
     data_ta16, data_tf16, data_ta17, data_tf17, data_ta18, data_tf18 : list"""

     #Importing datas from the excel file
     data = pd.read_excel(r'FillData.xlsx')

     df1 = pd.DataFrame(data, columns=['ta16'])
     df2 = pd.DataFrame(data, columns=['tf16'])
     df3 = pd.DataFrame(data, columns=['ta17'])
     df4 = pd.DataFrame(data, columns=['tf17'])
     df5 = pd.DataFrame(data, columns=['ta18'])
     df6 = pd.DataFrame(data, columns=['tf18'])
     

     #Setting the correct different data sets
     data1 = df1.dropna() #df1.loc[:44]
     data2 = df2.dropna()#df2.loc[:44]
     data3 = df3.dropna()#loc[:83]
     data4 = df4.dropna()#loc[:83]
     data5 = df5.dropna()#loc[:94]
     data6 = df6.dropna()#loc[:94]
     
     #Converting data into python lists
     data_ta16 = np.array(data1['ta16'].values.tolist())
     data_tf16 = np.array(data2['tf16'].values.tolist())
     data_ta17 = np.array(data3['ta17'].values.tolist())
     data_tf17 = np.array(data4['tf17'].values.tolist())
     data_ta18 = np.array(data5['ta18'].values.tolist())
     data_tf18 = np.array(data6['tf18'].values.tolist())
     
     return data_ta16, data_tf16, data_ta17,\
          data_tf17, data_ta18, data_tf18 

def Data_sec(array16, data_ta16, data_tf16, array17, data_ta17, data_tf17, array18, data_ta18, data_tf18):
     """Transform the data from seconds to hours.

     Args:
         array16, array17, array18, data_ta16, data_tf16, data_ta17, data_tf17, data_ta18, data_tf18: array

     Returns:
         data_16_sec, data_ta16_sec, data_tf16_sec, data_17_sec, data_ta17_sec, data_tf17_sec, data_18_sec, data_ta18_sec, data_tf18_sec: array
     """
     data_16_sec = array16*3600
     data_ta16_sec = data_ta16*3600 
     data_tf16_sec = data_tf16*3600  
     data_17_sec = array17*3600
     data_ta17_sec = data_ta17*3600 
     data_tf17_sec = data_tf17*3600
     data_18_sec = array18*3600
     data_ta18_sec = data_ta18*3600 
     data_tf18_sec = data_tf18*3600
     return data_16_sec, data_ta16_sec, data_tf16_sec, data_17_sec, data_ta17_sec, data_tf17_sec, data_18_sec, data_ta18_sec, data_tf18_sec
     
def FillNumber():
     """Creates the arrays that contain the number of fills' list.

     Returns:
         FillNumber16, FillNumber17, FillNumber18: array
     """
     data = pd.read_excel(r'FillData.xlsx')

     df1 = pd.DataFrame(data, columns=['NrFill_2016'])
     df2 = pd.DataFrame(data, columns=['NrFill_2017'])
     df3 = pd.DataFrame(data, columns=['NrFill_2018'])
     
     
     data1 = df1.dropna()#df1.loc[:44]
     data2 = df2.dropna()#loc[:83]
     data3 = df3.dropna()#loc[:94]
     
     NrF_16 = data1['NrFill_2016'].values.tolist()
     NrF_17 = data2['NrFill_2017'].values.tolist()
     NrF_18 = data3['NrFill_2018'].values.tolist()
     
     
     FillNumber16 = np.array(NrF_16)
     FillNumber17 = np.array(NrF_17)
     FillNumber18 = np.array(NrF_18)
     return FillNumber16, FillNumber17, FillNumber18

def MeasuredLuminosity():
     """Evaluates the measured luminosity from ATLAS data.

     Returns:
         L_mes16, L_mes17, L_mes18: array
     """
     
     FillNumber16, FillNumber17, FillNumber18 = FillNumber()
     L_mes16=[]
     for i in FillNumber16:
          text = str(int(i))
          f=open('ATLAS/ATLAS_summary_2016/{}_summary_ATLAS.txt'.format(text),"r")
          lines=f.readlines()
          for x in lines:
             result= float(x.split(' ')[3])
    
          L_mes16.append(result)   
     f.close()
     L_mes16=np.array(L_mes16)/1e9 
     
     L_mes17=[]
     for i in FillNumber17:
          text = str(int(i))
          f=open('ATLAS/ATLAS_summary_2017/{}_summary_ATLAS.txt'.format(text),"r")
          lines=f.readlines()
          for x in lines:
             result= float(x.split(' ')[3])
    
          L_mes17.append(result)   
     f.close()
     L_mes17=np.array(L_mes17)/1e9 
     
     L_mes18=[]
     for i in FillNumber18:
          text = str(int(i))
          f=open('ATLAS/ATLAS_summary_2018/{}_summary_ATLAS.txt'.format(text),"r")
          lines=f.readlines()
          for x in lines:
             result= float(x.split(' ')[3])
    
          L_mes18.append(result)   
     f.close()
     L_mes18=np.array(L_mes18)/1e9 
     return L_mes16, L_mes17, L_mes18
                       