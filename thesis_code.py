import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency, pearsonr
import statsmodels.api as sm
from datetime import datetime
from zepid import load_sample_data, spline
from zepid.causal.gformula import TimeFixedGFormula
from zepid.causal.snm import GEstimationSNM

#Datasets used in the thesis
Curry0910 = pd.read_csv('Curry0910.csv') #Steph Curry box score stats 2009-2010
Curry1011 = pd.read_csv('Curry1011.csv') #Steph Curry box score stats 2010-2011
Curry1213 = pd.read_csv('Curry1213.csv') #Steph Curry box score stats 2012-2013
Curry1314 = pd.read_csv('Curry1314.csv') #Steph Curry box score stats 2013-2014
Curry1415 = pd.read_csv('Curry1415.csv') #Steph Curry box score stats 2014-2015
Curry1516 = pd.read_csv('Curry1516.csv') #Steph Curry box score stats 2015-2016
Curry1617 = pd.read_csv('Curry1617.csv') #Steph Curry box score stats 2016-2017
Curry1718 = pd.read_csv('Curry1718.csv') #Steph Curry box score stats 2017-2018
CurryTotal = pd.read_csv('CurryTotal.csv') #Steph Curry box score stats 2009-2018
playoff1213 = pd.read_csv('playoff1213.csv') #Steph Curry box score stats for playoffs 2012-2013
playoff1314 = pd.read_csv('playoff1314.csv') #Steph Curry box score stats for playoffs 2013-2014
playoff1415 = pd.read_csv('playoff1415.csv') #Steph Curry box score stats for playoffs 2014-2015
playoff1516 = pd.read_csv('playoff1516.csv') #Steph Curry box score stats for playoffs 2015-2016
playoff1617 = pd.read_csv('playoff1617.csv') #Steph Curry box score stats for playoffs 2016-2017
playoff1718 = pd.read_csv('playoff1718.csv') #Steph Curry box score stats for playoffs 2017-2018


# Defining the game score metric (Hollinger 2002) for a player using weights
def GS(p, fgm, fga, fta, ftm, r, stl, ast, blk, pf, to):
    return p + 0.4*fgm - 0.7*fga - 0.4*(fta-ftm) + r + stl + 0.7*ast + 0.7*blk - 0.4*pf -to


#Turn the dates from string to integers in order to substruct them
dates1718 = []
datesTotal = []
dates0910 = []
dates1415 = []
dates1011 = []
dates1213 = []
dates1314 = []
dates1516 = []
dates1617 = []

for i in range(len(CurryTotal["Dates"])):
    datesTotal.append(datetime.strptime(CurryTotal['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1718["Dates"])):
    dates1718.append(datetime.strptime(Curry1718['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry0910["Dates"])):
    dates0910.append(datetime.strptime(Curry0910['Dates'][i], "%m/%d/%Y"))   

for i in range(len(Curry1415["Dates"])):
    dates1415.append(datetime.strptime(Curry1415['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1011["Dates"])):
    dates1011.append(datetime.strptime(Curry1011['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1213["Dates"])):
    dates1213.append(datetime.strptime(Curry1213['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1314["Dates"])):
    dates1314.append(datetime.strptime(Curry1314['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1516["Dates"])):
    dates1516.append(datetime.strptime(Curry1516['Dates'][i], "%m/%d/%Y"))

for i in range(len(Curry1617["Dates"])):
    dates1617.append(datetime.strptime(Curry1617['Dates'][i], "%m/%d/%Y"))


datesTotal_1 = pd.to_datetime(CurryTotal['Dates'], format='%m/%d/%Y')
reference_dateTotal = pd.Timestamp(datesTotal[0])

dates1718_1 = pd.to_datetime(Curry1718['Dates'], format='%m/%d/%Y')
reference_date1718 = pd.Timestamp(dates1718[0])

dates0910_1 = pd.to_datetime(Curry0910['Dates'], format='%m/%d/%Y')
reference_date0910 = pd.Timestamp(dates0910[0])

dates1415_1 = pd.to_datetime(Curry1415['Dates'], format='%m/%d/%Y')
reference_date1415 = pd.Timestamp(dates1415[0])

dates1011_1 = pd.to_datetime(Curry1011['Dates'], format='%m/%d/%Y')
reference_date1011 = pd.Timestamp(dates1011[0])

dates1213_1 = pd.to_datetime(Curry1213['Dates'], format='%m/%d/%Y')
reference_date1213 = pd.Timestamp(dates1213[0])

dates1314_1 = pd.to_datetime(Curry1314['Dates'], format='%m/%d/%Y')
reference_date1314 = pd.Timestamp(dates1314[0])

dates1516_1 = pd.to_datetime(Curry1516['Dates'], format='%m/%d/%Y')
reference_date1516 = pd.Timestamp(dates1516[0])

dates1617_1 = pd.to_datetime(Curry1617['Dates'], format='%m/%d/%Y')
reference_date1617 = pd.Timestamp(dates1617[0])

# Calculate the number of days since the reference date and convert to integer
datesTotal_int = (datesTotal_1 - reference_dateTotal).dt.days
dates1718_int = (dates1718_1 - reference_date1718).dt.days
dates0910_int = (dates0910_1 - reference_date0910).dt.days
dates1415_int = (dates1415_1 - reference_date1415).dt.days
dates1011_int = (dates1011_1 - reference_date1011).dt.days
dates1213_int = (dates1213_1 - reference_date1213).dt.days
dates1314_int = (dates1314_1 - reference_date1314).dt.days
dates1516_int = (dates1516_1 - reference_date1516).dt.days
dates1617_int = (dates1617_1 - reference_date1617).dt.days



##### ANALYSIS FOR 2017-2018 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1718 = []
Result_1718 = [] #The result of the game (1=W, 0=L)

for i in range(len(Curry1718['Result'])):
    if Curry1718['Result'][i] == 'W':
        Result_1718.append(1)
    else:
        Result_1718.append(0)

for i in range(len(Curry1718['PTS'])):
    GS_L_1718.append(GS(Curry1718['PTS'][i], Curry1718['Successful Shots'][i], Curry1718['Total Shots'][i], Curry1718['Total FT'][i], Curry1718['Successful FT'][i], Curry1718['REB'][i], Curry1718['STL'][i], Curry1718['AST'][i], Curry1718['BLK'][i], Curry1718['PF'][i], Curry1718['TO'][i]))



#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1718 = [0]
for i in range(len(dates1718_int)-1):
    if dates1718_int[i+1] - dates1718_int[i] == 1:
        Treatment_1718.append(1)
    else:
        Treatment_1718.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1718 = [0,0,0,0]
for i in range(len(GS_L_1718)-4):
    if np.mean([GS_L_1718[i], GS_L_1718[i+1], GS_L_1718[i+2], GS_L_1718[i+3]]) >= 20:
        Third_Treatment_1718.append(1)
    else:
        Third_Treatment_1718.append(0)


#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1718 = [0]
for i in range(1,len(Curry1718['Minutes'])):
    Covariate3_C_1718.append(Curry1718['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1718 = [0]
for i in range(len(Curry1718['Score GS'])-1):
    Covariate4_C_1718.append(Curry1718['Score Opponent'][i] - Curry1718['Score GS'][i])


###### G-ESTIMATION ######

#To g-estimate in question 1
data1718_C = pd.DataFrame({
    'outcome': GS_L_1718,
    'treatment': Treatment_1718,
    'covariate3': Covariate3_C_1718,
    'covariate4': Covariate4_C_1718
})
#To g-estimate in question 1.1
data1718_R = pd.DataFrame({
    'outcome': Result_1718,
    'treatment': Treatment_1718,
    'covariate4': Covariate4_C_1718
})
#To g-estimate in question 2
data1718_S = pd.DataFrame({
    'outcome': Result_1718,
    'treatment': Third_Treatment_1718,
    'covariate4': Covariate4_C_1718
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1718_R, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary()
'''


##### ANALYSIS FOR 2009-2010 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_0910 = []
Result_0910 = []

for i in range(len(Curry0910['Result'])):
    if Curry0910['Result'][i] == 'W':
        Result_0910.append(1)
    else:
        Result_0910.append(0)
       
for i in range(len(Curry0910['PTS'])):
    GS_L_0910.append(GS(Curry0910['PTS'][i], Curry0910['Successful Shots'][i], Curry0910['Total Shots'][i], Curry0910['Total FT'][i], Curry0910['Successful FT'][i], Curry0910['REB'][i], Curry0910['STL'][i], Curry0910['AST'][i], Curry0910['BLK'][i], Curry0910['PF'][i], Curry0910['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_0910 = [0]
for i in range(len(dates0910_int)-1):
    if dates0910_int[i+1] - dates0910_int[i] == 1:
        Treatment_0910.append(1)
    else:
        Treatment_0910.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_0910 = [0,0,0,0]
for i in range(len(GS_L_0910)-4):
    if np.mean([GS_L_0910[i], GS_L_0910[i+1], GS_L_0910[i+2], GS_L_0910[i+3]]) >= 20:
        Third_Treatment_0910.append(1)
    else:
        Third_Treatment_0910.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_0910 = [0]
for i in range(1,len(Curry0910['Minutes'])):
    Covariate3_C_0910.append(Curry0910['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_0910 = [0]
for i in range(len(Curry0910['Score GS'])-1):
    Covariate4_C_0910.append(Curry0910['Score Opponent'][i] - Curry0910['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data0910_C = pd.DataFrame({
    'outcome': GS_L_0910,
    'treatment': Treatment_0910,
    'covariate3': Covariate3_C_0910,
    'covariate4': Covariate4_C_0910
})
#To g-estimate in question 1.1
data0910_R = pd.DataFrame({
    'outcome': Result_0910,
    'treatment': Treatment_0910,
    'covariate4': Covariate4_C_0910
})
#To g-estimate in question 2
data0910_S = pd.DataFrame({
    'outcome': Result_0910,
    'treatment': Third_Treatment_0910,
    'covariate4': Covariate4_C_0910
})


#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data0910_C, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate3 + covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary()
'''


##### ANALYSIS FOR 2014-2015 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1415 = []
Result_1415 = []

for i in range(len(Curry1415['Result'])):
    if Curry1415['Result'][i] == 'W':
        Result_1415.append(1)
    else:
        Result_1415.append(0)

for i in range(len(Curry1415['PTS'])):
    GS_L_1415.append(GS(Curry1415['PTS'][i], Curry1415['Successful Shots'][i], Curry1415['Total Shots'][i], Curry1415['Total FT'][i], Curry1415['Successful FT'][i], Curry1415['REB'][i], Curry1415['STL'][i], Curry1415['AST'][i], Curry1415['BLK'][i], Curry1415['PF'][i], Curry1415['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1415 = [0]
for i in range(len(dates1415_int)-1):
    if dates1415_int[i+1] - dates1415_int[i] == 1:
        Treatment_1415.append(1)
    else:
        Treatment_1415.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1415 = [0,0,0,0]
for i in range(len(GS_L_1415)-4):
    if np.mean([GS_L_1415[i], GS_L_1415[i+1], GS_L_1415[i+2], GS_L_1415[i+3]]) >= 20:
        Third_Treatment_1415.append(1)
    else:
        Third_Treatment_1415.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1415 = [0]
for i in range(1,len(Curry1415['Minutes'])):
    Covariate3_C_1415.append(Curry1415['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1415 = [0]
for i in range(len(Curry1415['Score GS'])-1):
    Covariate4_C_1415.append(Curry1415['Score Opponent'][i] - Curry1415['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1415_C = pd.DataFrame({
    'outcome': GS_L_1415,
    'treatment': Treatment_1415,
    'covariate3': Covariate3_C_1415,
    'covariate4': Covariate4_C_1415
})
#To g-estimate in question 1.1
data1415_R = pd.DataFrame({
    'outcome': Result_1415,
    'treatment': Treatment_1415,
    'covariate4': Covariate4_C_1415
})
#To g-estimate in question 2
data1415_S = pd.DataFrame({
    'outcome': Result_1415,
    'treatment': Third_Treatment_1415,
    'covariate4': Covariate4_C_1415
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1415_S, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary()
'''


##### ANALYSIS FOR 2010-2011 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1011 = []
Result_1011 = []

for i in range(len(Curry1011['Result'])):
    if Curry1011['Result'][i] == 'W':
        Result_1011.append(1)
    else:
        Result_1011.append(0)

for i in range(len(Curry1011['PTS'])):
    GS_L_1011.append(GS(Curry1011['PTS'][i], Curry1011['Successful Shots'][i], Curry1011['Total Shots'][i], Curry1011['Total FT'][i], Curry1011['Successful FT'][i], Curry1011['REB'][i], Curry1011['STL'][i], Curry1011['AST'][i], Curry1011['BLK'][i], Curry1011['PF'][i], Curry1011['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1011 = [0]
for i in range(len(dates1011_int)-1):
    if dates1011_int[i+1] - dates1011_int[i] == 1:
        Treatment_1011.append(1)
    else:
        Treatment_1011.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1011 = [0,0,0,0]
for i in range(len(GS_L_1011)-4):
    if np.mean([GS_L_1011[i], GS_L_1011[i+1], GS_L_1011[i+2], GS_L_1011[i+3]]) >= 20:
        Third_Treatment_1011.append(1)
    else:
        Third_Treatment_1011.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1011 = [0]
for i in range(1,len(Curry1011['Minutes'])):
    Covariate3_C_1011.append(Curry1011['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1011 = [0]
for i in range(len(Curry1011['Score GS'])-1):
    Covariate4_C_1011.append(Curry1011['Score Opponent'][i] - Curry1011['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1011_C = pd.DataFrame({
    'outcome': GS_L_1011,
    'treatment': Treatment_1011,
    'covariate3': Covariate3_C_1011,
    'covariate4': Covariate4_C_1011
})
#To g-estimate in question 1.1
data1011_R = pd.DataFrame({
    'outcome': Result_1011,
    'treatment': Treatment_1011,
    'covariate4': Covariate4_C_1011
})
#To g-estimate in question 2
data1011_S = pd.DataFrame({
    'outcome': Result_1011,
    'treatment': Third_Treatment_1011,
    'covariate4': Covariate4_C_1011
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1011_S, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary() 
'''


##### ANALYSIS FOR 2012-2013 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1213 = []
Result_1213 = []

for i in range(len(Curry1213['Result'])):
    if Curry1213['Result'][i] == 'W':
        Result_1213.append(1)
    else:
        Result_1213.append(0)

for i in range(len(Curry1213['PTS'])):
    GS_L_1213.append(GS(Curry1213['PTS'][i], Curry1213['Successful Shots'][i], Curry1213['Total Shots'][i], Curry1213['Total FT'][i], Curry1213['Successful FT'][i], Curry1213['REB'][i], Curry1213['STL'][i], Curry1213['AST'][i], Curry1213['BLK'][i], Curry1213['PF'][i], Curry1213['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1213 = [0]
for i in range(len(dates1213_int)-1):
    if dates1213_int[i+1] - dates1213_int[i] == 1:
        Treatment_1213.append(1)
    else:
        Treatment_1213.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1213 = [0,0,0,0]
for i in range(len(GS_L_1213)-4):
    if np.mean([GS_L_1213[i], GS_L_1213[i+1], GS_L_1213[i+2], GS_L_1213[i+3]]) >= 20:
        Third_Treatment_1213.append(1)
    else:
        Third_Treatment_1213.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1213 = [0]
for i in range(1,len(Curry1213['Minutes'])):
    Covariate3_C_1213.append(Curry1213['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1213 = [0]
for i in range(len(Curry1213['Score GS'])-1):
    Covariate4_C_1213.append(Curry1213['Score Opponent'][i] - Curry1213['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1213_C = pd.DataFrame({
    'outcome': GS_L_1213,
    'treatment': Treatment_1213,
    'covariate3': Covariate3_C_1213,
    'covariate4': Covariate4_C_1213
})
#To g-estimate in question 1.1
data1213_R = pd.DataFrame({
    'outcome': Result_1213,
    'treatment': Treatment_1213,
    'covariate4': Covariate4_C_1213
})
#To g-estimate in question 2
data1213_S = pd.DataFrame({
    'outcome': Result_1213,
    'treatment': Third_Treatment_1213,
    'covariate4': Covariate4_C_1213
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1213_S, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary()
'''


##### ANALYSIS FOR 2013-2014 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1314 = []
Result_1314 = []

for i in range(len(Curry1314['Result'])):
    if Curry1314['Result'][i] == 'W':
        Result_1314.append(1)
    else:
        Result_1314.append(0)

for i in range(len(Curry1314['PTS'])):
    GS_L_1314.append(GS(Curry1314['PTS'][i], Curry1314['Successful Shots'][i], Curry1314['Total Shots'][i], Curry1314['Total FT'][i], Curry1314['Successful FT'][i], Curry1314['REB'][i], Curry1314['STL'][i], Curry1314['AST'][i], Curry1314['BLK'][i], Curry1314['PF'][i], Curry1314['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1314 = [0]
for i in range(len(dates1314_int)-1):
    if dates1314_int[i+1] - dates1314_int[i] == 1:
        Treatment_1314.append(1)
    else:
        Treatment_1314.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1314 = [0,0,0,0]
for i in range(len(GS_L_1314)-4):
    if np.mean([GS_L_1314[i], GS_L_1314[i+1], GS_L_1314[i+2], GS_L_1314[i+3]]) >= 20:
        Third_Treatment_1314.append(1)
    else:
        Third_Treatment_1314.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1314 = [0]
for i in range(1,len(Curry1314['Minutes'])):
    Covariate3_C_1314.append(Curry1314['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1314 = [0]
for i in range(len(Curry1314['Score GS'])-1):
    Covariate4_C_1314.append(Curry1314['Score Opponent'][i] - Curry1314['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1314_C = pd.DataFrame({
    'outcome': GS_L_1314,
    'treatment': Treatment_1314,
    'covariate3': Covariate3_C_1314,
    'covariate4': Covariate4_C_1314
})
#To g-estimate in question 1.1
data1314_R = pd.DataFrame({
    'outcome': Result_1314,
    'treatment': Treatment_1314,
    'covariate4': Covariate4_C_1314
})
#To g-estimate in question 2
data1314_S = pd.DataFrame({
    'outcome': Result_1314,
    'treatment': Third_Treatment_1314,
    'covariate4': Covariate4_C_1314
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1314_S, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary() 
'''


##### ANALYSIS FOR 2015-2016 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1516 = []
Result_1516 = []

for i in range(len(Curry1516['Result'])):
    if Curry1516['Result'][i] == 'W':
        Result_1516.append(1)
    else:
        Result_1516.append(0)

for i in range(len(Curry1516['PTS'])):
    GS_L_1516.append(GS(Curry1516['PTS'][i], Curry1516['Successful Shots'][i], Curry1516['Total Shots'][i], Curry1516['Total FT'][i], Curry1516['Successful FT'][i], Curry1516['REB'][i], Curry1516['STL'][i], Curry1516['AST'][i], Curry1516['BLK'][i], Curry1516['PF'][i], Curry1516['TO'][i]))


#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1516 = [0]
for i in range(len(dates1516_int)-1):
    if dates1516_int[i+1] - dates1516_int[i] == 1:
        Treatment_1516.append(1)
    else:
        Treatment_1516.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1516 = [0,0,0,0]
for i in range(len(GS_L_1516)-4):
    if np.mean([GS_L_1516[i], GS_L_1516[i+1], GS_L_1516[i+2], GS_L_1516[i+3]]) >= 20:
        Third_Treatment_1516.append(1)
    else:
        Third_Treatment_1516.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1516 = [0]
for i in range(1,len(Curry1516['Minutes'])):
    Covariate3_C_1516.append(Curry1516['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1516 = [0]
for i in range(len(Curry1516['Score GS'])-1):
    Covariate4_C_1516.append(Curry1516['Score Opponent'][i] - Curry1516['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1516_C = pd.DataFrame({
    'outcome': GS_L_1516,
    'treatment': Treatment_1516,
    'covariate3': Covariate3_C_1516,
    'covariate4': Covariate4_C_1516
})
#To g-estimate in question 1.1
data1516_R = pd.DataFrame({
    'outcome': Result_1516,
    'treatment': Treatment_1516,
    'covariate4': Covariate4_C_1516
})
#To g-estimate in question 2
data1516_S = pd.DataFrame({
    'outcome': Result_1516,
    'treatment': Third_Treatment_1516,
    'covariate4': Covariate4_C_1516
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1516_S, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary() 
'''


##### ANALYSIS FOR 2016-2017 SEASON #####
#The outcomes of interest (Y_GS and Y_R)
GS_L_1617 = []
Result_1617 = []

for i in range(len(Curry1617['Result'])):
    if Curry1617['Result'][i] == 'W':
        Result_1617.append(1)
    else:
        Result_1617.append(0)

for i in range(len(Curry1617['PTS'])):
    GS_L_1617.append(GS(Curry1617['PTS'][i], Curry1617['Successful Shots'][i], Curry1617['Total Shots'][i], Curry1617['Total FT'][i], Curry1617['Successful FT'][i], Curry1617['REB'][i], Curry1617['STL'][i], Curry1617['AST'][i], Curry1617['BLK'][i], Curry1617['PF'][i], Curry1617['TO'][i]))

#Defining the treatment (1 if 2 games are in 2 days and 0 otherwise)
Treatment_1617 = [0]
for i in range(len(dates1617_int)-1):
    if dates1617_int[i+1] - dates1617_int[i] == 1:
        Treatment_1617.append(1)
    else:
        Treatment_1617.append(0)

#Defining the treatment (1 if Curry is in form before the game and 0 otherwise)
Third_Treatment_1617 = [0,0,0,0]
for i in range(len(GS_L_1617)-4):
    if np.mean([GS_L_1617[i], GS_L_1617[i+1], GS_L_1617[i+2], GS_L_1617[i+3]]) >= 20:
        Third_Treatment_1617.append(1)
    else:
        Third_Treatment_1617.append(0)

#Defining the covariate3 = number of minutes (continuous), before the treatment
Covariate3_C_1617 = [0]
for i in range(1,len(Curry1617['Minutes'])):
    Covariate3_C_1617.append(Curry1617['Minutes'][i-1])

#Defining the covariate4 = points scored by oponent - points scored by GSW, before the treatment
Covariate4_C_1617 = [0]
for i in range(len(Curry1617['Score GS'])-1):
    Covariate4_C_1617.append(Curry1617['Score Opponent'][i] - Curry1617['Score GS'][i])

###### G-ESTIMATION ######

#To g-estimate in question 1
data1617_C = pd.DataFrame({
    'outcome': GS_L_1617,
    'treatment': Treatment_1617,
    'covariate3': Covariate3_C_1617,
    'covariate4': Covariate4_C_1617
})
#To g-estimate in question 1.1
data1617_R = pd.DataFrame({
    'outcome': Result_1617,
    'treatment': Treatment_1617,
    'covariate4': Covariate4_C_1617
})
#To g-estimate in question 2
data1617_S = pd.DataFrame({
    'outcome': Result_1617,
    'treatment': Third_Treatment_1617,
    'covariate4': Covariate4_C_1617
})

#Summary of the snm model and the estimate of the average causal effect
'''
snm = GEstimationSNM(data1617_C, exposure='treatment', outcome='outcome')
snm.exposure_model('covariate3 + covariate4')
snm.structural_nested_model(model='treatment')
snm.fit()
snm.summary()
'''

###------------------PLAYOFFS------------------###
GS_L_1213_playoff = []
for i in range(len(playoff1213['PTS'])):
    GS_L_1213_playoff.append(GS(playoff1213['PTS'][i], playoff1213['Successful Shots'][i], playoff1213['Total Shots'][i], playoff1213['Total FT'][i], playoff1213['Successful FT'][i], playoff1213['REB'][i], playoff1213['STL'][i], playoff1213['AST'][i], playoff1213['BLK'][i], playoff1213['PF'][i], playoff1213['TO'][i]))


GS_L_1314_playoff = []
for i in range(len(playoff1314['PTS'])):
    GS_L_1314_playoff.append(GS(playoff1314['PTS'][i], playoff1314['Successful Shots'][i], playoff1314['Total Shots'][i], playoff1314['Total FT'][i], playoff1314['Successful FT'][i], playoff1314['REB'][i], playoff1314['STL'][i], playoff1314['AST'][i], playoff1314['BLK'][i], playoff1314['PF'][i], playoff1314['TO'][i]))


GS_L_1415_playoff = []
for i in range(len(playoff1415['PTS'])):
    GS_L_1415_playoff.append(GS(playoff1415['PTS'][i], playoff1415['Successful Shots'][i], playoff1415['Total Shots'][i], playoff1415['Total FT'][i], playoff1415['Successful FT'][i], playoff1415['REB'][i], playoff1415['STL'][i], playoff1415['AST'][i], playoff1415['BLK'][i], playoff1415['PF'][i], playoff1415['TO'][i]))


GS_L_1516_playoff = []
for i in range(len(playoff1516['PTS'])):
    GS_L_1516_playoff.append(GS(playoff1516['PTS'][i], playoff1516['Successful Shots'][i], playoff1516['Total Shots'][i], playoff1516['Total FT'][i], playoff1516['Successful FT'][i], playoff1516['REB'][i], playoff1516['STL'][i], playoff1516['AST'][i], playoff1516['BLK'][i], playoff1516['PF'][i], playoff1516['TO'][i]))


GS_L_1617_playoff = []
for i in range(len(playoff1617['PTS'])):
    GS_L_1617_playoff.append(GS(playoff1617['PTS'][i], playoff1617['Successful Shots'][i], playoff1617['Total Shots'][i], playoff1617['Total FT'][i], playoff1617['Successful FT'][i], playoff1617['REB'][i], playoff1617['STL'][i], playoff1617['AST'][i], playoff1617['BLK'][i], playoff1617['PF'][i], playoff1617['TO'][i]))


GS_L_1718_playoff = []
for i in range(len(playoff1718['PTS'])):
    GS_L_1718_playoff.append(GS(playoff1718['PTS'][i], playoff1718['Successful Shots'][i], playoff1718['Total Shots'][i], playoff1718['Total FT'][i], playoff1718['Successful FT'][i], playoff1718['REB'][i], playoff1718['STL'][i], playoff1718['AST'][i], playoff1718['BLK'][i], playoff1718['PF'][i], playoff1718['TO'][i]))


#################### PLOTS FOR GS.MEANS MONTHLY FOR EACH SEASON ####################

# Add the GS_L_year to the DataFrame
Curry0910['GS_L_0910'] = GS_L_0910
Curry1011['GS_L_1011'] = GS_L_1011
Curry1213['GS_L_1213'] = GS_L_1213
Curry1314['GS_L_1314'] = GS_L_1314
Curry1415['GS_L_1415'] = GS_L_1415
Curry1516['GS_L_1516'] = GS_L_1516
Curry1617['GS_L_1617'] = GS_L_1617
Curry1718['GS_L_1718'] = GS_L_1718

# Ensure Dates column is properly parsed
Curry0910['Dates'] = pd.to_datetime(Curry0910['Dates'])
Curry1011['Dates'] = pd.to_datetime(Curry1011['Dates'])
Curry1213['Dates'] = pd.to_datetime(Curry1213['Dates'])
Curry1314['Dates'] = pd.to_datetime(Curry1314['Dates'])
Curry1415['Dates'] = pd.to_datetime(Curry1415['Dates'])
Curry1516['Dates'] = pd.to_datetime(Curry1516['Dates'])
Curry1617['Dates'] = pd.to_datetime(Curry1617['Dates'])
Curry1718['Dates'] = pd.to_datetime(Curry1718['Dates'])

# Create a custom month label that takes into account the year
Curry0910['Month_Year'] = Curry0910['Dates'].dt.strftime('%m/%Y')
Curry1011['Month_Year'] = Curry1011['Dates'].dt.strftime('%m/%Y')
Curry1213['Month_Year'] = Curry1213['Dates'].dt.strftime('%m/%Y')
Curry1314['Month_Year'] = Curry1314['Dates'].dt.strftime('%m/%Y')
Curry1415['Month_Year'] = Curry1415['Dates'].dt.strftime('%m/%Y')
Curry1516['Month_Year'] = Curry1516['Dates'].dt.strftime('%m/%Y')
Curry1617['Month_Year'] = Curry1617['Dates'].dt.strftime('%m/%Y')
Curry1718['Month_Year'] = Curry1718['Dates'].dt.strftime('%m/%Y')

# Define the correct order of months for the NBA season (starting from October)
correct_order0910 = ['10/2009', '11/2009', '12/2009', '01/2010', '02/2010', '03/2010', '04/2010']
correct_order1011 = ['10/2010', '11/2010', '12/2010', '01/2011', '02/2011', '03/2011', '04/2011']
correct_order1213 = ['10/2012', '11/2012', '12/2012', '01/2013', '02/2013', '03/2013', '04/2013']
correct_order1314 = ['10/2013', '11/2013', '12/2013', '01/2014', '02/2014', '03/2014', '04/2014']
correct_order1415 = ['10/2014', '11/2014', '12/2014', '01/2015', '02/2015', '03/2015', '04/2015']
correct_order1516 = ['10/2015', '11/2015', '12/2015', '01/2016', '02/2016', '03/2016', '04/2016']
correct_order1617 = ['10/2016', '11/2016', '12/2016', '01/2017', '02/2017', '03/2017', '04/2017']
correct_order1718 = ['10/2017', '11/2017', '12/2017', '01/2018', '02/2018', '03/2018', '04/2018']

# Group the data by this custom month label and calculate the mean of GS_L_year for each month
monthly_means_gs0910_series = Curry0910.groupby('Month_Year')['GS_L_0910'].mean()
monthly_means_gs1011_series = Curry1011.groupby('Month_Year')['GS_L_1011'].mean()
monthly_means_gs1213_series = Curry1213.groupby('Month_Year')['GS_L_1213'].mean()
monthly_means_gs1314_series = Curry1314.groupby('Month_Year')['GS_L_1314'].mean()
monthly_means_gs1415_series = Curry1415.groupby('Month_Year')['GS_L_1415'].mean()
monthly_means_gs1516_series = Curry1516.groupby('Month_Year')['GS_L_1516'].mean()
monthly_means_gs1617_series = Curry1617.groupby('Month_Year')['GS_L_1617'].mean()
monthly_means_gs1718_series = Curry1718.groupby('Month_Year')['GS_L_1718'].mean()

# Reindex the series to ensure correct month order
monthly_means_gs0910_series = monthly_means_gs0910_series.reindex(correct_order0910)
monthly_means_gs1011_series = monthly_means_gs1011_series.reindex(correct_order1011)
monthly_means_gs1213_series = monthly_means_gs1213_series.reindex(correct_order1213)
monthly_means_gs1314_series = monthly_means_gs1314_series.reindex(correct_order1314)
monthly_means_gs1415_series = monthly_means_gs1415_series.reindex(correct_order1415)
monthly_means_gs1516_series = monthly_means_gs1516_series.reindex(correct_order1516)
monthly_means_gs1617_series = monthly_means_gs1617_series.reindex(correct_order1617)
monthly_means_gs1718_series = monthly_means_gs1718_series.reindex(correct_order1718)

# Convert the series to a list
monthly_means_gs0910 = monthly_means_gs0910_series.tolist()
monthly_means_gs1011 = monthly_means_gs1011_series.tolist()
monthly_means_gs1213 = monthly_means_gs1213_series.tolist()
monthly_means_gs1314 = monthly_means_gs1314_series.tolist()
monthly_means_gs1415 = monthly_means_gs1415_series.tolist()
monthly_means_gs1516 = monthly_means_gs1516_series.tolist()
monthly_means_gs1617 = monthly_means_gs1617_series.tolist()
monthly_means_gs1718 = monthly_means_gs1718_series.tolist()

# Define the correct order of months for the NBA season (starting from October)
months = ['October', 'November', 'December', 'January', 'February', 'March', 'April']

'''
# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(months, monthly_means_gs1718, marker='o', linestyle='-', color='b')

# Add titles and labels
plt.title("Average Game Score for Curry per Month - 2017/2018 Season")
plt.xlabel("Month")
plt.ylabel("Mean of Game Score")

# Display the plot
plt.grid(True)
plt.show()
'''

#################### CONFIDENCE INTERVALS FOR THE ESTIMATED AVERAGE CAUSAL EFFECT ####################
'''
def bootstrap_confidence_interval(data, treatment, outcome, confounders, n_bootstrap=1000, alpha=0.05):
    """
    Calculate the confidence interval for the average causal effect using bootstrapping.

    Parameters:
    - data: DataFrame containing the data.
    - treatment: string, name of the treatment variable in the DataFrame.
    - outcome: string, name of the outcome variable in the DataFrame.
    - confounders: list of strings, names of the confounders in the DataFrame.
    - n_bootstrap: int, number of bootstrap iterations (default: 1000).
    - alpha: float, significance level for the confidence interval (default: 0.05).
    
    Returns:
    - lower_bound: float, lower bound of the confidence interval.
    - upper_bound: float, upper bound of the confidence interval.
    """
    
    # Placeholder for bootstrap estimates
    bootstrap_ace = []
    
    for i in range(n_bootstrap):
        # Resample the data with replacement
        sample_data = data.sample(frac=1, replace=True)
        
        # Recalculate the ACE using the GEstimationSNM or similar method on the resampled data
        snm = GEstimationSNM(sample_data, exposure=treatment, outcome=outcome)
        snm.exposure_model(' + '.join(confounders))
        snm.structural_nested_model(model='treatment')
        snm.fit()

        # Extract the estimated ACE from the SNM model (assuming 'psi' stores the ACE)
        estimated_ace = snm.psi
        
        # Store the bootstrap ACE
        bootstrap_ace.append(estimated_ace)
    
    # Calculate the confidence intervals
    lower_bound = np.percentile(bootstrap_ace, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_ace, 100 * (1 - alpha / 2))
    
    return lower_bound, upper_bound


# Calculate the 95% confidence interval for the ACE
lower, upper = bootstrap_confidence_interval(data1718_S, 'treatment', 'outcome', ['covariate4'])
print(f"95% Confidence Interval for ACE: ({lower}, {upper})")
'''
####### THE INTERVALS USED IN THESIS #######
'''
# Your confidence intervals
ci_data = {
    '2009-2010': [-0.16266443011101564, 0.30300758026712094],
    '2010-2011': [-0.45818928291116484, 0.04362085755410262],
    '2012-2013': [-0.2178423966472297, 0.21465453281197977],
    '2013-2014': [-0.22652224015111969, 0.1607956965585992],
    '2014-2015': [-0.11008551240613203, 0.2271084479764875],
    '2015-2016': [-0.09485846061270076, 0.3739353339179975],
    '2016-2017': [-0.2767365791506826, 0.048165803623831285],
    '2017-2018': [-0.015137314238144229, 0.3870501666687238],
}

# Extract the intervals and labels
labels = list(ci_data.keys())
lower_bounds = [ci[0] for ci in ci_data.values()]
upper_bounds = [ci[1] for ci in ci_data.values()]
means = [(ci[0] + ci[1]) / 2 for ci in ci_data.values()]

# Number of intervals
n = len(labels)

# Index for each interval
index = np.arange(n)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the error bars (confidence intervals)
ax.errorbar(means, index, xerr=[means - np.array(lower_bounds), np.array(upper_bounds) - means],
            fmt='o', color='blue', ecolor='blue', elinewidth=2, capsize=4, label="Confidence Intervals")

# Plot the overall mean (mu) line
ax.axvline(x=0, color='black', linestyle='--')

# Plot each point
for i, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
    ax.plot(mean, i, 'o', color='red')

# Customize the plot
ax.set_yticks(index)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # Invert y axis to have the first interval on top
ax.set_xlabel('Estimated Effect Size')
ax.set_title('Confidence Intervals for the a.c.e in each season')

# Show gridlines for y-axis
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()
'''

######## CHI-SQUARE INDEPENCE TESTS FOR TREATMENT T_C AND OUTCOME Y_R FOR QUESTION 1.1 ########
T = np.array(Treatment_1718)
Y = np.array(Result_1718)
# Create a contingency table
contingency_table = pd.crosstab(T, Y)

# Chi-Square Test of Independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

#print("Contingency Table:")
#print(contingency_table)
#print("\nChi-Square Test:")
#print(f"Chi2: {chi2}")
#print(f"P-value (Chi-Square Test): {p}")