# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:47:20 2020

@author: Kirstine Cort Graae
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:09:35 2020

@author: Kirstine Cort Graae
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

Eks1_NBG = np.array([[1693,33],[87,455]])
Eks1_NBM = np.array([[1705,21],[12,530]])
Eks1_KT = np.array([[1708,18],[53,489]])
Eks1_KB = np.array([[1642,84],[192,350]])
Eks2_NBG = np.array([[1658,68],[39,503]])
Eks2_NBM = np.array([[1701,25],[7,535]])
Eks2_KT = np.array([[1667,59],[17,525]])
Eks2_KB = np.array([[1579,147],[126,416]])

per1_NBG = round((Eks1_NBG[0,1]/(Eks1_NBG[1,0]+Eks1_NBG[0,1]))*100,ndigits = 3)
per2_NBG = round((Eks1_NBG[1,0]/(Eks1_NBG[1,0]+Eks1_NBG[0,1]))*100,ndigits = 3)

per1_NBM = round((Eks1_NBM[0,1]/(Eks1_NBM[1,0]+Eks1_NBM[0,1]))*100,ndigits = 3)
per2_NBM = round((Eks1_NBM[1,0]/(Eks1_NBM[1,0]+Eks1_NBM[0,1]))*100,ndigits = 3)

per1_KT = round((Eks1_KT[0,1]/(Eks1_KT[1,0]+Eks1_KT[0,1]))*100,ndigits = 3)
per2_KT = round((Eks1_KT[1,0]/(Eks1_KT[1,0]+Eks1_KT[0,1]))*100,ndigits = 3)

per1_KB = round((Eks1_KB[0,1]/(Eks1_KB[1,0]+Eks1_KB[0,1]))*100,ndigits = 3)
per2_KB = round((Eks1_KB[1,0]/(Eks1_KB[1,0]+Eks1_KB[0,1]))*100,ndigits = 3)

rc('font',weight = 'bold')

position_x = np.arange(4)

width_bars = 0.5

model = ['GaussNB','MultiNB','KNN TF-IDF','KNN BOW']

#NBG CONF
n_NBG = Eks1_NBG[1,1]+Eks1_NBG[1,0]
p_NBG = Eks1_NBG[1,0]/n_NBG
n1_NBG = n_NBG + 4
p1_NBG = (p_NBG*n_NBG+2)/(n_NBG+4)
conf_int_NBG = 1.96**2*np.sqrt((p1_NBG*(1-p1_NBG))/n1_NBG)*100

n_NBG1 = Eks1_NBG[0,0]+Eks1_NBG[0,1]
p_NBG1 = Eks1_NBG[0,1]/n_NBG1
n1_NBG1 = n_NBG + 4
p1_NBG1 = (p_NBG1*n_NBG1+2)/(n_NBG1+4)
conf_int_NBG1 = 1.96**2*np.sqrt((p1_NBG1*(1-p1_NBG1))/n1_NBG1)*100

#NBM CONF
n_NBM = Eks1_NBM[1,1]+Eks1_NBM[1,0]
p_NBM = Eks1_NBM[1,0]/n_NBM
n1_NBM = n_NBM + 4
p1_NBM = (p_NBM*n_NBM+2)/(n_NBM+4)
conf_int_NBM = 1.96**2*np.sqrt((p1_NBM*(1-p1_NBM))/n1_NBM)*100

n_NBM1 = Eks1_NBM[0,0]+Eks1_NBM[0,1]
p_NBM1 = Eks1_NBM[0,1]/n_NBM1
n1_NBM1 = n_NBM + 4
p1_NBM1 = (p_NBM1*n_NBM1+2)/(n_NBM1+4)
conf_int_NBM1 = 1.96**2*np.sqrt((p1_NBM1*(1-p1_NBM1))/n1_NBM1)*100

#KT CONF
n_KT = Eks1_KT[1,1]+Eks1_KT[1,0]
p_KT = Eks1_KT[1,0]/n_KT
n1_KT = n_KT + 4
p1_KT = (p_KT*n_KT+2)/(n_KT+4)
conf_int_KT = 1.96**2*np.sqrt((p1_KT*(1-p1_KT))/n1_KT)*100

n_KT1 = Eks1_KT[0,0]+Eks1_KT[0,1]
p_KT1 = Eks1_KT[0,1]/n_KT1
n1_KT1 = n_KT1 + 4
p1_KT1 = (p_KT1*n_KT1+2)/(n_KT1+4)
conf_int_KT1 = 1.96**2*np.sqrt((p1_KT1*(1-p1_KT1))/n1_KT1)*100

#KB CONF
n_KB = Eks1_KB[1,1]+Eks1_KB[1,0]
p_KB = Eks1_KB[1,0]/n_KB
n1_KB = n_KB + 4
p1_KB = (p_KB*n_KB+2)/(n_KB+4)
conf_int_KB = 1.96**2*np.sqrt((p1_KB*(1-p1_KB))/n1_KB)*100

n_KB1 = Eks1_KB[0,0]+Eks1_KB[0,1]
p_KB1 = Eks1_KB[0,1]/n_KB1
n1_KB1 = n_KB1 + 4
p1_KB1 = (p_KB1*n_KB1+2)/(n_KB1+4)
conf_int_KB1 = 1.96**2*np.sqrt((p1_KB1*(1-p1_KB1))/n1_KB1)*100


p17 = plt.bar(position_x[0],per2_NBG+per1_NBG,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBG)

p18 = plt.bar(position_x[0],per2_NBG,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBG1)


p19 = plt.bar(position_x[1],per1_NBM+per2_NBM,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBM)

p20 = plt.bar(position_x[1],per2_NBM,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBM1)


p21 = plt.bar(position_x[2],per1_KT+per2_KT,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KT)

p22 = plt.bar(position_x[2],per2_KT,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KT1)


p23 = plt.bar(position_x[3],per1_KB+per2_KB,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KB)

p24 = plt.bar(position_x[3],per2_KB,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KB1)


plt.text(position_x[0]-0.2,80,per1_NBG,color='white')
plt.text(position_x[1]-0.2,80,per1_NBM,color='white')
plt.text(position_x[2]-0.2,80,per1_KT,color='white')
plt.text(position_x[3]-0.2,80,per1_KB,color='white')
plt.text(position_x[0]-0.2,50,per2_NBG,color='white')
plt.text(position_x[1]-0.2,20,per2_NBM,color='white')
plt.text(position_x[2]-0.2,50,per2_KT,color='white')
plt.text(position_x[3]-0.2,50,per2_KB,color='white')

plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Forkerte Klassificerede eksperiment 1',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,100,10))
plt.ylabel('Test størrelse')
plt.legend((p17[0],p18[0]),('Ham as Spam','Spam as Ham'))
plt.savefig('Forkerte klassificerede1',dpi = 600)
plt.show()

per1_NBG = round((Eks2_NBG[0,1]/(Eks2_NBG[1,0]+Eks2_NBG[0,1]))*100,ndigits = 3)
per2_NBG = round((Eks2_NBG[1,0]/(Eks2_NBG[1,0]+Eks2_NBG[0,1]))*100,ndigits = 3)

per1_NBM = round((Eks2_NBM[0,1]/(Eks2_NBM[1,0]+Eks2_NBM[0,1]))*100,ndigits = 3)
per2_NBM = round((Eks2_NBM[1,0]/(Eks2_NBM[1,0]+Eks2_NBM[0,1]))*100,ndigits = 3)

per1_KT = round((Eks2_KT[0,1]/(Eks2_KT[1,0]+Eks2_KT[0,1]))*100,ndigits = 3)
per2_KT = round((Eks2_KT[1,0]/(Eks2_KT[1,0]+Eks2_KT[0,1]))*100,ndigits = 3)

per1_KB = round((Eks2_KB[0,1]/(Eks2_KB[1,0]+Eks2_KB[0,1]))*100,ndigits = 3)
per2_KB = round((Eks2_KB[1,0]/(Eks2_KB[1,0]+Eks2_KB[0,1]))*100,ndigits = 3)

rc('font',weight = 'bold')

position_x = np.arange(4)

width_bars = 0.5

model = ['GaussNB','MultiNB','KNN TF-IDF','KNN BOW']


#NBG CONF
n_NBG = Eks2_NBG[1,1]+Eks2_NBG[1,0]
p_NBG = Eks2_NBG[1,0]/n_NBG
n1_NBG = n_NBG + 4
p1_NBG = (p_NBG*n_NBG+2)/(n_NBG+4)
conf_int_NBG = 1.96**2*np.sqrt((p1_NBG*(1-p1_NBG))/n1_NBG)*100

n_NBG1 = Eks2_NBG[0,0]+Eks2_NBG[0,1]
p_NBG1 = Eks2_NBG[0,1]/n_NBG1
n1_NBG1 = n_NBG + 4
p1_NBG1 = (p_NBG1*n_NBG1+2)/(n_NBG1+4)
conf_int_NBG1 = 1.96**2*np.sqrt((p1_NBG1*(1-p1_NBG1))/n1_NBG1)*100

#NBM CONF
n_NBM = Eks2_NBM[1,1]+Eks2_NBM[1,0]
p_NBM = Eks2_NBM[1,0]/n_NBM
n1_NBM = n_NBM + 4
p1_NBM = (p_NBM*n_NBM+2)/(n_NBM+4)
conf_int_NBM = 1.96**2*np.sqrt((p1_NBM*(1-p1_NBM))/n1_NBM)*100

n_NBM1 = Eks2_NBM[0,0]+Eks2_NBM[0,1]
p_NBM1 = Eks2_NBM[0,1]/n_NBM1
n1_NBM1 = n_NBM + 4
p1_NBM1 = (p_NBM1*n_NBM1+2)/(n_NBM1+4)
conf_int_NBM1 = 1.96**2*np.sqrt((p1_NBM1*(1-p1_NBM1))/n1_NBM1)*100

#KT CONF
n_KT = Eks2_KT[1,1]+Eks2_KT[1,0]
p_KT = Eks2_KT[1,0]/n_KT
n1_KT = n_KT + 4
p1_KT = (p_KT*n_KT+2)/(n_KT+4)
conf_int_KT = 1.96**2*np.sqrt((p1_KT*(1-p1_KT))/n1_KT)*100

n_KT1 = Eks2_KT[0,0]+Eks2_KT[0,1]
p_KT1 = Eks2_KT[0,1]/n_KT1
n1_KT1 = n_KT1 + 4
p1_KT1 = (p_KT1*n_KT1+2)/(n_KT1+4)
conf_int_KT1 = 1.96**2*np.sqrt((p1_KT1*(1-p1_KT1))/n1_KT1)*100

#KB CONF
n_KB = Eks2_KB[1,1]+Eks2_KB[1,0]
p_KB = Eks2_KB[1,0]/n_KB
n1_KB = n_KB + 4
p1_KB = (p_KB*n_KB+2)/(n_KB+4)
conf_int_KB = 1.96**2*np.sqrt((p1_KB*(1-p1_KB))/n1_KB)*100

n_KB1 = Eks2_KB[0,0]+Eks2_KB[0,1]
p_KB1 = Eks2_KB[0,1]/n_KB1
n1_KB1 = n_KB1 + 4
p1_KB1 = (p_KB1*n_KB1+2)/(n_KB1+4)
conf_int_KB1 = 1.96**2*np.sqrt((p1_KB1*(1-p1_KB1))/n1_KB1)*100


p17 = plt.bar(position_x[0],per2_NBG+per1_NBG,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBG)

p18 = plt.bar(position_x[0],per2_NBG,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBG1)


p19 = plt.bar(position_x[1],per1_NBM+per2_NBM,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBM)

p20 = plt.bar(position_x[1],per2_NBM,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_NBM1)


p21 = plt.bar(position_x[2],per1_KT+per2_KT,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KT)

p22 = plt.bar(position_x[2],per2_KT,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KT1)


p23 = plt.bar(position_x[3],per1_KB+per2_KB,color = '#571B7E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KB)

p24 = plt.bar(position_x[3],per2_KB,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,yerr = conf_int_KB1)


plt.text(position_x[0]-0.2,80,per1_NBG,color='white')
plt.text(position_x[1]-0.2,80,per1_NBM,color='white')
plt.text(position_x[2]-0.2,80,per1_KT,color='white')
plt.text(position_x[3]-0.2,80,per1_KB,color='white')
plt.text(position_x[0]-0.2,20,per2_NBG,color='white')
plt.text(position_x[1]-0.2,10,per2_NBM,color='white')
plt.text(position_x[2]-0.2,10,per2_KT,color='white')
plt.text(position_x[3]-0.2,20,per2_KB,color='white')

plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Forkerte Klassificerede eksperiment 2',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,100,10))
plt.ylabel('Test størrelse')
plt.legend((p17[0],p18[0]),('Ham as Spam','Spam as Ham'))
plt.savefig('Forkerte klassificerede2',dpi = 600)
plt.show()



















































