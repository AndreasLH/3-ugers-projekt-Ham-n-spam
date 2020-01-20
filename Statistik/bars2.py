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
conf_int_NBG = 0.007977475002*100
conf_int_NBG1 = 0.005064197040*100

conf_int_NBM = 0.003217877947*100
conf_int_NBM1 = 0.004116254577*100

conf_int_KT = 0.006319864574*100
conf_int_KT1 = 0.003840986420*100

conf_int_KB = 0.01149124853*100
conf_int_KB1 = 0.007847257165*100


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
conf_int_NBG = 0.005473754236*100
conf_int_NBG1 = 0.007105608219*100

conf_int_NBM = 0.002582897147*100
conf_int_NBM1 = 0.004455881530*100

conf_int_KT = 0.003744561580*100
conf_int_KT1 = 0.006646650907*100

conf_int_KB = 0.009481148660*100
conf_int_KB1 = 0.01017915464*100

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



















































