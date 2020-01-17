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


rc('font',weight = 'bold')

position_x = np.arange(4)

width_bars = 0.5

model = ['GaussNB','MultiNB','KNN TF-IDF','KNN BOW']

train_size_conf3 = np.sum(Eks1_NBG[1,0]+Eks1_NBG[0,1]+Eks1_NBG[0,1])
 
height_y3 = train_size_conf3 + 50

p17 = plt.bar(position_x[0],Eks1_NBG[1,0]+Eks1_NBG[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p18 = plt.bar(position_x[0],Eks1_NBG[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p19 = plt.bar(position_x[1],Eks1_NBM[1,0]+Eks1_NBM[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p20 = plt.bar(position_x[1],Eks1_NBM[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p21 = plt.bar(position_x[2],Eks1_KT[1,0]+Eks1_KT[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p22 = plt.bar(position_x[2],Eks1_KT[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p23 = plt.bar(position_x[3],Eks1_KB[1,0]+Eks1_KB[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p24 = plt.bar(position_x[3],Eks1_KB[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)

per1 = round((Eks1_NBG[0,1]/(Eks1_NBG[1,0]+Eks1_NBG[0,1]))*100,ndigits = 3)
per2 = round((Eks1_NBM[0,1]/(Eks1_NBM[1,0]+Eks1_NBM[0,1]))*100,ndigits = 3)
per3 = round((Eks1_KT[0,1]/(Eks1_KT[1,0]+Eks1_KT[0,1]))*100,ndigits = 3)
per4 = round((Eks1_KB[0,1]/(Eks1_KB[1,0]+Eks1_KB[0,1]))*100,ndigits = 3)


plt.text(position_x[0]-0.2,((Eks1_NBG[0,1])/2),per1,color='white')
plt.text(position_x[1]-0.2,((Eks1_NBM[0,1])/2-3),per2,color='white')
plt.text(position_x[2]-0.2,((Eks1_KT[0,1])/2-3),per3,color='white')
plt.text(position_x[3]-0.2,((Eks1_KB[0,1])/2),per4,color='white')


plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Forkerte Klassificerede eksperiment 1',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,height_y3,100))
plt.ylabel('Test størrelse')
plt.legend((p17[0],p18[0]),('Forkerte klassificerede','Ham as Spam'))
plt.savefig('Forkerte klassificerede1')
plt.show()


train_size_conf4 = np.sum(Eks2_NBG[1,0]+Eks2_NBG[0,1]+Eks2_NBG[0,1])
 
height_y4 = train_size_conf4 + 50

p25 = plt.bar(position_x[0],Eks2_NBG[1,0]+Eks2_NBG[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p26 = plt.bar(position_x[0],Eks2_NBG[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p27 = plt.bar(position_x[1],Eks2_NBM[1,0]+Eks2_NBM[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p28 = plt.bar(position_x[1],Eks2_NBM[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p29 = plt.bar(position_x[2],Eks2_KT[1,0]+Eks2_KT[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p30 = plt.bar(position_x[2],Eks2_KT[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p31 = plt.bar(position_x[3],Eks2_KB[1,0]+Eks2_KB[0,1],color = '#7E587E',edgecolor = 'white', 
        width = width_bars)

p32 = plt.bar(position_x[3],Eks2_KB[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)

per5 = round((Eks2_NBG[0,1]/(Eks2_NBG[1,0]+Eks2_NBG[0,1]))*100,ndigits = 3)
per6 = round((Eks2_NBM[0,1]/(Eks2_NBM[1,0]+Eks2_NBM[0,1]))*100,ndigits = 3)
per7 = round((Eks2_KT[0,1]/(Eks2_KT[1,0]+Eks2_KT[0,1]))*100,ndigits = 3)
per8 = round((Eks2_KB[0,1]/(Eks2_KB[1,0]+Eks2_KB[0,1]))*100,ndigits = 3)


plt.text(position_x[0]-0.2,((Eks2_NBG[0,1])/2),per5,color='white')
plt.text(position_x[1]-0.2,((Eks2_NBM[0,1])/2-3),per6,color='white')
plt.text(position_x[2]-0.2,((Eks2_KT[0,1])/2),per7,color='white')
plt.text(position_x[3]-0.2,((Eks2_KB[0,1])/2),per8,color='white')


plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Forkerte Klassificerede eksperiment 2',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,height_y3,100))
plt.ylabel('Test størrelse')
plt.legend((p25[0],p26[0]),('Forkerte klassificerede','Ham as Spam'))
plt.savefig('Forkerte klassificerede2')
plt.show()




















































