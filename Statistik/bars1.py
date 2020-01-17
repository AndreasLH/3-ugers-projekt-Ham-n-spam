# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:51:52 2020

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

width_bars = 0.7

model = ['GaussNB','MultiNB','KNN TF-IDF','KNN BOW'] 

#Den procetnvise part hver del udgør udregnes:
NBG_per1_1 = round((Eks1_NBG[0,0]/np.sum(Eks1_NBG))*100,ndigits = 3)
NBG_per1_2 = round((Eks1_NBG[1,1]/np.sum(Eks1_NBG))*100,ndigits = 3)
NBG_per1_3 = round((Eks1_NBG[1,0]/np.sum(Eks1_NBG))*100,ndigits = 3)
NBG_per1_4 = round((Eks1_NBG[0,1]/np.sum(Eks1_NBG))*100,ndigits = 3)

NBM_per1_1 = round((Eks1_NBM[0,0]/np.sum(Eks1_NBM))*100,ndigits = 3)
NBM_per1_2 = round((Eks1_NBM[1,1]/np.sum(Eks1_NBM))*100,ndigits = 3)
NBM_per1_3 = round((Eks1_NBM[1,0]/np.sum(Eks1_NBM))*100,ndigits = 3)
NBM_per1_4 = round((Eks1_NBM[0,1]/np.sum(Eks1_NBM))*100,ndigits = 3)

KT_per1_1 = round((Eks1_KT[0,0]/np.sum(Eks1_KT))*100,ndigits = 3)
KT_per1_2 = round((Eks1_KT[1,1]/np.sum(Eks1_KT))*100,ndigits = 3)
KT_per1_3 = round((Eks1_KT[1,0]/np.sum(Eks1_KT))*100,ndigits = 3)
KT_per1_4 = round((Eks1_KT[0,1]/np.sum(Eks1_KT))*100,ndigits = 3)

KB_per1_1 = round((Eks1_KB[0,0]/np.sum(Eks1_KB))*100,ndigits = 3)
KB_per1_2 = round((Eks1_KB[1,1]/np.sum(Eks1_KB))*100,ndigits = 3)
KB_per1_3 = round((Eks1_KB[1,0]/np.sum(Eks1_KB))*100,ndigits = 3)
KB_per1_4 = round((Eks1_KB[0,1]/np.sum(Eks1_KB))*100,ndigits = 3)

#Barplottetene laves

p4 = plt.bar(position_x[0],NBG_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = NBG_per1_2+NBG_per1_3+NBG_per1_4)

p3 = plt.bar(position_x[0],NBG_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = NBG_per1_4 + NBG_per1_3)

p2 = plt.bar(position_x[0],NBG_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = NBG_per1_4)

p1 = plt.bar(position_x[0],NBG_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p8 = plt.bar(position_x[1],NBM_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = NBM_per1_2+NBM_per1_3+NBM_per1_4)

p7 = plt.bar(position_x[1],NBM_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = NBM_per1_4 + NBM_per1_3)

p6 = plt.bar(position_x[1],NBM_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = NBM_per1_4)

p5 = plt.bar(position_x[1],NBM_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p12 = plt.bar(position_x[2],KT_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = KT_per1_2+KT_per1_3+KT_per1_4)

p11 = plt.bar(position_x[2],KT_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = KT_per1_4 + KT_per1_3)

p10 = plt.bar(position_x[2],KT_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = KT_per1_4)

p9 = plt.bar(position_x[2],KT_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p16 = plt.bar(position_x[3],KB_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = KB_per1_2+KB_per1_3+KB_per1_4)

p15 = plt.bar(position_x[3],KB_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = KB_per1_4 + KB_per1_3)

p14 = plt.bar(position_x[3],KB_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = KB_per1_4)

p13 = plt.bar(position_x[3],KB_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)

#Teksten skrives

plt.text(position_x[0]-0.18,((NBG_per1_1+NBG_per1_2+NBG_per1_3+NBG_per1_4-25)),NBG_per1_1,color='white')
plt.text(position_x[0]-0.18,((NBG_per1_2+NBG_per1_3+NBG_per1_4)-15),NBG_per1_2,color='white')
plt.text(position_x[0]-0.18,((NBG_per1_3+NBG_per1_4)-1.5),NBG_per1_3,color='white')
plt.text(position_x[0]-0.18,(NBG_per1_4/2+0.2),NBG_per1_4,color='white')

plt.text(position_x[1]-0.18,((NBM_per1_1+NBM_per1_2+NBM_per1_3+NBM_per1_4-25)),NBM_per1_1,color='white')
plt.text(position_x[1]-0.18,((NBM_per1_2+NBM_per1_3+NBM_per1_4)-15),NBM_per1_2,color='white')
plt.text(position_x[1]-0.18,((NBM_per1_3+NBM_per1_4)-0.3),NBM_per1_3,color='white')
plt.text(position_x[1]-0.18,(NBM_per1_4/2+0.3),NBM_per1_4,color='white')

plt.text(position_x[2]-0.18,((KT_per1_1+KT_per1_2+KT_per1_3+KT_per1_4-25)),KT_per1_1,color='white')
plt.text(position_x[2]-0.18,((KT_per1_2+KT_per1_3+KT_per1_4)-15),KT_per1_2,color='white')
plt.text(position_x[2]-0.18,((KT_per1_3+KT_per1_4)-1),KT_per1_3,color='white')

plt.text(position_x[3]-0.18,((KB_per1_1+KB_per1_2+KB_per1_3+KB_per1_4-25)),KB_per1_1,color='white')
plt.text(position_x[3]-0.18,((KB_per1_2+KB_per1_3+KB_per1_4)-10),KB_per1_2,color='white')
plt.text(position_x[3]-0.18,((KB_per1_3+KB_per1_4)-4),KB_per1_3,color='white')
plt.text(position_x[3]-0.18,(KB_per1_4/2+0.2),KB_per1_4,color='white')

#Der plottes
plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Eksperiment 1 fordeling',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,100,10))
plt.yscale('log')
plt.ylabel('Procent (log)')
plt.savefig('conf_vis_eks1',dpi=600)
plt.show()



NBG_per1_1 = round((Eks2_NBG[0,0]/np.sum(Eks2_NBG))*100,ndigits = 3)
NBG_per1_2 = round((Eks2_NBG[1,1]/np.sum(Eks2_NBG))*100,ndigits = 3)
NBG_per1_3 = round((Eks2_NBG[1,0]/np.sum(Eks2_NBG))*100,ndigits = 3)
NBG_per1_4 = round((Eks2_NBG[0,1]/np.sum(Eks2_NBG))*100,ndigits = 3)

NBM_per1_1 = round((Eks2_NBM[0,0]/np.sum(Eks2_NBM))*100,ndigits = 3)
NBM_per1_2 = round((Eks2_NBM[1,1]/np.sum(Eks2_NBM))*100,ndigits = 3)
NBM_per1_3 = round((Eks2_NBM[1,0]/np.sum(Eks2_NBM))*100,ndigits = 3)
NBM_per1_4 = round((Eks2_NBM[0,1]/np.sum(Eks2_NBM))*100,ndigits = 3)

KT_per1_1 = round((Eks2_KT[0,0]/np.sum(Eks2_KT))*100,ndigits = 3)
KT_per1_2 = round((Eks2_KT[1,1]/np.sum(Eks2_KT))*100,ndigits = 3)
KT_per1_3 = round((Eks2_KT[1,0]/np.sum(Eks2_KT))*100,ndigits = 3)
KT_per1_4 = round((Eks2_KT[0,1]/np.sum(Eks2_KT))*100,ndigits = 3)

KB_per1_1 = round((Eks2_KB[0,0]/np.sum(Eks2_KB))*100,ndigits = 3)
KB_per1_2 = round((Eks2_KB[1,1]/np.sum(Eks2_KB))*100,ndigits = 3)
KB_per1_3 = round((Eks2_KB[1,0]/np.sum(Eks2_KB))*100,ndigits = 3)
KB_per1_4 = round((Eks2_KB[0,1]/np.sum(Eks2_KB))*100,ndigits = 3)

p4 = plt.bar(position_x[0],NBG_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = NBG_per1_2+NBG_per1_3+NBG_per1_4)

p3 = plt.bar(position_x[0],NBG_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = NBG_per1_4 + NBG_per1_3)

p2 = plt.bar(position_x[0],NBG_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = NBG_per1_4)

p1 = plt.bar(position_x[0],NBG_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p8 = plt.bar(position_x[1],NBM_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = NBM_per1_2+NBM_per1_3+NBM_per1_4)

p7 = plt.bar(position_x[1],NBM_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = NBM_per1_4 + NBM_per1_3)

p6 = plt.bar(position_x[1],NBM_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = NBM_per1_4)

p5 = plt.bar(position_x[1],NBM_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p12 = plt.bar(position_x[2],KT_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = KT_per1_2+KT_per1_3+KT_per1_4)

p11 = plt.bar(position_x[2],KT_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = KT_per1_4 + KT_per1_3)

p10 = plt.bar(position_x[2],KT_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = KT_per1_4)

p9 = plt.bar(position_x[2],KT_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p16 = plt.bar(position_x[3],KB_per1_1,color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = KB_per1_2+KB_per1_3+KB_per1_4)

p15 = plt.bar(position_x[3],KB_per1_2,color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = KB_per1_4 + KB_per1_3)

p14 = plt.bar(position_x[3],KB_per1_3,color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = KB_per1_4)

p13 = plt.bar(position_x[3],KB_per1_4,color = '#571B7E',edgecolor = 'white', 
        width = width_bars)

plt.text(position_x[0]-0.18,((NBG_per1_1+NBG_per1_2+NBG_per1_3+NBG_per1_4-25)),NBG_per1_1,color='white')
plt.text(position_x[0]-0.18,((NBG_per1_2+NBG_per1_3+NBG_per1_4)-15),NBG_per1_2,color='white')
plt.text(position_x[0]-0.18,((NBG_per1_3+NBG_per1_4)-1.5),NBG_per1_3,color='white')
plt.text(position_x[0]-0.18,(NBG_per1_4/2+0.2),NBG_per1_4,color='white')

plt.text(position_x[1]-0.18,((NBM_per1_1+NBM_per1_2+NBM_per1_3+NBM_per1_4-25)),NBM_per1_1,color='white')
plt.text(position_x[1]-0.18,((NBM_per1_2+NBM_per1_3+NBM_per1_4)-15),NBM_per1_2,color='white')
plt.text(position_x[1]-0.18,((NBM_per1_3+NBM_per1_4)-0.3),NBM_per1_3,color='white')

plt.text(position_x[2]-0.18,((KT_per1_1+KT_per1_2+KT_per1_3+KT_per1_4-25)),KT_per1_1,color='white')
plt.text(position_x[2]-0.18,((KT_per1_2+KT_per1_3+KT_per1_4)-15),KT_per1_2,color='white')
plt.text(position_x[2]-0.18,((KT_per1_3+KT_per1_4)-0.65),KT_per1_3,color='white')
plt.text(position_x[2]-0.18,(KT_per1_4/2+0.2),KT_per1_4,color='white')

plt.text(position_x[3]-0.18,((KB_per1_1+KB_per1_2+KB_per1_3+KB_per1_4-25)),KB_per1_1,color='white')
plt.text(position_x[3]-0.18,((KB_per1_2+KB_per1_3+KB_per1_4)-10),KB_per1_2,color='white')
plt.text(position_x[3]-0.18,((KB_per1_3+KB_per1_4)-4),KB_per1_3,color='white')
plt.text(position_x[3]-0.18,(KB_per1_4/2+0.2),KB_per1_4,color='white')

plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Eksperiment 2 fordeling',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,100,10))
plt.yscale('log')
plt.ylabel('Procent (log)')
plt.savefig('conf_vis_eks2',dpi=600)
plt.show()