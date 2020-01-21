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
Eks2_KT = np.array([[1703,23],[34,508]])
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

#Conf_int
conf_int_NBG1 = 0.007977475002*100
conf_int_NBG2 = 0.01648147578*100
conf_int_NBG3 = 0.005064197040*100
conf_int_NBG4 = 0.01790415782*100

conf_int_NBM1 = 0.003217877947*100
conf_int_NBM2 = 0.01741623664*100
conf_int_NBM3 = 0.004116254577*100
conf_int_NBM4 = 0.01777902273*100

conf_int_KT1 = 0.006319864574*100
conf_int_KT2 = 0.01692520352*100
conf_int_KT3 = 0.003840986420*100
conf_int_KT4 = 0.01774718362*100

conf_int_KB1 = 0.01149124853*100
conf_int_KB2 = 0.01486790645*100
conf_int_KB3 = 0.007847257165*100
conf_int_KB4 = 0.01839777228*100


#Barplottetene laves

p4 = plt.bar(position_x[0],NBG_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = NBG_per1_2+NBG_per1_3+NBG_per1_4,yerr = conf_int_NBG4)

p3 = plt.bar(position_x[0],NBG_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = NBG_per1_4 + NBG_per1_3,yerr = conf_int_NBG2)

p2 = plt.bar(position_x[0],NBG_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = NBG_per1_4, yerr = conf_int_NBG1)

p1 = plt.bar(position_x[0],NBG_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_NBG3)


p8 = plt.bar(position_x[1],NBM_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = NBM_per1_2+NBM_per1_3+NBM_per1_4, yerr = conf_int_NBM4)

p7 = plt.bar(position_x[1],NBM_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = NBM_per1_4 + NBM_per1_3, yerr = conf_int_NBM2)

p6 = plt.bar(position_x[1],NBM_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = NBM_per1_4, yerr = conf_int_NBM1)

p5 = plt.bar(position_x[1],NBM_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_NBM3)


p12 = plt.bar(position_x[2],KT_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = KT_per1_2+KT_per1_3+KT_per1_4, yerr = conf_int_KT4)

p11 = plt.bar(position_x[2],KT_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = KT_per1_4 + KT_per1_3, yerr = conf_int_KT2)

p10 = plt.bar(position_x[2],KT_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = KT_per1_4, yerr = conf_int_KT1)

p9 = plt.bar(position_x[2],KT_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_KT3)


p16 = plt.bar(position_x[3],KB_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = KB_per1_2+KB_per1_3+KB_per1_4, yerr = conf_int_KB4)

p15 = plt.bar(position_x[3],KB_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = KB_per1_4 + KB_per1_3, yerr = conf_int_KB2)

p14 = plt.bar(position_x[3],KB_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = KB_per1_4, yerr = conf_int_KB1)

p13 = plt.bar(position_x[3],KB_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_KB3)

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
plt.savefig('conf_vis_eks1', dpi = 600)
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

#Conf_int
conf_int_NBG1 = 0.005473754236*100
conf_int_NBG2 = 0.01709809963*100
conf_int_NBG3 = 0.007105608219*100
conf_int_NBG4 = 0.01824940362*100

conf_int_NBM1 = 0.002582897147*100
conf_int_NBM2 = 0.01747300770*100
conf_int_NBM3 = 0.004455881530*100
conf_int_NBM4 = 0.01782112770*100

conf_int_KT1 = 0.005134885106*100
conf_int_KT2 = 0.01715851450*100
conf_int_KT3 = 0.004289583451*100
conf_int_KT4 = 0.01780012467*100

conf_int_KB1 = 0.009481148660*100
conf_int_KB2 = 0.01592790485*100
conf_int_KB3 = 0.01017915464*100
conf_int_KB4 = 0.01892745280*100


#Barplottetene laves

p4 = plt.bar(position_x[0],NBG_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = NBG_per1_2+NBG_per1_3+NBG_per1_4,yerr = conf_int_NBG4)

p3 = plt.bar(position_x[0],NBG_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = NBG_per1_4 + NBG_per1_3,yerr = conf_int_NBG2)

p2 = plt.bar(position_x[0],NBG_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = NBG_per1_4, yerr = conf_int_NBG1)

p1 = plt.bar(position_x[0],NBG_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_NBG3)


p8 = plt.bar(position_x[1],NBM_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = NBM_per1_2+NBM_per1_3+NBM_per1_4, yerr = conf_int_NBM4)

p7 = plt.bar(position_x[1],NBM_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = NBM_per1_4 + NBM_per1_3, yerr = conf_int_NBM2)

p6 = plt.bar(position_x[1],NBM_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = NBM_per1_4, yerr = conf_int_NBM1)

p5 = plt.bar(position_x[1],NBM_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_NBM3)


p12 = plt.bar(position_x[2],KT_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = KT_per1_2+KT_per1_3+KT_per1_4, yerr = conf_int_KT4)

p11 = plt.bar(position_x[2],KT_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = KT_per1_4 + KT_per1_3, yerr = conf_int_KT2)

p10 = plt.bar(position_x[2],KT_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = KT_per1_4, yerr = conf_int_KT1)

p9 = plt.bar(position_x[2],KT_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_KT3)


p16 = plt.bar(position_x[3],KB_per1_1,color = '#C25283',edgecolor = 'white',
        width = width_bars, bottom = KB_per1_2+KB_per1_3+KB_per1_4, yerr = conf_int_KB4)

p15 = plt.bar(position_x[3],KB_per1_2,color = '#7D0552',edgecolor = 'white',
        width = width_bars,bottom = KB_per1_4 + KB_per1_3, yerr = conf_int_KB2)

p14 = plt.bar(position_x[3],KB_per1_3,color = '#7E587E',edgecolor = 'white',
        width = width_bars,bottom = KB_per1_4, yerr = conf_int_KB1)

p13 = plt.bar(position_x[3],KB_per1_4,color = '#571B7E',edgecolor = 'white',
        width = width_bars, yerr = conf_int_KB3)

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
plt.savefig('conf_vis_eks2', dpi=600)
plt.show()