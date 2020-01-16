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
Eks1_KB = np.array([[1543,183],[144,398]])
Eks2_NBG = np.array([[1460,60],[36,444]])
Eks2_NBM = np.array([[1503,17],[10,470]])
Eks2_KT = np.array([[1467,53],[15,465]])
Eks2_KB = np.array([[1193,327],[44,436]])


rc('font',weight = 'bold')

position_x = np.arange(4)

width_bars = 0.5

model = ['GaussNB','MultiNB','KNN TF-IDF','KNN BOW']

train_size_conf = np.sum(Eks1_NBG)
 
height_y = train_size_conf + 50

p4 = plt.bar(position_x[0],Eks1_NBG[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks1_NBG[0,1]+Eks1_NBG[1,0]+Eks1_NBG[1,1])

p3 = plt.bar(position_x[0],Eks1_NBG[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_NBG[0,1]+Eks1_NBG[1,0])

p2 = plt.bar(position_x[0],Eks1_NBG[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_NBG[0,1])

p1 = plt.bar(position_x[0],Eks1_NBG[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p8 = plt.bar(position_x[1],Eks1_NBM[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks1_NBM[0,1]+Eks1_NBM[1,0]+Eks1_NBM[1,1])

p7 = plt.bar(position_x[1],Eks1_NBM[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_NBM[0,1]+Eks1_NBM[1,0])

p6 = plt.bar(position_x[1],Eks1_NBM[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_NBM[0,1])

p5 = plt.bar(position_x[1],Eks1_NBM[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p12 = plt.bar(position_x[2],Eks1_KT[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks1_KT[0,1]+Eks1_KT[1,0]+Eks1_KT[1,1])

p11 = plt.bar(position_x[2],Eks1_KT[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_KT[0,1]+Eks1_KT[1,0])

p10 = plt.bar(position_x[2],Eks1_KT[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_KT[0,1])

p9 = plt.bar(position_x[2],Eks1_KT[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p16 = plt.bar(position_x[3],Eks1_KB[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks1_KB[0,1]+Eks1_KB[1,0]+Eks1_KB[1,1])

p15 = plt.bar(position_x[3],Eks1_KB[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_KB[0,1]+Eks1_KB[1,0])

p14 = plt.bar(position_x[3],Eks1_KB[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks1_KB[0,1])

p13 = plt.bar(position_x[3],Eks1_KB[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.title('Eksperiment 1 fordeling',fontsize = 20,fontweight = 'bold')
plt.yticks(np.arange(0,height_y,100))
plt.ylabel('Test size')
plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))
plt.savefig('conf_vis_eks1')
plt.show()


train_size_conf2 = np.sum(Eks2_NBG)
height_y2 = train_size_conf2

p4 = plt.bar(position_x[0],Eks2_NBG[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks2_NBG[0,1]+Eks2_NBG[1,0]+Eks2_NBG[1,1])

p3 = plt.bar(position_x[0],Eks2_NBG[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_NBG[0,1]+Eks2_NBG[1,0])

p2 = plt.bar(position_x[0],Eks2_NBG[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_NBG[0,1])

p1 = plt.bar(position_x[0],Eks2_NBG[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p8 = plt.bar(position_x[1],Eks2_NBM[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks2_NBM[0,1]+Eks2_NBM[1,0]+Eks2_NBM[1,1])

p7 = plt.bar(position_x[1],Eks2_NBM[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_NBM[0,1]+Eks2_NBM[1,0])

p6 = plt.bar(position_x[1],Eks2_NBM[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_NBM[0,1])

p5 = plt.bar(position_x[1],Eks2_NBM[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p12 = plt.bar(position_x[2],Eks2_KT[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks2_KT[0,1]+Eks2_KT[1,0]+Eks2_KT[1,1])

p11 = plt.bar(position_x[2],Eks2_KT[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_KT[0,1]+Eks2_KT[1,0])

p10 = plt.bar(position_x[2],Eks2_KT[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_KT[0,1])

p9 = plt.bar(position_x[2],Eks2_KT[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


p16 = plt.bar(position_x[3],Eks2_KB[0,0],color = '#C25283',edgecolor = 'white', 
        width = width_bars, bottom = Eks2_KB[0,1]+Eks2_KB[1,0]+Eks2_KB[1,1])

p15 = plt.bar(position_x[3],Eks2_KB[1,1],color = '#7D0552',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_KB[0,1]+Eks2_KB[1,0])

p14 = plt.bar(position_x[3],Eks2_KB[1,0],color = '#7E587E',edgecolor = 'white', 
        width = width_bars,bottom = Eks2_KB[0,1])

p13 = plt.bar(position_x[3],Eks2_KB[0,1],color = '#571B7E',edgecolor = 'white', 
        width = width_bars)


plt.xticks(position_x,model,fontweight = 'bold')
plt.xlabel('Model')
plt.yticks(np.arange(0,height_y2,100))
plt.title('Eksperiment 2 fordeling',fontsize = 20,fontweight = 'bold')
plt.ylabel('Test size')
plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))
plt.savefig('conf_vis_eks2')
plt.show()






