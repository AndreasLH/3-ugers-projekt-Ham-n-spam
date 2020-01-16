# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:03:19 2020

@author: Kirstine Cort Graae
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def confusion_matrix_visual(conf_matrix,model):
    if model == 0:
        rc('font',weight = 'bold')
        width_bars = 0.5
        
        model = ['GaussNB','MultiNB','KNN']
        
        test_size_conf = conf_matrix[0,0] + conf_matrix[0,1] + conf_matrix[1,0] + conf_matrix[1,1]
        
        height_y = test_size_conf + 50
        
        position_x = 0
        
        p4 = plt.bar(position_x,conf_matrix[0,0],color = '#C25283',edgecolor = 'white', 
                width = width_bars, bottom = conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
        p3 = plt.bar(position_x,conf_matrix[1,1],color = '#7D0552',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1]+conf_matrix[1,0])
        p2 = plt.bar(position_x,conf_matrix[1,0],color = '#7E587E',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1])
        p1 = plt.bar(position_x,conf_matrix[0,1],color = '#571B7E',edgecolor = 'white', 
                width = width_bars)
        plt.xticks(np.arange(3),model,fontweight = 'bold')
        plt.xlabel('Model')
        plt.yticks(np.arange(0,height_y,100))
        plt.ylabel('Test size')
        plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))
        
        
    elif model == 1:
        rc('font',weight = 'bold')
        width_bars = 0.5
        
        model = ['GaussNB','MultiNB','KNN']
        
        test_size_conf = conf_matrix[0,0] + conf_matrix[0,1] + conf_matrix[1,0] + conf_matrix[1,1]
        
        height_y = test_size_conf + 50
        
        position_x = 1
        
        p4 = plt.bar(position_x,conf_matrix[0,0],color = '#C25283',edgecolor = 'white', 
                width = width_bars, bottom = conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
        p3 = plt.bar(position_x,conf_matrix[1,1],color = '#7D0552',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1]+conf_matrix[1,0])
        p2 = plt.bar(position_x,conf_matrix[1,0],color = '#7E587E',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1])
        p1 = plt.bar(position_x,conf_matrix[0,1],color = '#571B7E',edgecolor = 'white', 
                width = width_bars)
        plt.xticks(np.arange(3),model,fontweight = 'bold')
        plt.xlabel('Model')
        plt.yticks(np.arange(0,height_y,100))
        plt.ylabel('Test size')
        plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))
        
        
    elif model == 2:
        rc('font',weight = 'bold')
        width_bars = 0.5
        
        model = ['GaussNB','MultiNB','KNN']
        
        test_size_conf = conf_matrix[0,0] + conf_matrix[0,1] + conf_matrix[1,0] + conf_matrix[1,1]
        
        height_y = test_size_conf + 50
        
        position_x = 2
        
        p4 = plt.bar(position_x,conf_matrix[0,0],color = '#C25283',edgecolor = 'white', 
                width = width_bars, bottom = conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
        p3 = plt.bar(position_x,conf_matrix[1,1],color = '#7D0552',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1]+conf_matrix[1,0])
        p2 = plt.bar(position_x,conf_matrix[1,0],color = '#7E587E',edgecolor = 'white', 
                width = width_bars,bottom = conf_matrix[0,1])
        p1 = plt.bar(position_x,conf_matrix[0,1],color = '#571B7E',edgecolor = 'white', 
                width = width_bars)
        plt.xticks(np.arange(3),model,fontweight = 'bold')
        plt.xlabel('Model')
        plt.yticks(np.arange(0,height_y,100))
        plt.ylabel('Test size')
        plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))

    plt.show()        


        
# =============================================================================
# rc('font',weight = 'bold')
# position_x =np.arange(3)
# width_bars = 0.5
# 
# model = ['GaussNB','MultiNB','KNN']
# 
# test_size_conf = conf_matrix[0,0] + conf_matrix[0,1] + conf_matrix[1,0] + conf_matrix[1,1]
# 
# height_y = test_size_conf + 50
# 
# p4 = plt.bar(position_x[0],conf_matrix[0,0],color = '#C25283',edgecolor = 'white', 
#         width = width_bars, bottom = conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
# p3 = plt.bar(position_x[0],conf_matrix[1,1],color = '#7D0552',edgecolor = 'white', 
#         width = width_bars,bottom = conf_matrix[0,1]+conf_matrix[1,0])
# p2 = plt.bar(position_x[0],conf_matrix[1,0],color = '#7E587E',edgecolor = 'white', 
#         width = width_bars,bottom = conf_matrix[0,1])
# p1 = plt.bar(position_x[0],conf_matrix[0,1],color = '#571B7E',edgecolor = 'white', 
#         width = width_bars)
# 
# plt.xticks(position_x,model,fontweight = 'bold')
# plt.xlabel('Model')
# plt.yticks(np.arange(0,height_y,100))
# plt.ylabel('Test size')
# plt.legend((p1[0],p2[0],p3[0],p4[0]),('Ham as Spam','Spam as Ham','Spam as Spam','Ham as Ham'))
# =============================================================================
    
    
