from Engine import *
from data_load_fit import *
import numpy

PerFieldDataFreeDeltas = Pdata

f, ax = plt.subplots(1)
ax.plot(PerB,(numpy.flipud(PerFieldDataFreeDeltas).transpose())[2],color = 'black',label = '$\Gamma_1$',marker = 's',\
             markersize = 6,linewidth = 0)
ax.plot(PerB,(numpy.flipud(PerFieldDataFreeDeltas).transpose())[3],color = 'red',label = '$\Gamma_2$',marker = 's',\
             markersize = 6,linewidth = 0)
ax.plot(PerB,(numpy.flipud(PerFieldDataFreeDeltas).transpose())[0],color = 'blue',label = '$\Delta_1$',marker = 's',\
             markersize = 6,linewidth = 0)
ax.plot(PerB,(numpy.flipud(PerFieldDataFreeDeltas).transpose())[1],color = 'green',label = '$\Delta_2$',marker = 's',\
             markersize = 6,linewidth = 0)
ax.set_xlabel('$B_\perp$  (mT)') 
ax.set_ylabel('$\Gamma$, $\Delta$ (eV)')
ax.set_title('(b).')
ax.set_ylim([0,1.4])
ax.legend()
f.show()

f, ax = plt.subplots(1)
for k in range(46,0,-3):
    V, G, Gasym= LoadandSymmetrizePerField(k,-0.035)
    LoadFitPlotG(ax,V,G,makeParamaters(PerFieldDataFreeDeltas[46-k]),[0,3],False,(46-k)*0.1)



