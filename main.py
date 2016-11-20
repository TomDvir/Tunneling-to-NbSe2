from Engine import *
from data_load_fit import *

AllData = read_Alldata()

Tempdata = AllData['TempData']
k = 1
V, G, Gasym = LoadandSymmetrizeTemp(k,0.14)
# fig = LoadFitPlot(V,G,makeParamaters(Tempdata[k-1]),[0,2],0.0001)


Es = numpy.linspace(0,2,100)
Ds = numpy.array([1 + 0.01j,0.5+0.01j]).real
Ns = AG_DOS(Es[0],Ds,0.1)
for k in range(1,len(Es)):
    Ns = numpy.vstack((Ns,AG_DOS(Es[k],Ds,0)))
plt.plot(Es,(Ns.transpose())[0])
plt.plot(Es,(Ns.transpose())[1])


Ns = AG_DOS(Es[0],Ds,0.1)
for k in range(1,len(Es)):
    Ns = numpy.vstack((Ns,AG_DOS(Es[k],Ds,0.01)))

plt.plot(Es,(Ns.transpose())[0])
plt.plot(Es,(Ns.transpose())[1])

plt.show()