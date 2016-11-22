from Engine import *
from data_load_fit import *

AllData= read_Alldata()
Tsample = AllData['Tsample']

f,axes = plt.subplots(1)
f.set_size_inches(6, 3)
img = mpimg.imread('device_scale.png')
axes.imshow(img)
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

V, G, Gasym = LoadandSymmetrizeTemp(1,0.125)
T = Tempdata.transpose()[6]/0.08
Delta01 = Tempdata.transpose()[0]
Delta02 = Tempdata.transpose()[1]

f,axes = plt.subplots(1)
f.set_size_inches(6, 3)
params = makeParamaters(Tempdata[0])
Delta0 = [params['Delta01'].value, params['Delta02'].value]
Gamma = [params['Gamma1'].value, params['Gamma2'].value]
Neff = [params['Neff1'].value, params['Neff2'].value]
T = params['T'].value
Vs = numpy.linspace(min(V),max(V),300)
I = IofV(Vs,Delta0,Gamma,Neff,T)
axes.plot(V,G ,color = 'black',label = 'data',marker = 's',markersize = 3,linewidth = 0)
axes.plot(Vs[0:len(Vs)-1],numpy.diff(I)/numpy.diff(Vs),'r',linewidth=1.5,label = 'fit')
axes.set_xlim([0, 2])
axes.set_xlabel('V (mV)')
axes.set_ylabel('G (a.u.)')


f,axes = plt.subplots(1)
f.set_size_inches(6, 3)
params = makeParamaters(Tempdata[0])
Delta0 = [params['Delta01'].value, params['Delta02'].value]
Gamma = [params['Gamma1'].value, params['Gamma2'].value]
Neff = [params['Neff1'].value, params['Neff2'].value]
T = params['T'].value
Vs = numpy.linspace(min(V),max(V),300)
I = IofV(Vs,Delta0,Gamma,Neff,T)
axes.plot(V,G ,color = 'black',label = 'data',marker = 's',markersize = 3,linewidth = 0)
axes.plot(Vs[0:len(Vs)-1],numpy.diff(I)/numpy.diff(Vs),'r',linewidth=1.5,label = 'fit')
axes.set_xlim([0, 2])
axes.set_ylim([1e-3,3])
axes.set_yscale('log')
axes.set_xlabel('V (mV)')
axes.set_ylabel('G (a.u.)')


f,axes = plt.subplots(1)
f.set_size_inches(6, 3)
axes.plot(V[1:len(V)],numpy.diff(G)/numpy.diff(V),color = 'black',label = 'data',marker = 's',\
     markersize = 2.5,linewidth = 0)
axes.plot(Vs[1:len(Vs)-1],numpy.diff(I,2)/(numpy.diff(Vs[1:len(Vs)])**2),'r',linewidth=1.5,label = 'fit')
axes.set_xlim([0, 2])
axes.set_xlabel('V (mV)')
axes.set_ylabel('dG/dV (a.u.)')

f,axes = plt.subplots(1)
f.set_size_inches(6, 5)
for k in range(1,9):
    V, G, Gasym = LoadandSymmetrizeTemp(k,0.125)
    LoadFitPlotG(axes,V,G,makeParamaters(Tempdata[k-1]),[-3,3],False,(k-1)*0.4)
    axes.annotate('T = {} K'.format(Tsample[k-1]),xy = (2,1.3+(k-1)*0.4))


# axarr[1,1].plot(T,Delta01,label = '$\Delta_L$',marker = 's',\
#              markersize = 6,linewidth = 0)
# axarr[1,1].plot(T,Delta02,label = '$\Delta_S$',marker = 's',\
#              markersize = 6,linewidth = 0)
# axarr[1,1].set_xlabel('$T_{fit} $ (K)')
# axarr[1,1].set_ylabel('$\Delta $ (eV)')
# temp = axarr[1,1].legend()