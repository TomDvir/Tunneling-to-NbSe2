## Data manipulation and fit functions
from Engine import IofV
import scipy
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from lmfit import Parameters
import matplotlib.image as mpimg



def LoadandSymmetrizeTemp(index, shift):
    """
    LoadandSymmetrizeTemp(index, shift)
    return symmetrize(V, G, shift)
    """
    
    bigdata = scipy.io.loadmat('summary-data.mat')
    tempdata = bigdata['Temperature']
    data = numpy.array(tempdata[0][index - 1][1][0][0][0]).transpose()
    V = data[1]
    G = data[2]

    return symmetrize(V, G, shift)


def LoadandSymmetrizePerField(index, shift):
    bigdata = scipy.io.loadmat('summary-data.mat')
    PerpFieldData = bigdata['PerpField']
    data = numpy.array(PerpFieldData[0][index - 1][0]).transpose()
    V = data[1]
    G = data[2]
    B = 4 * (15 - (index - 1) / 3)

    return symmetrize(V, G, shift)


def LoadandSymmetrizeParField(index, shift, sym=True):
    bigdata = scipy.io.loadmat('summary-data.mat')
    PerpFieldData = bigdata['ParFieldwComp']
    data = numpy.array(PerpFieldData[0][index - 1][0]).transpose()
    V = data[1]
    G = data[2]
    B = bigdata['ParFieldwComp'][0][index - 1][1][0][0]
    if sym:
        V, G, Gasym = symmetrize(V, G, shift)
    else:
        Gasym = V * 0
    return V, G, Gasym, B


def symmetrize(V, G, shift):
    bigdata = scipy.io.loadmat('summary-data.mat')
    tempdata = bigdata['Temperature']

    #     high temp data
    data = numpy.array(tempdata[0][9][1][0][0][0]).transpose()
    Vhigh = data[1]
    Ghigh = data[2]

    #     dividing the low T by the High T
    GhighI = interp1d(Vhigh, Ghigh)(V)
    Gnew = G / GhighI

    #     shifting the voltgae to correct for zero drift
    V = V + shift

    Gflip = interp1d(V, numpy.flipud(Gnew))(V)
    Gsym = (Gnew + Gflip) / 2;
    Gasym = (Gnew - Gflip) / 2;

    return V, Gsym, Gasym


# build 3D data arrays for parallel field:
def LoadAllParField(shift, sym=True):
    #     returns Vs, Bs, Gs, Vsm1, Bsm1, dGdVs, ParB
    k = 1
    Vs, Gs, Gasym, B = LoadandSymmetrizeParField(1, shift, sym)
    V = Vs
    Vsm1 = V[0:len(V) - 1]
    G = Gs
    dGdVs = numpy.diff(G) / numpy.diff(V)
    Bs = numpy.ones(len(V)) * B
    Bsm1 = numpy.ones(len(V) - 1) * B
    for k in range(2, 35):
        V, G, Gasym, B = LoadandSymmetrizeParField(k, shift, sym)
        dGdV = numpy.diff(G) / numpy.diff(V)

        Vs = numpy.vstack((Vs, V))
        Vsm1 = numpy.vstack((Vsm1, V[0:len(V) - 1]))
        Gs = numpy.vstack((Gs, G))
        dGdVs = numpy.vstack((dGdVs, dGdV))
        Bs = numpy.vstack((Bs, numpy.ones(len(V)) * B))
        Bsm1 = numpy.vstack((Bsm1, numpy.ones(len(V) - 1) * B))
    ParB = Bs.transpose()[0]

    return Vs, Bs, Gs, Vsm1, Bsm1, dGdVs, ParB


def LoadFitPlot(V, G, params, Vlim, alpha=0):
    Delta0 = [params['Delta01'].value, params['Delta02'].value]
    Gamma = [params['Gamma1'].value, params['Gamma2'].value]
    Neff = [params['Neff1'].value, params['Neff2'].value]
    T = params['T'].value
    if 'dynes' in params:
        dynes = params['dynes'].value
    else:
        dynes = 0

#    print(readParameters(params))

    Vs = numpy.linspace(min(V), max(V), 300)
    I = IofV(Vs, Delta0, Gamma, Neff, T, alpha,dynes)

    fig = plt.figure(figsize=(8, 8), dpi=80)
    plt.subplot(2, 1, 1)

    plt.plot(V, G, color='black', label='data', marker='s', markersize=2.5, linewidth=0)
    plt.plot(Vs[0:len(Vs) - 1], numpy.diff(I) / numpy.diff(Vs), 'r', linewidth=1.5, label='fit')
    plt.xlim(Vlim[0], Vlim[1])
    plt.ylim(0, 2)
    plt.legend()
    plt.xlabel('V (mV)')
    plt.ylabel('G (a.u.)')

    plt.subplot(2, 1, 2)
    plt.plot(V[1:len(V)], numpy.diff(G) / numpy.diff(V), color='black', label='data', marker='s', \
             markersize=2.5, linewidth=0)
    plt.plot(Vs[1:len(Vs) - 1], numpy.diff(I, 2) / (numpy.diff(Vs[1:len(Vs)]) ** 2), 'r', linewidth=1.5, label='fit')
    plt.xlim(Vlim[0], Vlim[1])
    plt.legend()
    plt.xlabel('V (mV)')
    plt.ylabel('dG/dV (a.u.)')

    plt.show()

    return fig
def LoadFitPlotG(axes,V,G,params,Vlim,dGdV = False,shift = 0.,Logscale = False):
        
    Delta0 = [params['Delta01'].value, params['Delta02'].value]
    Gamma = [params['Gamma1'].value, params['Gamma2'].value]
    Neff = [params['Neff1'].value, params['Neff2'].value]
    T = params['T'].value
    
    Vs = numpy.linspace(min(V),max(V),300)
    I = IofV(Vs,Delta0,Gamma,Neff,T)
    
    axes.plot(V,G + shift ,color = 'black',label = 'data',marker = 's',markersize = 3,linewidth = 0)
    axes.plot(Vs[0:len(Vs)-1],shift + numpy.diff(I)/numpy.diff(Vs),'r',linewidth=1.5,label = 'fit')
    axes.set_xlim([Vlim[0], Vlim[1]])
#     axes.set_ylim([0,2])
#     axes.legend()
    axes.set_xlabel('V (mV)')
    axes.set_ylabel('G (a.u.)')
    
    if dGdV == True:
        t_axes = inset_axes(axes, width="30%", height=1.3,loc=3)
        t_axes.plot(V[1:len(V)],numpy.diff(G)/numpy.diff(V),color = 'black',label = 'data',marker = 's',\
             markersize = 2.5,linewidth = 0)
        t_axes.plot(Vs[1:len(Vs)-1],numpy.diff(I,2)/(numpy.diff(Vs[1:len(Vs)])**2),'r',linewidth=1.5,label = 'fit')
        t_axes.set_xlim([Vlim[0], Vlim[1]])
        t_axes.get_xaxis().set_ticks([])
        t_axes.get_yaxis().set_ticks([])

    if Logscale == True:
        t_axes = inset_axes(axes, width="30%", height=1.3,loc=4)
        t_axes.plot(V,G,color = 'black',label = 'data',marker = 's',\
             markersize = 2.5,linewidth = 0)
        t_axes.plot(Vs[0:len(Vs)-1],shift + numpy.diff(I)/numpy.diff(Vs),'r',linewidth=1.5,label = 'fit')
        t_axes.set_xlim([Vlim[0], Vlim[1]])
        t_axes.get_xaxis().set_ticks([])
        t_axes.get_yaxis().set_ticks([])
        t_axes.set_ylim([1e-3,3])
        t_axes.set_yscale('log')
        
def LoadFitPlotdGdV(axes,V,G,params,Vlim,shift = 0.):
        
    Delta0 = [params['Delta01'].value, params['Delta02'].value]
    Gamma = [params['Gamma1'].value, params['Gamma2'].value]
    Neff = [params['Neff1'].value, params['Neff2'].value]
    T = params['T'].value
    
    Vs = numpy.linspace(min(V),max(V),300)
    I = IofV(Vs,Delta0,Gamma,Neff,T)
    
    
    axes.plot(V[1:len(V)],numpy.diff(G)/numpy.diff(V)+ shift,color = 'black',label = 'data',marker = 's',\
             markersize = 2.5,linewidth = 0)
    axes.plot(Vs[1:len(Vs)-1],numpy.diff(I,2)/(numpy.diff(Vs[1:len(Vs)])**2)+ shift,'r',linewidth=1.5,label = 'fit')
    axes.set_xlim([Vlim[0], Vlim[1]])
#     axes.set_ylim([0,2])
#     axes.legend()
    axes.set_xlabel('V (mV)')
    axes.set_ylabel('dG/dV (a.u.)')



def Goodness(params, V, G, Gerr, Vlim):
    Delta0 = [params['Delta01'].value, params['Delta02'].value]
    Gamma = [params['Gamma1'].value, params['Gamma2'].value]
    Neff = [params['Neff1'].value, params['Neff2'].value]
    T = params['T'].value

    Vfit = numpy.linspace(-6, 6, 500)
    I = IofV(Vfit, Delta0, Gamma, Neff, T)
    Gfit = numpy.diff(I) / numpy.diff(Vfit)

    #  trimming:
    Vmin = Vlim[0]
    Vmax = Vlim[1]
    MinIndex = numpy.argmin(abs(V - Vmin))
    MaxIndex = numpy.argmin(abs(V - Vmax))
    V = V[MinIndex:MaxIndex]
    G = G[MinIndex:MaxIndex]
    Gerr = Gerr[MinIndex:MaxIndex]

    GforChi = interp1d(Vfit[1:len(Vfit)], Gfit)(V)

    return (GforChi - G) / Gerr


def Goodness2(params, V, G, Gerr, Vlim):
    Delta0 = [params['Delta01'].value, params['Delta02'].value]
    Gamma = [params['Gamma1'].value, params['Gamma2'].value]
    Neff = [params['Neff1'].value, params['Neff2'].value]
    T = params['T'].value

    Vfit = numpy.linspace(-6, 6, 500)
    I = IofV(Vfit, Delta0, Gamma, Neff, T)
    Gfit = numpy.diff(I) / numpy.diff(Vfit)

    #  trimming:
    Vmin = Vlim[0]
    Vmax = Vlim[1]
    MinIndex = numpy.argmin(abs(V - Vmin))
    MaxIndex = numpy.argmin(abs(V - Vmax))
    V = V[MinIndex:MaxIndex]
    G = G[MinIndex:MaxIndex]
    Gerr = Gerr[MinIndex:MaxIndex]

    GforChi = interp1d(Vfit[1:len(Vfit)], Gfit)(V)

    return (numpy.diff(GforChi) - numpy.diff(G)) / Gerr[1:len(G)]


def VortexFit(Vparams, V, G, Gerr, Vlim):
    a = Vparams['a'].value
    b = Vparams['b'].value
    c = Vparams['c'].value
    V0 = Vparams['V0'].value

    Vmin = Vlim[0]
    Vmax = Vlim[1]
    MinIndex = numpy.argmin(abs(V - Vmin))
    MaxIndex = numpy.argmin(abs(V - Vmax))
    V = V[MinIndex:MaxIndex]
    G = G[MinIndex:MaxIndex]
    Gerr = Gerr[MinIndex:MaxIndex]

    Gfit = a * (V - V0) ** 2 + b * abs((V - V0)) + c

    return (Gfit - G) / Gerr


def GiveGs(V, Vparams):
    a = Vparams[0]
    b = Vparams[1]
    c = Vparams[2]
    V0 = Vparams[3]
    return (a * (V - V0) ** 2 + b * abs((V - V0)) + c).transpose()


## Assisting functions for using the lmfit package

def makeParamaters(dataline):
    params = Parameters()
    params.add('Delta01', value=dataline[0], min=0.0, vary=True)
    params.add('Delta02', value=dataline[1], min=0.0, vary=True)
    params.add('Gamma1', value=dataline[2], min=0.0, vary=True)
    params.add('Gamma2', value=dataline[3], min=0.0, vary=True)
    params.add('Neff1', value=dataline[4], vary=False)
    params.add('Neff2', value=dataline[5], min=0.0, vary=True)
    params.add('T', value=dataline[6], min=0.0, vary=True)
    if len(dataline)>7:
        params.add('dynes', value=dataline[7], min=0.0, vary=False)
    return params


def readParameters(params):
    dataline = numpy.ndarray(7)
    dataline[0] = params['Delta01'].value
    dataline[1] = params['Delta02'].value
    dataline[2] = params['Gamma1'].value
    dataline[3] = params['Gamma2'].value
    dataline[4] = params['Neff1'].value
    dataline[5] = params['Neff2'].value
    dataline[6] = params['T'].value
    if 'dynes' in params:
        dataline[7] = params['dynes'].value
    return dataline


def makeVParamaters(dataline):
    params = Parameters()
    params.add('a', value=dataline[0], min=0.0, vary=False)
    params.add('b', value=dataline[1], min=0.0, vary=True)
    params.add('c', value=dataline[2], min=0.0, vary=True)
    params.add('V0', value=dataline[3], vary=True)
    return params


def readVParameters(params):
    dataline = numpy.ndarray(4)
    dataline[0] = params['a'].value
    dataline[1] = params['b'].value
    dataline[2] = params['c'].value
    dataline[3] = params['V0'].value
    return dataline


# def update_Alldata(AllData,paramname,paramvalue):
    # AllData = {}
    #
    # AllData['TempData'] = Tempdata
    # AllData['TempDataRaw'] = TempdataRaw
    # AllData['Tsample'] = Tsample
    # AllData['FitNoDelta2'] = FitNoDelta2
    # AllData['FitNoDelta2Raw'] = FitNoDelta2Raw
    # AllData['FitFixedT200mK'] = FitFixedT200mK
    # AllData['FitFixedT200mKRaw'] = FitFixedT200mKRaw
    #
    # AllData['PerFieldData'] = PerFieldData
    # AllData['PerFieldDataRaw'] = PerFielddataRaw
    # AllData['PerFieldLinFit_0p1'] = PerFieldLinFit_0p1
    # AllData['PerFieldLinFit_0p2'] = PerFieldLinFit_0p2
    # AllData['PerFieldLinFit_0p3'] = PerFieldLinFit_0p3
    # AllData['PerB'] = PerB
    # AllData['PerFieldDataFreeDeltas'] = PerFieldDataFreeDeltas
    # AllData['PerFielddataFreeDeltasRaw'] = PerFielddataFreeDeltasRaw
    # AllData['PerFieldDataFixedGammas'] = PerFieldDataFixedGammas
    # AllData['PerFielddataFixedGammasRaw'] = PerFielddataFixedGammasRaw
    #
    # AllData['ParB'] = ParB
    # AllData['ParFieldLinFit_0p1'] = ParFieldLinFit_0p1
    # AllData['ParFieldLinFit_0p2'] = ParFieldLinFit_0p2
    # AllData['ParFieldLinFit_0p3'] = ParFieldLinFit_0p3

    # numpy.save('AnalysisData',AllData)

def read_Alldata():
    AllData = numpy.load('AnalysisData.npy')
    return AllData[()]