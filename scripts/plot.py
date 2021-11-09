import matplotlib
from scipy.stats import pearsonr
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
#matplotlib.use('TkAgg')
import tkinter
import numpy as np
import h5py as h5
import os
from optparse import OptionParser
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
    
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

import matplotlib as mpl
from matplotlib import rc
rc('font', family='serif')
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

#
mpl.rcParams.update({'font.size': 19})
#mpl.rcParams.update({'legend.fontsize': 18})
mpl.rcParams.update({'xtick.labelsize': 18}) 
mpl.rcParams.update({'ytick.labelsize': 18}) 
mpl.rcParams.update({'axes.labelsize': 18}) 
mpl.rcParams.update({'legend.frameon': False}) 

import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style(hep.style.CMS)
hep.set_style("CMS") 




def SetFig(xlabel,ylabel):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)

    ax0.minorticks_on()
    return fig, ax0

def DisCo(X,Y):
    ajk = np.abs(np.reshape(np.repeat(X,len(X)),[len(X),len(X)]) - np.transpose(X))
    bjk = np.abs(np.reshape(np.repeat(Y,len(Y)),[len(Y),len(Y)]) - np.transpose(Y))
    Ajk = ajk - np.mean(ajk,axis=0)[None, :] - np.mean(ajk,axis=1)[:, None] + np.mean(ajk)
    Bjk = bjk - np.mean(bjk,axis=0)[None, :] - np.mean(bjk,axis=1)[:, None] + np.mean(bjk)
    dcor = np.sum(Ajk*Bjk) / np.sqrt(np.sum(Ajk*Ajk)*np.sum(Bjk*Bjk))
    return dcor

def calc_sig(data,bkg,unc=0):
    #print(poisson.pmf(k=data, mu=bkg),poisson.pmf(k=data, mu=data))
    if data/bkg<1:return 0
    return (max(data-(1+unc)*bkg,0))/np.sqrt(data)
    return np.sqrt(-2*np.log(poisson.pmf(k=data, mu=bkg)/(poisson.pmf(k=data, mu=data))))

signal_list = [r'A$\rightarrow$ 4l',r'h$^\pm\rightarrow\tau\nu$',r'h$^0\rightarrow\tau\tau$','LQ']
style = ['-','--']
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o','p','s','P','*','X','D']

name_conversion = {
    'AE1': 'Double DisCo 1',
    'AE2': 'Double DisCo 2',
    'baseline': 'Supervised',
    'AE': 'Single AE',
    'combined':'Combined double AE'
}


def FixSigEff(labels,eff=0.01):
    signals = np.unique(labels)
    bkg_idx = 0
    bkg_size = np.sum(labels==bkg_idx)
    sig_cap = int(eff*bkg_size/(1-eff))
    keep_mask = np.zeros(labels.shape[0])
    
    for signal in signals:
        if signal == bkg_idx: 
            keep_mask += labels==signal
        else:
            signal_mask = labels==signal
            if np.sum(signal_mask) > sig_cap:
                nsig=0
                for ievt, event in enumerate(signal_mask):
                    if nsig>sig_cap:
                        signal_mask[ievt]=0
                    nsig+=event
            keep_mask+=signal_mask
    return keep_mask

def CombineAE(data1,data2,label,load = False):
    
    data = np.concatenate((np.expand_dims(data1,1),np.expand_dims(data2,1)),1)
    checkpoint_file = "../weights/saved-model-combined.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_file,mode='auto',period=1,save_weights_only=True)

    inputs = Input((2, ))
    layer = Dense(64, activation='relu')(inputs)
    layer = Dense(32, activation='relu')(layer)
    outputs = Dense(1, activation='sigmoid')(layer)
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model_ = Model(inputs=inputs,outputs=outputs)
    
    model_.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    if load:
        model_.load_weights(checkpoint_file)
    else:
        fit_model = model_.fit(data, label, epochs=200,
                               callbacks=[checkpoint],
                               batch_size=512)
    pred = model_.predict(data,batch_size=1000)
    
    return pred
    fpr, tpr, _ = roc_curve(label,pred, pos_label=1)    

    return fpr,tpr

def CombineDiagAE(data1,data2,label):
    fpr = np.linspace(1e-5,0.999,5000)
    tpr = np.zeros(fpr.size)
    for ibkg,bkg_eff in enumerate(fpr):
        eff = np.sqrt(bkg_eff)
        xcut = np.quantile(data1[label==0],1- eff)
        ycut = np.quantile(data2[(label==0) & (data1> xcut)],1.0 - bkg_eff/eff)
        tpr[ibkg] = np.sum((data1[label!=0]>xcut) & (data2[label!=0]>ycut))/np.sum(label!=0)


    return tpr,fpr
    


def Plot_MSE(folder_name):
    fig,_ = SetFig("Reconstruction error","Events / bin")
    n,b,_=plt.hist(data_dict['AE'][label==0][:1000],alpha=0.5,label='SM')
    plt.hist(data_dict['AE'][label==1][:1000],alpha=0.5,bins=b,label=r'A$\rightarrow$ 4l')
    plt.yscale("log")
    plt.legend(frameon=False,fontsize=15)
    plt.savefig(os.path.join(folder_name,'mse_AE.pdf'),dpi=1200)


def Plot_2D(folder_name):
    fig,_ = SetFig("1st AE","2nd AE")

    #Faster
    bkg_corr = pearsonr(data_dict['AE1'][label==0], data_dict['AE2'][label==0])[0]
    sig_corr = pearsonr(data_dict['AE1'][label==1], data_dict['AE2'][label==1])[0]
    
    plt.scatter(data_dict['AE1'][label==1][0:500],data_dict['AE2'][label==1][0:500],label=r'A$\rightarrow$ 4l ({:.2f})'.format(sig_corr))
    plt.scatter(data_dict['AE1'][label==0][0:500],data_dict['AE2'][label==0][0:500],label="SM ({:.2f})".format(bkg_corr))
    
    plt.legend(frameon=True,fontsize=20)
    plt.savefig(os.path.join(folder_name,'disco_AE.pdf'),dpi=1200)


def Plot_ROC(folder_name):
    fig,_ = SetFig("True positive rate","1 - Fake Rate")
    
    for algo in name_conversion:    
        if 'baseline' in algo:
            fpr, tpr, _ = roc_curve(label[label<=1],data_dict[algo][label<=1][:,1], pos_label=1)    
        else:
            fpr, tpr, _ = roc_curve(label[label<=1],data_dict[algo][label<=1], pos_label=1)    
        plt.plot(tpr,1-fpr,label="{} ({:.2f})".format(name_conversion[algo],auc(tpr,1-fpr)))

    plt.legend(frameon=False,fontsize=20)
    plt.savefig(os.path.join(folder_name,'roc.pdf'),dpi=1200)


def Plot_SIC(folder_name):
    fig,_ = SetFig("True positive rate",'SIC')
    for algo in name_conversion:    
        if 'baseline' in algo:continue
        if 'baseline' in algo:
            fpr, tpr, _ = roc_curve(label[label<=1],data_dict[algo][label<=1][:,1], pos_label=1)    
        else:
            fpr, tpr, _ = roc_curve(label[label<=1],data_dict[algo][label<=1], pos_label=1)    
        finite = fpr>0
        tpr = tpr[finite]
        fpr=fpr[finite]
        plt.plot(tpr,tpr/np.sqrt(fpr),label="{} ({:.2f})".format(name_conversion[algo],auc(fpr,tpr)))

    plt.legend(frameon=False,fontsize=20)
    plt.savefig(os.path.join(folder_name,'sic.pdf'),dpi=1200)



def Plot_Closure(folder_name):
    fig,_ = SetFig("Background efficiency",r"N$_{>,>}/N$_{>,>}^{\mathrm{predicted}}$")
    #intervals = np.linspace(np.min(data_dict['AE1'][label==0]),np.max(data_dict['AE1'][label==0]),20)
    bkg_effs = np.linspace(0.005,0.05,10)
    keep_mask =FixSigEff(label,1e-3) #keep 1% signal
    xaxis=[]
    bonly = []
    sb = []
    thresh = 100
    for bkg_eff in bkg_effs:
        effs = np.linspace(bkg_eff,0.2,10)
        for eff in effs:
            xcut = np.quantile(data_dict['AE1'][label==0],1- eff)
            ycut = np.quantile(data_dict['AE2'][(label==0) & (data_dict['AE1']> xcut)],1.0 - bkg_eff/eff)
    
            for isel, sel in enumerate([label==0,(keep_mask==1) & (label<=1)]):
                A = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]>ycut))
                B = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]<ycut))
                C = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]>ycut))
                D = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]<ycut))
                if A < thresh or B < thresh or C < thresh or D < thresh: continue
                if isel==0:
                    bonly.append(1.0*A*D/(B*C))
                    xaxis.append(bkg_eff)
                else:
                    sb.append(1.0*A*D/(B*C))


    plt.scatter(xaxis,bonly,label="SM")
    plt.scatter(xaxis,sb,label=r'A$\rightarrow$ 4l + SM')
    plt.xticks(fontsize=15)
    plt.axhline(y=1.0, color='r', linestyle='-')
    #plt.ylim([0.9,1.15])
    plt.legend(frameon=False,fontsize=20)
    plt.savefig(os.path.join(folder_name,'closure.pdf'),dpi=1200)


def Plot_Closure_Multi(folder_name):
    fig,_ = SetFig("SM efficiency",r"$N_{>,>}/N_{>,>}^{\mathrm{predicted}}$")
    #intervals = np.linspace(np.min(data_dict['AE1'][label==0]),np.max(data_dict['AE1'][label==0]),20)
    bkg_effs = np.linspace(0.005,0.05,10)
    keep_mask =FixSigEff(label,1e-3) #keep 1% signal
    xaxis=[]
    closures = {}
    
    thresh = 100
    for isig, signal in enumerate(['SM']+signal_list):
        closures[signal] = []
        sel = ((label==0) | (label==isig)) & (keep_mask ==1)
    
        for bkg_eff in bkg_effs:
            effs = np.linspace(0.7*np.sqrt(bkg_eff),1.1*np.sqrt(bkg_eff),5)
            for eff in effs:
                xcut = np.quantile(data_dict['AE1'][label==0],1-eff)
                ycut = np.quantile(data_dict['AE2'][(label==0) & (data_dict['AE1']> xcut)],1.0 - bkg_eff/(eff))

                
                A = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]>ycut))
                B = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]<ycut))
                C = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]>ycut))
                D = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]<ycut))
                if A < thresh or B < thresh or C < thresh or D < thresh: continue #avoid large statistical fluctuations
                if isig==0:
                    xaxis.append(bkg_eff)
                    #print(A/(A+B+C+D))
                    # print(1.0*A*D/(B*C), bkg_eff,eff)
                    # input()

                closures[signal].append(1.0*A*D/(B*C))


        plt.scatter(xaxis,closures[signal],label="{}".format(signal),color = color_list[isig],marker=markers[isig])
        # plt.scatter(xaxis,sb,label=r'A$\rightarrow$ 4l + SM')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # y_loc, _ = plt.yticks()
        # y_update = ['%.2f' % y for y in y_loc]
        # plt.yticks(y_loc, y_update)

        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.ylim([0.95,1.10])
        plt.legend(frameon=False,fontsize=20,ncol=2)
    plt.savefig(os.path.join(folder_name,'closure_multi.pdf'),dpi=1200)


def Plot_ROC_Multi(folder_name):

    fig,ax = SetFig("True positive rate","1 - Fake Rate")
    name_conversion = {
        'AE1': 'Double Disco 1',
        'AE2': 'Double Disco 2',   
    }
    
    for isig, signal in enumerate(signal_list):
        sel = (label==0) | (label==isig+1)
        for ialgo, algo in enumerate(name_conversion):    
            fpr, tpr, _ = roc_curve(label[sel],data_dict[algo][sel], pos_label=isig+1)    
            #plt.plot(tpr,1-fpr,style[ialgo],color = color_list[isig],label="{} ({:.2f})".format(signal,auc(tpr,1-fpr)))
            line,=plt.plot(tpr,1-fpr,style[ialgo],color = color_list[isig])
            if ialgo ==0:
                line.set_label(signal)
            
        tpr,fpr = CombineDiagAE(data_dict['AE1'][sel],data_dict['AE2'][sel],label[sel])
        #plt.plot(tpr,1-fpr,':',color = color_list[isig],label="{} ({:.2f})".format(signal,auc(tpr,1-fpr)))      
        plt.plot(tpr,1-fpr,':',color = color_list[isig])      
    leg1=plt.legend(frameon=False,fontsize=14,ncol=2,loc='center left')
    line = mlines.Line2D([], [], color='black', label='Autoencoder 1')
    dash = mlines.Line2D([], [], linestyle='--',color='black', label='Autoencoder 2')
    dot = mlines.Line2D([], [], linestyle=':',color='black', label='Combined autoencoder')
    plt.legend(handles=[line,dash,dot],frameon=False,fontsize=14,loc='lower left')
    ax.add_artist(leg1)
    plt.savefig(os.path.join(folder_name,'roc_comparison.pdf'),bbox_inches='tight',dpi=1200)



def Plot_SIC_Multi(folder_name):            
    name_conversion = {
        'AE1': 'Double Disco 1',
        'AE2': 'Double Disco 2',   
    }

    fig,ax = SetFig("True positive rate","SIC")
    for isig, signal in enumerate(signal_list):
        sel = (label==0) | (label==isig+1)
        for ialgo, algo in enumerate(name_conversion):    
            fpr, tpr, _ = roc_curve(label[sel],data_dict[algo][sel], pos_label=isig+1)    
            line,=plt.plot(tpr,tpr/(fpr**0.5),style[ialgo],color = color_list[isig])
            if ialgo ==0:
                line.set_label(signal)
        tpr,fpr = CombineDiagAE(data_dict['AE1'][sel],data_dict['AE2'][sel],label[sel])
        #plt.plot(tpr,tpr/(fpr**0.5),':',color = color_list[isig],label="{} ({:.2f})".format(signal,auc(tpr,1-fpr)))    
        plt.plot(tpr,tpr/(fpr**0.5),':',color = color_list[isig])      
    leg1 = plt.legend(frameon=False,fontsize=14,ncol=2,loc=(0.1,0.8))
    line = mlines.Line2D([], [], color='black', label='Autoencoder 1')
    dash = mlines.Line2D([], [], linestyle='--',color='black', label='Autoencoder 2')
    dot = mlines.Line2D([], [], linestyle=':',color='black', label='Combined autoencoder')
    plt.legend(handles=[line,dash,dot],frameon=False,fontsize=14,loc='upper right')
    ax.add_artist(leg1)
    plt.savefig(os.path.join(folder_name,'sic_comparison.pdf'),bbox_inches='tight',dpi=1200)


def Plot_SIC_2D(folder_name):            
    name_conversion = {
        'AE1': 'Double Disco 1',
        'AE2': 'Double Disco 2',   
    }

    thresholds = np.linspace(0,0.02,10)
    sic = {}
    cmap = plt.get_cmap('PiYG')
    for isig, signal in enumerate(signal_list):
        fig,ax = SetFig("Autoencoder 1 loss","Autoencoder 2 loss")
        sic[signal] = np.zeros((len(thresholds),len(thresholds)))
        for x,xpoint in enumerate(thresholds):
            for y,ypoint in enumerate(thresholds):
                sig = 1.0*np.sum((data_dict['AE1'][label==isig+1]>xpoint) & (data_dict['AE2'][label==isig+1]>ypoint))/np.sum(label==isig+1)
                bkg = 1.0*np.sum((data_dict['AE1'][label==0]>xpoint) & (data_dict['AE2'][label==0]>ypoint))/np.sum(label==0)
                sic[signal][x,y] = sig/(bkg**0.5)


        im = ax.pcolormesh(thresholds, thresholds, sic[signal], cmap=cmap)
        fig.colorbar(im, ax=ax,label='SIC')
        bar = ax.set_title(signal)
        plt.savefig(os.path.join(folder_name,'sic_2d_{}.pdf'.format(signal)),bbox_inches='tight',dpi=1200)



def Plot_Significance(folder_name):
    from scipy.stats import poisson
    sig_eff = 1e-3
    keep_mask =FixSigEff(label,sig_eff)
    fig,ax = SetFig("SM efficiency","Significance")
        

    bkg_effs = np.linspace(0.005,0.05,10)
    significances = {}
    significances_abcd = {}
    
    for ibkg, bkg_eff in enumerate(bkg_effs):
        eff = np.sqrt(bkg_eff)
        xcut = np.quantile(data_dict['AE1'][label==0],1- eff)
        ycut = np.quantile(data_dict['AE2'][(label==0) & (data_dict['AE1']> xcut)],1.0 - bkg_eff/eff)
        bkg = 1.0*np.sum((data_dict['AE1'][label==0] > xcut) & (data_dict['AE2'][label==0] > ycut))

        for isig, signal in enumerate(['SM'] + signal_list):
            if ibkg==0: 
                significances[signal] = []
                significances_abcd[signal] = []
            sel = ((label==0) | (label==isig)) & (keep_mask ==1)            
            #data = 1.0*np.sum((data_dict['AE1'][sel] > xcut) & (data_dict['AE2'][sel] > ycut))
        
            A = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]>ycut))
            B = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]<ycut))
            C = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]>ycut))
            D = np.sum((data_dict['AE1'][sel]<xcut)*(data_dict['AE2'][sel]<ycut))

            significances[signal].append(calc_sig(A,bkg))
            significances_abcd[signal].append(calc_sig(1.0*A,1.0*(B*C)/D,.05))
            #print(bkg_eff,(B*C)/(A*D))
    maxsig = 0
    for isig, signal in enumerate(signal_list+['SM']):
        plt.plot(bkg_effs,significances[signal],'--',color=color_list[isig])
        plt.plot(bkg_effs,significances_abcd[signal],color=color_list[isig],label="{}".format(signal))
        if maxsig < np.max(significances_abcd[signal]):
            maxsig = np.max(significances_abcd[signal])

    plt.ylim([0,1.3*maxsig])
    leg1=plt.legend(frameon=False,fontsize=14,ncol=2)
    line = mlines.Line2D([], [], color='black', label='ABCD prediction')
    dash = mlines.Line2D([], [], linestyle='--',color='black', label='True background')
    plt.legend(handles=[line,dash],frameon=False,fontsize=14,loc='upper left')
    ax.add_artist(leg1)
    
    plt.savefig(os.path.join(folder_name,'sig_{}.pdf'.format(sig_eff)),bbox_inches='tight',dpi=1200)



def Plot_Significance_comp(folder_name):
    from scipy.stats import poisson
    sig_eff = 1e-3
    keep_mask =FixSigEff(label,sig_eff)
    fig,ax = SetFig("SM efficiency","Significance")
        

    bkg_effs = np.linspace(0.005,0.05,10)
    significances = {}
    significances_single = {}

    
    for ibkg, bkg_eff in enumerate(bkg_effs):
        eff = np.sqrt(bkg_eff)
        xcut = np.quantile(data_dict['AE1'][label==0],1- eff)
        ycut = np.quantile(data_dict['AE2'][(label==0) & (data_dict['AE1']> xcut)],1.0 - bkg_eff/eff)
        cut = np.quantile(data_dict['AE'][(label==0)],1.0 - bkg_eff)
        bkg = 1.0*np.sum((data_dict['AE1'][label==0] > xcut) & (data_dict['AE2'][label==0] > ycut))
        

        for isig, signal in enumerate(signal_list):
            if ibkg==0: 
                significances[signal] = []
                significances_single[signal] = []
            sel = ((label==0) | (label==isig+1)) & (keep_mask ==1)            

        
            sig_double = np.sum((data_dict['AE1'][sel]>xcut)*(data_dict['AE2'][sel]>ycut))
            sig_single = np.sum((data_dict['AE'][sel]>cut))

            significances[signal].append(calc_sig(sig_double,bkg))
            significances_single[signal].append(calc_sig(sig_single,bkg))

    maxsig = 0
    for isig, signal in enumerate(signal_list):
        plt.plot(bkg_effs,significances[signal],color=color_list[isig],label="{}".format(signal))
        plt.plot(bkg_effs,significances_single[signal],':',color=color_list[isig+1])
        #,label="{}".format(signal)
        if maxsig < np.max(significances_single[signal]):
            maxsig = np.max(significances_single[signal])

    plt.ylim([0,1.1*maxsig])    
    leg1 = plt.legend(frameon=False,fontsize=14,ncol=2,loc='lower right')
    line = mlines.Line2D([], [], color='black', label='Double autoencoder')
    dash = mlines.Line2D([], [], linestyle=':',color='black', label='Single autoencoder')
    plt.legend(handles=[line,dash],frameon=False,fontsize=14,loc='lower left')
    ax.add_artist(leg1)
    plt.savefig(os.path.join(folder_name,'sig_comp_{}.pdf'.format(sig_eff)),bbox_inches='tight',dpi=1200)



if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default="./", help="Folder containing input files")
    parser.add_option("--file", type="string", default="AEDisco_10.h5", help="Name of input file")
    (flags, args) = parser.parse_args()
    
    data_dict = {}
    with h5.File(os.path.join(flags.folder, flags.file),"r") as f:
        for key in f.keys():
            data_dict[key] = f[key][:]

    label = data_dict['label']
    print(np.sum(label==0))
    data_dict['combined'] = CombineAE(data_dict['AE1'],data_dict['AE2'],label,True)
    version = flags.file.split(".h5")[0]
    folder_name = os.path.join("..","plots",version)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #Uncomment the plot routines wanted
    plot_list = {
        # 'plot_mse':Plot_MSE,
        # 'plot_2d':Plot_2D,
        # 'plot_roc':Plot_ROC,
        # 'plot_sic':Plot_SIC,
        #'plot abcd closure':Plot_Closure,
        'plot abcd closure all':Plot_Closure_Multi,
        # 'plot multi roc':Plot_ROC_Multi,
        # 'plot multi sic':Plot_SIC_Multi,
        # 'plot 2d sic':Plot_SIC_2D,
        #'plot significance':Plot_Significance,
        # 'plot significance comparison':Plot_Significance_comp,
    }

    for func in plot_list:
        print("Calling: "+func)
        plot_list[func](folder_name)
