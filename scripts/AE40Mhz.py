import numpy as np
import h5py as h5
import os
from optparse import OptionParser
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense, Flatten, Input, Dropout, BatchNormalization, Activation
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Disco import DisCo
from keras_flops import get_flops

import sys
os.environ['CUDA_VISIBLE_DEVICES']="1"



NLAYERS = 4
LAYERSIZE = [256,128,64,32]
ENCODESIZE = 5
EPOCHS = 400
LR = 0.0001 #learning rate
opt = Adam(learning_rate=LR)


def AE(NFEAT):
    inputs = Input((NFEAT, ))
    layer = Dense(LAYERSIZE[0], activation='relu', use_bias=False)(inputs)
    #Encoder
    for il in range(1,NLAYERS):
        layer = Dense(LAYERSIZE[il], activation='linear', use_bias=False)(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

    layer = Dense(ENCODESIZE, activation='linear', use_bias=False)(layer)
    #Decoder
    for il in range(NLAYERS):
        layer = Dense(LAYERSIZE[NLAYERS-il-1], activation='linear', use_bias=False)(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
    #layer = Dropout(0.25)(layer)
    outputs = Dense(NFEAT, activation='linear', use_bias=False)(layer)

    return inputs,outputs

def baseline(NFEAT,NCAT):
    inputs = Input((NFEAT, ))
    layer = Dense(128, activation='relu')(inputs)
    for il in range(NLAYERS):
        layer = Dense(128, activation='relu')(layer)

    outputs = Dense(NCAT, activation='softmax')(layer)
    return inputs,outputs




def TrainSupervisedModel(data,path,ncat,load,callbacks):
    checkpoint = ModelCheckpoint(path,save_best_only=True,mode='auto',period=1,save_weights_only=True)

    inputs,outputs = baseline(data['X_train'].shape[1],ncat)
    model_baseline = Model(inputs=inputs,outputs=outputs)
    model_baseline.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    print("Starting supervised training for benchmarking")
    if load:
        model_baseline.load_weights(path)
    else:
        hist_baseline = model_baseline.fit(data['X_train'], LabelBinarizer().fit_transform(data['Y_train']), epochs=EPOCHS, 
                                           callbacks=[callbacks,checkpoint],
                                           batch_size=1024,validation_data=(data['X_val'], LabelBinarizer().fit_transform(data['Y_val'])))

    return model_baseline.predict(data['X_test'],batch_size=1000)

def TrainDoubleAE(data,path,load,callbacks):
    checkpoint = ModelCheckpoint(path,save_best_only=True,mode='auto',period=1,save_weights_only=True)
    inputs1,outputs1 = AE(data['X_train'].shape[1])
    inputs2,outputs2 = AE(data['X_train'].shape[1])
    mymodel = Model([inputs1, inputs2], tf.keras.layers.concatenate([outputs1, outputs2]))
    mymodel.compile(loss=DisCo, optimizer=opt, metrics=['accuracy'])
    print("Starting double AE")
    if load:
        mymodel.load_weights(path)
        trainableParams = np.sum([np.prod(v.get_shape()) for v in mymodel.trainable_weights])
        print("Number of trainable parameters: {}".format(trainableParams))
        print("FLOPS: {}".format(get_flops(mymodel,batch_size=1)))
    else:
        hist_ae = mymodel.fit([data['X_train'][(data['Y_train']==0)],data['X_train'][(data['Y_train']==0)]],
                              data['X_train'][(data['Y_train']==0)],
                              epochs=EPOCHS, batch_size=10000,
                              callbacks=[callbacks,checkpoint],                        
                              validation_data=([data['X_val'][(data['Y_val']==0)],data['X_val'][(data['Y_val']==0)]], data['X_val'][(data['Y_val']==0)]))


    mypreds = mymodel.predict([data['X_test'],data['X_test']],batch_size=10000)
                              
    mypreds1 = mypreds[:,:data['X_train'].shape[1]]
    mypreds2 = mypreds[:,data['X_train'].shape[1]:]
    out_dict = {}
    
    for pred,name in zip([mypreds1,mypreds2],['AE1','AE2']):
        mses = np.mean(np.square(pred - data['X_test']),-1)
        print("Signal loss", np.mean(mses[data['Y_test']==1],-1))
        print("Background loss", np.mean(mses[data['Y_test']==0],-1))
        fpr, tpr, _ = roc_curve(data['Y_test'],mses, pos_label=1)    
        print("AE AUC: {}".format(auc(fpr, tpr)))
        out_dict[name] = mses
    return out_dict
                    
def TrainSingleAE(data,path,load,callbacks):
    inputs,outputs = AE(data['X_train'].shape[1])
    mymodel = Model(inputs=inputs,outputs=outputs)
    mymodel.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(path,save_best_only=True,mode='auto',period=1,save_weights_only=True)
    print("Starting single AE")

    if load:
        mymodel.load_weights(path)
    else:
        hist_ae = mymodel.fit(data['X_train'][(data['Y_train']==0)], 
                              data['X_train'][(data['Y_train']==0)],
                              epochs=EPOCHS, 
                              callbacks=[callbacks,checkpoint],
                              batch_size=1024,
                              validation_data=(data['X_val'][(data['Y_val']==0)], data['X_val'][(data['Y_val']==0)]))

    single_AE = mymodel.predict(data['X_test'],batch_size=1000)


    return np.mean(np.square(single_AE - data['X_test']),-1)



if __name__ == "__main__":
    
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--folder", type="string", default="/clusterfs/ml4hep/vmikuni/data", help="Folder containing input files")
    parser.add_option("--file", type="string", default="SM_dataset.h5", help="Name of SM input file")
    parser.add_option("--out", type="string", default="AEDisco.h5", help="Name of output file")

    parser.add_option("--double", action="store_true", default=False, help="Train double AE")
    parser.add_option("--single", action="store_true", default=False, help="Train single AE")
    parser.add_option("--supervised", action="store_true", default=False, help="Train supervised classifier")
    parser.add_option("--all", action="store_true", default=False, help="Perform multiple trainings")
    parser.add_option("--load", action="store_true", default=False, help="Load trained models")
    parser.add_option("-n", type = 'int', default=-1, help="Number of training events to use")
    (flags, args) = parser.parse_args()

    SM_file = h5.File(os.path.join(flags.folder, flags.file),"r")

    sig_list = ['Ato4l_lepFilter_13TeV_dataset.h5','hChToTauNu_13TeV_PU20_dataset.h5','hToTauTau_13TeV_PU20_dataset.h5','leptoquark_LOWMASS_lepFilter_13TeV_dataset.h5']
    sig_files = []
    for sig in sig_list:
        sig_files.append(h5.File(os.path.join(flags.folder, sig),"r"))
        
    if flags.n<0:
        train_size = np.array(SM_file['X_train']).shape[0]
    else:
        train_size = flags.n

    test_size = np.array(SM_file['X_test']).shape[0]
    val_size = np.array(SM_file['X_val']).shape[0]
    sig_size = min(train_size,np.array(sig_files[0]['Data'][:]).shape[0])

    signals = []
    signals_labels = []
    for isig, sig in enumerate(sig_files):
        if isig==0:
            signals = sig['Data'][:sig_size]
            signals_labels = (isig+1)*np.ones(sig_size)
        else:
            signals = np.concatenate([signals,sig['Data'][:sig_size]],0)
            signals_labels = np.concatenate([signals_labels,(isig+1)*np.ones(sig_size)],0)
    
    feed_dict ={ 
        'X_train' :np.concatenate([np.array(SM_file['X_train'][:train_size]),signals]),
        'X_test' : np.concatenate([np.array(SM_file['X_test'][:test_size]),signals]),
        'X_val' : np.concatenate([np.array(SM_file['X_val'][:val_size]),signals]),

        'Y_train' : np.concatenate([np.zeros(train_size),signals_labels]),
        'Y_test' : np.concatenate([np.zeros(test_size),signals_labels]),
        'Y_val' : np.concatenate([np.zeros(val_size),signals_labels]),
    }


    scaler = MinMaxScaler()

    #Normalize inputs
    scaler.fit(feed_dict['X_train'][feed_dict['Y_train']==0])
    feed_dict['X_train'] = scaler.transform(feed_dict['X_train'])
    feed_dict['X_test'] = scaler.transform(feed_dict['X_test'])
    feed_dict['X_val'] = scaler.transform(feed_dict['X_val'])
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)]
    
    if flags.supervised or flags.all:        
        pred_baseline = TrainSupervisedModel(
            feed_dict,path = "../weights/saved-model-supervised.hdf5",
            ncat=len(sig_list)+1,
            load = flags.load,callbacks=callbacks)
    if flags.double or flags.all:
        pred_double = TrainDoubleAE(
            feed_dict,path="../weights/saved-model-doubleAE_10_v1.hdf5",
            load = flags.load,callbacks=callbacks)
    if flags.single or flags.all:
        pred_single = TrainSingleAE(
            feed_dict,path = "../weights/saved-model-singleAE.hdf5",
            load = flags.load,callbacks=callbacks)

    if flags.all:
        if not os.path.exists('../h5'):
            os.makedirs('../h5')

        with h5.File(os.path.join('../h5',flags.out), "w") as fh5:
            dset = fh5.create_dataset("AE1", data=pred_double['AE1'])
            dset = fh5.create_dataset("AE2", data=pred_double['AE2'])
            dset = fh5.create_dataset("baseline", data=pred_baseline)
            dset = fh5.create_dataset("AE", data=pred_single)        
            dset = fh5.create_dataset("label", data=feed_dict['Y_test'])
