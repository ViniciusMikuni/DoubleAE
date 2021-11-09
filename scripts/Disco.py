
def DisCo(y_true, y_pred, alpha = 10.): #was 100
    #alpha determines the amount of decorrelation; 0 means no decorrelation.
    #Note that the decorrelating feature is also used for learning.
    import tensorflow as tf
    from tensorflow.keras.losses import binary_crossentropy, mse
    from tensorflow.keras import backend as K
    # first_ae = tf.gather(y_pred, [0,1,2], axis=1)
    # second_ae = tf.gather(y_pred, [3,4,5], axis=1)

    first_ae = y_pred[:,:y_pred.get_shape()[1]//2]
    second_ae = y_pred[:,y_pred.get_shape()[1]//2:]
    
    X = mse(first_ae,y_true)
    Y = mse(second_ae,y_true)
    
    LX = K.shape(X)[0]
    LY = K.shape(Y)[0]
    
    X=K.reshape(X,shape=(LX,1))
    Y=K.reshape(Y,shape=(LY,1))    
    
    ajk = K.abs(K.reshape(K.repeat(X,LX),shape=(LX,LX)) - K.transpose(X))
    bjk = K.abs(K.reshape(K.repeat(Y,LY),shape=(LY,LY)) - K.transpose(Y))

    Ajk = ajk - K.mean(ajk,axis=0)[None, :] - K.mean(ajk,axis=1)[:, None] + K.mean(ajk)
    Bjk = bjk - K.mean(bjk,axis=0)[None, :] - K.mean(bjk,axis=1)[:, None] + K.mean(bjk)

    dcor = K.sum(Ajk*Bjk) / K.sqrt(K.sum(Ajk*Ajk)*K.sum(Bjk*Bjk))    
    
    return X + Y + alpha*dcor
