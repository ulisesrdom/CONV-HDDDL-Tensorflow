"""
Hierarchical Discriminative Deep Dictionary Learning
Application : (Cropped) AR Faces
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
              Oscar Dalmau               - CIMAT
----------------------------------------------------------
------           Auxiliary functions file            -----

"""
import numpy as np
import os,sys
from PIL import Image
from time import time

# Auxiliary functions (data reading) --------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# READ AR database images -----------------------------------------
# Source: 
#         A. M. Martinez and R. Benavente. The AR Face Database.
#         CVC Technical Report No. 24, 1998.
# **Note: For comparisons make sure to get the cropped version
#         of this database, which contains a subset from the
#         original database, with 1300 images of 50 males and
#         1300 images of 50 females, which is the version that
#         this function uses.
# -----------------------------------------------------------------
def read_prepare_AR(path, rs_rows,rs_cols, ntrain_pc, ntest_pc):
    t0   = time()
    X    = []
    y    = []
    for dirname , dirnames , filenames in os.walk(path):
        for f in filenames:
                subject_path = os.path.join(dirname , f)
                try:
                   if f[-3:] == 'bmp':
                    c       = int( f[2:5]  )  - 1
                    # check if woman :
                    if f[0] == 'W' :
                       c = c + 50
                    img     = Image.open(subject_path,'r').convert('L')
                    #img     = img.resize((rs_rows,rs_cols), Image.ANTIALIAS)
                    img     = img.resize((rs_cols,rs_rows), Image.ANTIALIAS)
                    img_arr = np.asarray(img.getdata()).reshape(-1,)
                    X.append( img_arr  )
                    y.append(c)
                    print("Image {} read.".format(f))
                except IOError:
                    print("I/O error({0}): {1}".format("errno", "strerror"))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
    t1 = time()
    print("Data read in %0.3fs" % (t1 - t0))
    X           = np.asarray( X, dtype=np.double )
    y           = np.asarray( y, dtype=np.int32 )
    classes     = np.unique( y )
    ntot        = X.shape[0]
    print("CLASSES = {}".format(classes))
    print("Number of total samples = {}".format(ntot))
    print("Number of classes = {}".format(classes.shape[0]))
    print("Max val = {}, Min val = {}".format(X.max(),X.min()))
    # random selection of train and test samples per class
    ind_train    = []
    ind_test     = []
    for c in range(0,classes.shape[0]):
       n_pc	 = ntrain_pc + ntest_pc
       ind_pc        = np.where( y==classes[c]  )[0]
       ind_train_pc  = np.random.choice(ind_pc, size=(ntrain_pc,),replace=False)
       ind_test_pc   = np.asarray( list(set(ind_pc) - set(ind_train_pc)) )
       for i in range(0,ntrain_pc):
          ind_train.append( ind_train_pc[i]  )
       for i in range(0,ntest_pc):
          ind_test.append( ind_test_pc[i] )
    ind_total_train = np.asarray(ind_train,dtype=np.int32)
    ind_total_test  = np.asarray(ind_test,dtype=np.int32)
    X_train = X[ind_total_train,:].copy()
    X_test  = X[ind_total_test,:].copy()
    y_train = y[ind_total_train].copy()
    y_test  = y[ind_total_test].copy()
    print("Data prepare in %0.3fs" % (time() - t1))
    return X_train, y_train, X_test, y_test
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

