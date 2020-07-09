"""
Hierarchical Discriminative Deep Dictionary Learning
Application : (Cropped) AR Faces
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
              Oscar Dalmau               - CIMAT
----------------------------------------------------------
------  Convolutional + Hierarchical Discriminative  -----
------      Dictionary layers model class file       -----

"""
from func_tools import read_prepare_AR
from func_ext import(
    xavier_initializer,
    variance_scaling_initializer
)
import pickle
import numpy as np
import tensorflow as tf


# Hierarchical Discriminative Deep Dictionary Learning
# Application : AR Faces
# Authors     : Ulises Rodriguez Dominguez - CIMAT
#              ulises.rodriguez@cimat.mx
#              Oscar Dalmau               - CIMAT

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class ConvHDDDL(object):

  def __init__(self, \
               sess, \
               NConv, NDL, NC, \
               nFilters,h,mPool,dropTrue,drop, Lambda,K,nAtoms,nDicts, out_sc,\
               learning_rate,num_epochs,batch_size,display_step,\
               input_folder,output_folder,\
               nrows,ncols, ntrain_pc,ntest_pc,NCLASSES):
     self.sess      = sess
     self.NConv     = NConv
     self.NDL       = NDL
     self.NC        = NC
     self.nFilt     = nFilters
     self.h         = h
     self.mPool     = mPool
     self.dropTrue  = dropTrue
     self.drop      = drop
     self.Lambda    = Lambda
     self.K         = K
     self.nAtoms    = nAtoms
     self.nDicts    = nDicts
     self.out_sc    = out_sc
     self.l_r       = learning_rate
     self.num_ep    = num_epochs
     self.batch_s   = batch_size
     self.gamma     = float(self.batch_s)
     self.disp_s    = display_step
     self.input_f   = input_folder
     self.output_f  = output_folder
     self.nrows     = nrows
     self.ncols     = ncols
     self.ntrain_pc = ntrain_pc
     self.ntest_pc  = ntest_pc
     self.NCLASSES  = NCLASSES
     self.ntrain    = 0
     self.ntest     = 0
     self.d         = 0
     self.load_data()
     self.build_model()

  def load_data(self):
     # Load images from Cropped AR database
     # --------------------------------------------------------------------
     print("Obtaining training and testing data..")
     X_train0, y_train0, X_test0, y_test0 = read_prepare_AR(self.input_f, \
                                                 self.nrows,self.ncols, \
                                                 self.ntrain_pc, self.ntest_pc)
     # Normalize input to [0-1]
     # --------------------------------------------------------------------
     maxV           = X_train0.max()
     minV           = X_train0.min()
     self.X_train   = (X_train0 - minV) / (maxV - minV)
     maxV           = X_test0.max()
     minV           = X_test0.min()
     self.X_test    = (X_test0 - minV) / (maxV - minV)
     self.X_train   = self.X_train.astype('float32')
     self.ntrain    = self.X_train.shape[0]
     self.ntest     = self.X_test.shape[0]
     self.d         = self.X_train.shape[1]
     print("ntrain={}, XTrain.shape={}".format(self.ntrain,self.X_train.shape))
     print("Xtrain = {}".format(self.X_train))
     # Reshape input to Neural Network format
     # --------------------------------------------------------------------
     self.X_train   = self.X_train.reshape((self.ntrain,self.d))
     self.X_test    = self.X_test.reshape((self.ntest,self.d))
     self.y_train   = np.zeros((self.ntrain,self.NCLASSES),dtype=np.int32)
     self.y_test    = np.zeros((self.ntest,self.NCLASSES),dtype=np.int32)
     y_unique  = np.unique( y_train0 )
     for c in range(0,self.NCLASSES):
        row_labels        = np.zeros((self.NCLASSES,),dtype=np.int32)
        row_labels[ c ]   = 1
        self.y_train[ y_train0 == y_unique[c], : ] = row_labels
        self.y_test[ y_test0 == y_unique[c], : ]   = row_labels
     print("Train.shape={}".format(self.X_train.shape))
     print("Test.shape={}".format(self.X_test.shape))
     print("YTrain.shape={}".format(self.y_train.shape))
     print("YTest.shape={}".format(self.y_test.shape))

  def build_model(self):
     # Build Tensorflow Graph input
     # --------------------------------------------------------------------
     self.inputs = []
     self.inputs.append( \
         tf.compat.v1.placeholder(tf.float32, shape=(None,self.d),name='input0')
     )
     self.Y      = tf.compat.v1.placeholder(tf.float32, shape=(None,self.NCLASSES),name='Y')
     # --------------------------------------------------------------------
     # Construct convolutional modules + HDDL layers
     # --------------------------------------------------------------------
     z_concat    = self.ConvHDDL_layers()
     # Classification layers (before final output) ------------------
     # --------------------------------------------------------------------
     o_conv_s    = z_concat.get_shape().as_list()[1]
     di          = int(o_conv_s/self.NCLASSES)
     # First Multilayer Perceptron module with NC layers
     # --------------------------------------------------------------------
     out1_c      = tf.add(tf.matmul(tf.reshape(z_concat[:,0:di],[-1,di]), \
                     tf.compat.v1.get_variable('W1O1c0', [di, self.out_sc[0]],\
                                               initializer=xavier_initializer())), \
                     tf.compat.v1.get_variable('b1O1c0', [self.out_sc[0]],\
                                               initializer=xavier_initializer()) )
     for l in range(1,self.NC):
        # add relu in between
        out1_c    = tf.nn.leaky_relu(out1_c,alpha=0.01)
        out1_c    = tf.add(tf.matmul(out1_c, \
                     tf.compat.v1.get_variable('W'+str(l+1)+'O1c0', \
                                             [self.out_sc[l-1],self.out_sc[l]], \
                                             initializer=variance_scaling_initializer())), \
                     tf.compat.v1.get_variable('b'+str(l+1)+'O1c0', \
                                             [self.out_sc[l]],\
                                             initializer=variance_scaling_initializer())  )
     # Rest of the Multilayer Perceptron modules, each with NC layers
     # --------------------------------------------------------------------
     out1 = out1_c
     for c in range(1,self.NCLASSES):
        out1_c      = tf.add(tf.matmul(tf.reshape(z_concat[:,(c*di):((c+1)*di)],[-1,di]), \
                          tf.compat.v1.get_variable('W1O1c'+str(c), \
                                 [di, self.out_sc[0]], \
                                 initializer=xavier_initializer())), \
                          tf.compat.v1.get_variable('b1O1c'+str(c), \
                                 [self.out_sc[0]], \
                                 initializer=xavier_initializer()) )
        for l in range(1,self.NC):
           # add relu in between
           out1_c    = tf.nn.leaky_relu(out1_c,alpha=0.01)
           out1_c    = tf.add(tf.matmul(out1_c, \
                         tf.compat.v1.get_variable('W'+str(l+1)+'O1c'+str(c), \
                                 [self.out_sc[l-1],self.out_sc[l]], \
                                 initializer=variance_scaling_initializer())), \
                         tf.compat.v1.get_variable('b'+str(l+1)+'O1c'+str(c), \
                                 [self.out_sc[l]], \
                                 initializer=variance_scaling_initializer())  )
        out1 = tf.concat([out1,out1_c],1)
     out1         = tf.reshape( out1, [-1,self.NCLASSES] )
     # Softmax activation to get one-hot-code representation
     # --------------------------------------------------------------------
     self.prediction1  = tf.nn.softmax(out1,name='predict1')
     # Define loss and optimizer
     # --------------------------------------------------------------------
     self.loss_op      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                                 logits=self.prediction1, labels=self.Y))
     self.optimizer    = tf.compat.v1.train.AdamOptimizer(learning_rate=self.l_r)
     self.train_op     = self.optimizer.minimize(self.loss_op)
     # Evaluate model
     # --------------------------------------------------------------------
     self.correct_pred = tf.equal(tf.argmax(self.prediction1, 1), tf.argmax(self.Y, 1))
     self.avg_acc      = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
     # Initialize the variables (i.e. assign their default value)
     self.init         = tf.compat.v1.global_variables_initializer()


  # Convolutional layer with batch normalization
  def conv2d(self, x, W, b, strides=1):
     # Conv2D wrapper, with bias and relu activation
     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
     x = tf.nn.bias_add(x, b)
     # normalization layer
     x = tf.compat.v1.layers.batch_normalization(x,training=True)
     return tf.nn.leaky_relu(x,alpha=0.01)

  # Max-pooling layer
  def maxpool2d(self, x, k=2):
     # MaxPool2D wrapper
     return tf.nn.max_pool2d(x, ksize=[1, k+1, k+1, 1], strides=[1, k, k, 1],padding='SAME')

  # Single dictionary layer
  #    (outputs coefficients of K ISTA-like iterations)
  # ---------------------------------------------------------------------
  def dictionary_layer(self, x,z0,c1,c2,D,K,n_atoms):
     # iterative gradient descent step
     dl_sk  = tf.add(z0, \
                    tf.matmul( tf.multiply( tf.add(tf.matmul(tf.reshape(z0,[-1,n_atoms]),tf.transpose(D)), -x) , -c1 ) ,\
                    D ))
     # soft thresholding step
     dl_zk = tf.multiply(  tf.maximum( tf.add( tf.abs(dl_sk) , -c2 ) , 0.0 ) ,\
                              tf.sign(dl_sk)  )
     for k in range(1,K+1):
        # iterative gradient descent step
        dl_sk = tf.add(dl_zk, \
                    tf.matmul( tf.multiply( tf.add(tf.matmul(tf.reshape(dl_zk,[-1,n_atoms]),tf.transpose(D)), -x) , -c1 ) ,\
                    D ))
        # soft thresholding step
        dl_zk = tf.multiply(  tf.maximum( tf.add( tf.abs(dl_sk) , -c2 ) , 0.0 ) ,\
                              tf.sign(dl_sk)  )
     return dl_zk

  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def ConvHDDL_layers(self):
     # define constants for dictionarly learning layers
     c1 = tf.constant([1.],dtype=tf.float32)
     c1 = tf.compat.v1.div( c1, tf.constant([self.gamma],dtype=tf.float32) )
     c2 = tf.constant([self.Lambda],dtype=tf.float32)
     c2 = tf.compat.v1.div( c2, tf.constant([self.gamma],dtype=tf.float32) )
     # Append convolutional modules
     # ----------------------------------------------------------------
     # Each sample input is a 1-D vector of nrows*ncols*1.
     # Each tensor input become 4-D: [Batch Size, Height, Width, N-Channels]
     x = \
           tf.reshape(self.inputs[0], shape=[-1, self.nrows, self.ncols, 1])
     # Convolutional layers -----------------------------------------
     for l in range(0,self.NConv):
        if l == 0 :
           conv_l = self.conv2d(x, tf.compat.v1.get_variable('Wconv'+str(l+1), \
                              [self.h[l],self.h[l],1,self.nFilt[l]], \
                              initializer=variance_scaling_initializer()), \
                              tf.compat.v1.get_variable('bconv'+str(l+1), \
                              [self.nFilt[l]], \
                              initializer=variance_scaling_initializer()), strides=1 )
        else :
           conv_l = self.conv2d(conv_l, tf.compat.v1.get_variable('Wconv'+str(l+1), \
                              [self.h[l],self.h[l],self.nFilt[l-1],self.nFilt[l]], \
                              initializer=variance_scaling_initializer()), \
                              tf.compat.v1.get_variable('bconv'+str(l+1), \
                              [self.nFilt[l]], \
                              initializer=variance_scaling_initializer()), strides=1 )
        if self.dropTrue[l] == 1 :
           conv_l = tf.nn.dropout(conv_l,rate=self.drop)
           #conv_l = tf.nn.dropout(conv_l,keep_prob=(1.-self.drop))
        if self.mPool[l] == 1 :
           conv_l = self.maxpool2d(conv_l, k=2)
     # Define input for HDDL layers
     # --------------------------------------------------------------
     if self.NConv == 0 :
        out_af = x
     else :
        out_af = conv_l
     # reshape to match HDDL layers input
     o_af_s    = out_af.get_shape().as_list()
     #                     (batch size, out_size1 * out_size2 * ouf_nfilters_size)
     o_af      = tf.reshape(out_af,[-1,o_af_s[1]*o_af_s[2]*o_af_s[3]])
     o_af_s    = o_af.get_shape().as_list()[1]
     # HDDL layers ----------------------------------------------------
     # ----------------------------------------------------------------
     z0        = tf.compat.v1.get_variable('z0', [self.nAtoms[0]], initializer=xavier_initializer())
     D1        = tf.compat.v1.get_variable('D1', [o_af_s,self.nAtoms[0]], initializer=xavier_initializer())
     z_out     = self.dictionary_layer(o_af,z0,c1,c2,D1,self.K,self.nAtoms[0])
     for l in range(1,self.NDL):
        d_di      = int(self.nDicts[l-1]*self.nAtoms[l-1]/self.nDicts[l])
        zl        = tf.compat.v1.get_variable('z'+str(l)+'d0', [self.nAtoms[l]], \
                                              initializer=xavier_initializer())
        Dl        = tf.compat.v1.get_variable('D'+str(l+1)+'d0', [d_di,self.nAtoms[l]],\
                                              initializer=xavier_initializer())
        z_out_di  = self.dictionary_layer(z_out[:,0:d_di],zl,c1,c2,Dl,self.K,self.nAtoms[l])
        for dict_i in range(1,self.nDicts[l]):
           zl     = tf.compat.v1.get_variable('z'+str(l)+'d'+str(dict_i), [self.nAtoms[l]],\
                                              initializer=xavier_initializer())
           Dl     = tf.compat.v1.get_variable('D'+str(l+1)+'d'+str(dict_i), [d_di,self.nAtoms[l]],\
                                              initializer=xavier_initializer())
           z_out_di= tf.concat( [ z_out_di, \
                        self.dictionary_layer(z_out[:,(dict_i*d_di):((dict_i+1)*d_di)],zl,c1,c2,\
                                         Dl,self.K,self.nAtoms[l]) ],\
                      1 )
        z_out = z_out_di
     return z_out
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  def train(self):
     # Start training
     # --------------------------------------------------------------------
     loss_train = []
     acc_train  = []
     acc_test   = []
     # Run the initializer
     self.sess.run(self.init)
     #Create a saver object which will save all the variables
     self.saver = tf.compat.v1.train.Saver()
     for epoch in range(1, self.num_ep+1):
      rand_indexes = np.random.choice(self.ntrain,size=self.ntrain)
      print("EPOCH {} -----------------------------------------------------".format(epoch))
      for batch_i in range(0,int(self.ntrain/self.batch_s)):
         ind_s = rand_indexes[(batch_i*self.batch_s):((batch_i+1)*self.batch_s)]
         batch_x_data = []
         ind_sc = 0
         batch_x_data.append( self.X_train[ind_s,:].reshape(self.batch_s,-1)  )
         batch_y = self.y_train[ind_s,:]
         dict_feed = {self.inputs[0] : batch_x_data[0]}
         dict_feed.update({self.Y:batch_y})
         self.sess.run(self.train_op, feed_dict=dict_feed)
         if (batch_i % self.disp_s==0) or (batch_i == 0 and epoch==1):
             # Calculate batch loss and accuracy
             loss, a_acc = self.sess.run([self.loss_op, self.avg_acc], \
                                          feed_dict=dict_feed)
             loss_train.append(loss)
             acc_train.append(a_acc)
             print("Batch {}, Minibatch Loss = {}, Minibatch (Avg.) Accuracy = {}, epoch = {}".format(str(batch_i),loss,a_acc,epoch))
         if epoch % 1 == 0 and batch_i==0 :
             global_test_avg_acc = self.sess.run( self.avg_acc, \
                                          feed_dict={self.inputs[0]:self.X_test, self.Y:self.y_test})
             acc_test.append( global_test_avg_acc )
             pickle.dump(acc_test,open(self.output_f+'/AR_ACC_TEST.p','wb'))
             print("-------------------------------------------------------")
             print("Testing Accuracy:", global_test_avg_acc)
             print("-------------------------------------------------------")
      pickle.dump(loss_train,open(self.output_f+'/AR_LOSS_TRAIN.p','wb'))
      pickle.dump(acc_train,open(self.output_f+'/AR_ACC_TRAIN.p','wb'))
      print("EPOCH {} -----------------------------------------------------".format(epoch))
     print("Optimization Finished!")
     global_test_avg_acc = self.sess.run( self.avg_acc, \
                                          feed_dict={self.inputs[0]:self.X_test, \
                                                     self.Y:self.y_test})
     print("-------------------------------------------------------")
     print("Testing Accuracy:", global_test_avg_acc)
     print("-------------------------------------------------------")

