# CONV-HDDDL-Tensorflow
Tensorflow implementation of Convolutional Hierarchical Discriminative Deep Dictionary Learning.

## Python libraries required
 * tensorflow (compatibility with version 1)
 * numpy
 * pickle (only needed to save loss and accuracy results)
 * PIL (to read input images)

## Usage
Example execution: `python main.py -NCONV_I 2 -NDL 2 -NC 2 -N_FILT_I 16,16 -H_FILT_I 3,5 -MP_I 1,1 -D_I 1,1 -drop 0.4 -l_l1 0.0001 -K 10 -n_atoms 1000,90 -n_dicts 1,10 -out_sc1 7,1 -l_r 0.0001 -n_e 400 -b_s 50 -d_s 1 -in_f DATA/AR_DB_Cropped -ou_f OUT_FOLDER`
<br>

## Result
The suggested execution should reproduce on average the result for the HDDDL method on the (cropped) AR database as reported in :<br><br>

* [1]() 
  * Rodriguez-Dominguez and Dalmau (2020). Hierarchical Discriminative Deep
Dictionary Learning.
<br>
