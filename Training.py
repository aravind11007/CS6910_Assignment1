#!/usr/bin/env python
# coding: utf-8

# In[14]:

###importing the necessary library
import wandb
import numpy as np
import numpy as np
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import argparse


# In[15]:

##sigmoid activation

def Sigmoid(z):
    return 1/(1+np.exp(-z))

##ReLu activation
def Relu(z):
    return np.maximum(0,z)

### Tanh activation
def Tanh(z):
    num=np.exp(z)-np.exp(-z)
    dem=np.exp(z)+np.exp(-z)
    
    return num/dem

## softmax implementation
def softmax(a):
    out=a.copy()
    for i in range(out.shape[1]):
        out[:,i]=np.exp(out[:,i]-np.max(out[:,i]))
        out[:,i]= out[:,i]/np.sum(out[:,i])
    return out

####derivative of sigmoid
    
def deri_sigmoid(z):
    return z*(1-z)

###derivative of relu

def deri_relu(z):
    return np.int64(z>0)

###derivative of tanh
def deri_tanh(z):
    return 1-(Tanh(z)*Tanh(z))




# In[16]:

### Random initialisation
def random_initialisation(size_hidden):
    weight=[]
    bais=[]
    for i in range(len(size_hidden)):
        if i==0:
            W=np.random.randn(size_hidden[i],x_train.shape[0])*.01
            b=np.zeros((size_hidden[i],1))
            weight.append(W)
            bais.append(b)
        else:
            W=np.random.randn(size_hidden[i],size_hidden[i-1])*.01
            b=np.zeros((size_hidden[i],1))
            weight.append(W)
            bais.append(b)
    return weight,bais


####xavier initilaisation
def xavier_initialisation(size_hidden):
    weight=[]
    bais=[]
    for i in range(len(size_hidden)):
        if i==0:
            W=np.random.randn(size_hidden[i],x_train.shape[0])*np.sqrt(2/(size_hidden[i]+x_train.shape[0]))
            b=np.zeros((size_hidden[i],1))
            weight.append(W)
            bais.append(b)
        else:
            W=np.random.randn(size_hidden[i],size_hidden[i-1])*np.sqrt(2/(size_hidden[i]+size_hidden[i-1]))
            b=np.zeros((size_hidden[i],1))
            weight.append(W)
            bais.append(b)
    return weight,bais

### forward propogation function

def forward_prop(X,W,B,act='sigmoid'):
    A=[]
    H=[]
    activations=['sigmoid', 'tanh', 'ReLU','identity']
    if act in activations:
        for i in range(len(W)):
            if i==0:
                a=(W[i]@X)+B[i]
                A.append(a)
                if act=='sigmoid':
                    h=Sigmoid(a)
                    H.append(h)
                elif act=='tanh':        #### activation function is applied based on the argument given
                    h=Tanh(a)
                    H.append(h)
                elif act=='ReLU':
                    h=Relu(a)
                    H.append(h)
                elif act=='identity':
                    h=a.copy()
                    H.append(h)
                
            else:
                a=(W[i]@h)+B[i]
                A.append(a)
                if i+1!=len(W):
                    if act=='sigmoid':
                        h=Sigmoid(a)
                        H.append(h)
                    elif act=='tanh':
                        h=Tanh(a)
                        H.append(h)
                    elif act=='ReLU':
                        h=Relu(a)
                        H.append(h)
                    elif act=='identity':
                        h=a.copy()
                        H.append(h)
                else:
                    y=softmax(a)
                    H.append(y)
                
        return A,H
        
    else:
        raise ValueError('you have given a wrong activation function')
        

###one hot encoding function for converting the the label in to a vector where the index correspond to label is given as 1 
        
def one_hot(y):
    y_=np.zeros((10,len(y)))
    
    for i in range(y.shape[0]):
        label=y[i]
        y_[label,i]=1
    return y_

####backpropogation function###
        
def back_prop(w,b,a,h,ypred,y_hot,x,act,loss_type):
    GRAD_h,GRAD_a,GRAD_w,GRAD_b=[],[],[],[]
    
    if loss_type=='cross_entropy':
        grad_a=-(y_hot-ypred)
        grad_h=0                                      ##### changing the final layer gradient based on the loss function
    elif loss_type=='mean_squared_error':
        grad_a=(ypred-y_hot)*ypred*(1-ypred)
        grad_h=0
    else:
        raise ValueError('Loss function not found')
    
    GRAD_a.append(grad_a)
    GRAD_h.append(grad_h)
    N=x.shape[1]
    
    activations=['sigmoid', 'tanh', 'ReLU','identity']
    if act in activations:
    
        for z,i in enumerate(range(len(h)-1,-1,-1)):


            if z==0:
                GRAD_w.append((GRAD_a[z]@h[i-1].T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
            elif i!=0:
                grad_h=w[i+1].T@GRAD_a[z-1]
                if act=='sigmoid':
                    grad_a=grad_h*deri_sigmoid(h[i])
                elif act=='ReLU':                        #### gradient changing based on the activation function
                    grad_a=grad_h*deri_relu(h[i])
                elif act=='tanh':
                    grad_a=grad_h*deri_tanh(h[i])
                elif act=='identity':
                    grad_a=grad_h

                GRAD_h.append(grad_h)
                GRAD_a.append(grad_a)

                GRAD_w.append((GRAD_a[z]@h[i-1].T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
            else:
                grad_h=w[i+1].T@GRAD_a[z-1]
                if act=='sigmoid':
                    grad_a=grad_h*deri_sigmoid(h[i])
                elif act=='ReLU':
                    grad_a=grad_h*deri_relu(h[i])
                elif act=='tanh':                      #### gradient changing based on the activation function
                    grad_a=grad_h*deri_tanh(h[i])
                elif act=='identity':
                    grad_a=grad_h
                GRAD_h.append(grad_h)
                GRAD_a.append(grad_a)

                GRAD_w.append((GRAD_a[z]@x.T)/N)
                GRAD_b.append((np.sum(GRAD_a[z],axis=1,keepdims=True))/N)
                
    else:
        raise ValueError('you have given a wrong activation function')
            
            
    return GRAD_w[::-1],GRAD_b[::-1]

###calculation of loss function
def cost(y_hot,y_pred,w,w_d,loss_type):
    if loss_type=='cross_entropy':
    
        loss=np.multiply(y_hot,y_pred)
        loss=np.sum(loss,axis=0)
        loss=-1*np.sum(np.log(loss))
        loss=loss/y_pred.shape[1]
    elif loss_type=='mean_squared_error':
        loss=np.sum((y_pred-y_hot)**2)
        loss=loss/y_pred.shape[1]
    
    ####for adding the regularisation term contributing loss
    weight_loss=0
    for weight in w:
        weight_loss+=(w_d/2)*np.sum(weight**2)
    
    
    return loss+weight_loss



###sgd optimizer####
def sgd(grad_w,w,grad_b,b,lr,w_d):
    w_update=[]
    b_update=[]
    for i in range(len(w)):
        w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
        b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
        
    return w_update,b_update
    
###momentum optimizer####     
def momentum(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,beta,w_d):
    if iter_==0:

        w_update=[]
        b_update=[]
        prev_grad_w=[]
        prev_grad_b=[]
        for i in range(len(w)):
            w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
            prev_grad_w.append(lr*grad_w[i])
            prev_grad_b.append(lr*grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b
    else:
        w_update=[]
        b_update=[]
        for i in range(len(w)):
            update_w=(lr*grad_w[i]+beta*prev_grad_w[i])
            update_b=(lr*grad_b[i]+beta*prev_grad_b[i])
            w_update.append(w[i]*(1-lr*w_d)-update_w)
            b_update.append(b[i]*(1-lr*w_d)-update_b)
            prev_grad_w[i]=update_w
            prev_grad_b[i]=update_b

        return w_update,b_update,prev_grad_w,prev_grad_b

###nag optimizer####  
def nag(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,beta,w_d,x,y_hot,activation,loss_type):
    if iter_==0:

        w_update=[]
        b_update=[]
        prev_grad_w=[]
        prev_grad_b=[]
        for i in range(len(w)):
            w_update.append(w[i]*(1-lr*w_d)-lr*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-lr*grad_b[i])
            prev_grad_w.append(lr*grad_w[i])
            prev_grad_b.append(lr*grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b
    
    else:
        w_update=[]
        b_update=[]
        w_look=[]
        b_look=[]
        for i in range(len(w)):
            update_w=(beta*prev_grad_w[i])
            update_b=(beta*prev_grad_b[i])
            w_look.append(w[i]*(1-lr*w_d)-update_w)
            b_look.append(b[i]*(1-lr*w_d)-update_b)
        A,H=forward_prop(x,w_look,b_look,act=activation)
        back_prop(w,b,A,H,H[-1],y_hot,x,activation,loss_type)
        grad_w,grad_b=back_prop(w_look,b_look,A,H,H[-1],y_hot,x,activation,loss_type)
        
        for i in range(len(w)):
            prev_grad_w[i]=beta*prev_grad_w[i]+lr*grad_w[i]
            prev_grad_b[i]=beta*prev_grad_b[i]+lr*grad_b[i]
            
            w_update.append(w[i]*(1-lr*w_d)-prev_grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-prev_grad_b[i])

        return w_update,b_update,prev_grad_w,prev_grad_b

###rmsprop optimizer####  
def rmsprop(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta,w_d):
    
    if iter_==0:
        eps=args.epsilon

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        for i in range(len(w)):
            vt_w.append((1-beta)*grad_w[i]**2)
            vt_b.append((1-beta)*grad_b[i]**2)
            
            div_w=(1/np.sqrt(vt_w[i]+eps))
            div_b=(1/np.sqrt(vt_b[i]+eps))
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*div_w)*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-(lr*div_b)*grad_b[i])
            

        return w_update,b_update,vt_w,vt_b
    else:
        eps=args.epsilon
        w_update=[]
        b_update=[]
        
        for i in range(len(w)):
            vt_w[i]=beta*vt_w[i]+(1-beta)*(grad_w[i]**2)
            vt_b[i]=beta*vt_b[i]+(1-beta)*(grad_b[i]**2)
            
            '''
            div_w=(lr/np.sqrt(vt_w[i]+eps))
            div_b=(lr/np.sqrt(vt_b[i]+eps))
            '''
            
            div_w=np.multiply(lr,np.reciprocal(np.sqrt(vt_w[i]+eps)))
            div_b=np.multiply(lr,np.reciprocal(np.sqrt(vt_b[i]+eps)))
            
            
            w_update.append(w[i]*(1-lr*w_d)-div_w*grad_w[i])
            b_update.append(b[i]*(1-lr*w_d)-div_b*grad_b[i])
        return w_update,b_update,vt_w,vt_b
        

    
###ADAM####          
def ADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d):
    if iter_==0:
        eps=args.epsilon

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        mt_w=[]
        mt_b=[]
        for i in range(len(w)):
            
            vt_w.append((1-beta2)*grad_w[i]**2)
            vt_b.append((1-beta2)*grad_b[i]**2)
            
            mt_w.append((1-beta1)*grad_w[i])
            mt_b.append((1-beta1)*grad_b[i])
            

            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
            

        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    else:
        eps=args.epsilon
        w_update=[]
        b_update=[]
        
        for i in range(len(w)):
            

            vt_w[i]=vt_w[i]*beta2+(1-beta2)*grad_w[i]**2
            vt_b[i]=vt_b[i]*beta2+(1-beta2)*grad_b[i]**2
            
            
            mt_w[i]=beta1*mt_w[i]+(1-beta1)*grad_w[i]
            mt_b[i]=beta1*mt_b[i]+(1-beta1)*grad_b[i]
            
        
            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
        
        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    

###NADAM optimizer####      
def NADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d,x,y_hot,activation,loss_type):
    if iter_==0:
        eps=args.epsilon

        w_update=[]
        b_update=[]
        vt_w=[]
        vt_b=[]
        mt_w=[]
        mt_b=[]
        for i in range(len(w)):
            
            vt_w.append((1-beta2)*grad_w[i]**2)
            vt_b.append((1-beta2)*grad_b[i]**2)
            
            mt_w.append((1-beta1)*grad_w[i])
            mt_b.append((1-beta1)*grad_b[i])
            

            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
            

        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
    else:
        eps=args.epsilon
        w_update=[]
        b_update=[]
        w_look=[]
        b_look=[]
        
        for i in range(len(w)):
            w_look.append(w[i]-beta1*mt_w[i])
            b_look.append(b[i]-beta1*mt_b[i])
            
        A,H=forward_prop(x,w_look,b_look,act=activation)
        grad_w,grad_b=back_prop(w_look,b_look,A,H,H[-1],y_hot,x,activation,loss_type)
        
        for i in range(len(w)):

            
            mt_w[i]=beta1*mt_w[i]+(1-beta1)*grad_w[i]
            mt_b[i]=beta1*mt_b[i]+(1-beta1)*grad_b[i]
            
            
            
            vt_w[i]=vt_w[i]*beta2+(1-beta2)*grad_w[i]**2
            vt_b[i]=vt_b[i]*beta2+(1-beta2)*grad_b[i]**2
            
            
        
            
            vt_w_=vt_w[i]/(1-np.power(beta2,ep+1))
            vt_b_=vt_b[i]/(1-np.power(beta2,ep+1))
            
            mt_w_=mt_w[i]/(1-np.power(beta1,ep+1))
            mt_b_=mt_b[i]/(1-np.power(beta1,ep+1))
            
            w_=mt_w_/(np.sqrt(vt_w_+eps))
            b_=mt_b_/(np.sqrt(vt_b_+eps))
            
            
            w_update.append(w[i]*(1-lr*w_d)-(lr*w_))
            b_update.append(b[i]*(1-lr*w_d)-(lr*b_))
        
        return w_update,b_update,mt_w,mt_b,vt_w,vt_b
            
      


# In[17]:

####TRAINING FUNCTION
def train_NN(optimizer,lr,w_d,num_layers,hidden_size,batch_size,loss_type,weight_init,activation,epochs,logs):
    size_hidden=[]
    for z in range(num_layers):
        size_hidden.append(hidden_size)
    size_hidden.append(10)

    beta1=args.beta1
    beta2=args.beta2
    beta=args.beta
    m=args.momentum
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    prev_grad_w=[]
    prev_grad_b=[]
    vt_w=[]
    vt_b=[]
    mt_w=[]
    mt_b=[]
    if weight_init=='random':
        w,b=random_initialisation(size_hidden)
    elif weight_init=='xavier':                        ####WEIGHT INITIALISATION
        w,b=xavier_initialisation(size_hidden)                  
    else:
        raise ValueError('Given Initiatisation not found')
    for ep in range(epochs):
        curr_train_loss=[]
        curr_train_acc=[]

        for i in range(0,len(y_train),batch_size):
            if i==0 and ep==0:
                iter_=0
            else:
                iter_=1
            if i+batch_size>len(y_train):
                break
            else:
                x,y=x_train[:,i:i+batch_size],y_train[i:i+batch_size]


                A,H=forward_prop(x,w,b,act=activation) ###FORWARD PROPOGATION
                y_hot=one_hot(y) ##ONE HOT ENCODING

                curr_train_loss.append(cost(y_hot,H[-1],w,w_d,loss_type))  ##CALCULATING LOSS FUNCTION AND APPENDING TO LOSS

                grad_w,grad_b=back_prop(w,b,A,H,H[-1],y_hot,x,activation,loss_type)###BACK PROPOGATION

                if optimizer=='sgd':
                    w,b=sgd(grad_w,w,grad_b,b,lr,w_d)   ####OPTIMIZWE
                elif optimizer=='momentum':
                    w,b,prev_grad_w,prev_grad_b=momentum(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,m,w_d)
                elif optimizer=='nag':
                    w,b,prev_grad_w,prev_grad_b=nag(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,m,w_d,x,y_hot,activation,loss_type)

                elif optimizer=='rmsprop':
                    w,b,vt_w,vt_b=rmsprop(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta,w_d)
                elif optimizer=='adam':
                    w,b,mt_w,mt_b,vt_w,vt_b=ADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d)

                elif optimizer=='nadam':
                    w,b,mt_w,mt_b,vt_w,vt_b=NADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d,x,y_hot,activation,loss_type)
                    
                                             


                else:
                    raise ValueError('Wrong Optimizer Given')



                acc=(np.argmax(H[-1],axis=0)==y).sum()  ####ACCURACY CALCULATION

                curr_train_acc.append(acc/len(y))

        train_loss.append(np.average(curr_train_loss))

        train_acc.append(np.average(curr_train_acc))



        A,H=forward_prop(x_test,w,b,act=activation)  ###FORWARD PROPOGATION FOR TEST
        y_hot=one_hot(y_test)                        ####ONE HOT ENCODING FOR TEST
        val_loss.append(cost(y_hot,H[-1],w,w_d,loss_type)) ### LOSS CALCULATION FOR TEST
        acc=(np.argmax(H[-1],axis=0)==y_test).sum()###ACCURACY FOR TEST

        val_acc.append(acc/len(y_test))
        
        print(f'Epochs{ep} completed, Current train loss and accuray is {train_loss[ep],train_acc[ep]} and'
          ,f'Current val loss and accuray is {val_loss[ep],val_acc[ep]} ')
        print("")               ##PRINT THE AFTER EACH EPOCHS
        if logs==True: ###IF TRUE WILL LOG TO WANDB
       
            wandb.log({"Train_Accuracy":np.round(train_acc[ep]*100,2),"Train_Loss":train_loss[ep],
                       "Val_Accuracy":np.round(val_acc[ep]*100,2),"Val_Loss":val_loss[ep],"Epoch":ep})
            
            


# In[ ]:


parser = argparse.ArgumentParser()
 
parser.add_argument("-wp", "--wandb_project", default = "myprojectname", help = "Project name used to track experiments ")
parser.add_argument("-we", "--wandb_entity", default = "ee22s060", help = "Wandb Entity ")
parser.add_argument("-d", "--dataset", default = "fashion_mnist", choices=['mnist', 'fashion_mnist'],help = "Dataset" )
parser.add_argument("-e", "--epochs", default = 15, help = "Number of epochs to train neural network." , type=int)
parser.add_argument("-b", "--batch_size", default = 16, help = "Batch size ", type=int)
parser.add_argument("-l", "--loss", default = "cross_entropy", choices=['mean_squared_error', 'cross_entropy'],help = "Loss function ")
parser.add_argument("-o", "--optimizer", default = "nadam", choices=['sgd', 'momentum','nag','rmsprop','adam','nadam'] )
parser.add_argument("-lr", "--learning_rate", default = 0.001, help = "Learning rate " , type = float)
parser.add_argument("-m", "--momentum", default = 0.5, help = "Momentum" , type = float)
parser.add_argument("-beta", "--beta", default = 0.9, help = "Beta", type = float)

parser.add_argument("-beta1", "--beta1", default = 0.9, help = "Beta1" , type = float)
parser.add_argument("-beta2", "--beta2", default = 0.999, help = "Beta2", type = float)
parser.add_argument("-eps", "--epsilon", default = 1e-8, help = "Epsilon used by optimizers." , type = float)
parser.add_argument("-w_d", "--weight_decay", default = 0.0005, help = "Weight decay used by optimizers.", type = float)
parser.add_argument("-w_i", "--weight_init", default = "xavier", choices=['random', 'xavier'],help = "Initialisation" )
parser.add_argument("-nhl", "--num_layers", default = 4, help = "Number of hidden layers", type = int)
parser.add_argument("-sz", "--hidden_size", default = 128, help = "Number of hidden neurons in a feedforward layer." , type = int)
parser.add_argument("-a", "--activation", default = "tanh", choices = ["identity", "sigmoid", "tanh", "ReLU"],help = "Activation function to use" )
parser.add_argument("-lg", "--logs", default = "False", choices = ["True","False"],help = "whether to log or not" )
args = parser.parse_args()

if(args.dataset == "mnist"):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()  ###FOR LOADING THE DATASET
    
x_train=np.zeros((X_train.shape[1]*X_train.shape[2],X_train.shape[0]))
x_test=np.zeros((X_test.shape[1]*X_test.shape[2],X_test.shape[0]))

for i in range(X_train.shape[0]):
    img=X_train[i]
    img=img.flatten()
    x_train[:,i]=img

for i in range(X_test.shape[0]):
    img=X_test[i]
    img=img.flatten()                   ####PROPROCESSING THE DATASET
    x_test[:,i]=img
x_train=x_train/np.max(x_train)
x_test=x_test/np.max(x_test)
'''

wandb.login(key="5bfaaa474f16b4400560a3efa1e961104ed54810")
wandb.init(project=args.wandb_project,entity=args.wandb_entity)
'''

# In[ ]:


parameters=train_NN(args.optimizer,args.learning_rate,args.weight_decay,args.num_layers,args.hidden_size,args.batch_size,args.loss,args.weight_init,args.activation,args.epochs,args.logs)  ###MAIN FUNCTION

