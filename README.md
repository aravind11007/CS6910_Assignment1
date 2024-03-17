# CS6910_assignment1

Goal of this assignment is to implement a feed-forward neural network from scratch using numpy or pandas

## Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for 
training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed 
to use any automatic differentiation packages. This network will be trained and tested using the 
Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, 
the network will be trained to classify the image into 1 of 10 classes.

## Process

* X_train, y_train, X_test, y_test was loaded using the fashion mnist dataset
* 10 % data of train is given as validation dataset
* The weight initialisation is given using the if and elif condition. If the given weight intialisation is not there, it will give out the 'value error'. Therefore a new weight initialisation can be easily added in the new elif condition
```
  if weight_init=='random':
        w,b=random_initialisation(size_hidden)
    elif weight_init=='xavier':
        w,b=xavier_initialisation(size_hidden)
    else:
        raise ValueError('Given Initiatisation not found')
```
* A feedforward neural network has been developed to analyze images from the fashion-mnist dataset. It operates by passing input data through successive hidden layers, where each layer processes the data using an activation function to introduce non-linearity. This function transforms the weighted sum of inputs, aiding the network in learning complex patterns and generating a probability distribution across the 10 classes as output.

```
A,H=forward_prop(x,w,b,act=activation)
where x is the input, w is the weight, b is the bias, act is the activation function
```
* Loss was calculated, and the backpropagation algorithm was applied using various optimizers. The dataset was divided into minibatches for efficient processing. Backpropagation serves as a learning mechanism, iteratively adjusting the connection weights between neurons to minimize the difference between desired and actual outputs. This process fine-tunes the neural network by optimizing the weight values with respect to the inputs, gradually aligning the system's predictions with the desired outcomes.
```
grad_w,grad_b=back_prop(w,b,A,H,H[-1],y_hot,x,activation,loss_type)

where y_hot is one hot encoding of the ground truth

loss=cost(y_hot,H[-1],w,w_d,loss_type)

def cost(y_hot,y_pred,w,w_d,loss_type):
    if loss_type=='cross_entropy':
    
        loss=np.multiply(y_hot,y_pred)
        loss=np.sum(loss,axis=0)
        loss=-1*np.sum(np.log(loss))
        loss=loss/y_pred.shape[1]
    elif loss_type=='mean_squared_error':
        loss=np.sum((y_pred-y_hot)**2)
        loss=loss/y_pred.shape[1]
    
    weight_loss=0
    for weight in w:
        weight_loss+=(w_d/2)*np.sum(weight**2)
    
    
    return loss+weight_loss

```
* For optimizer the parameters, 6 optimizer algorithms were used
 ```
 if optimizer=='sgd':
                w,b=sgd(grad_w,w,grad_b,b,lr,w_d)
            elif optimizer=='momentum':
                w,b,prev_grad_w,prev_grad_b=momentum(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,m,w_d)
            elif optimizer=='nag':
                w,b,prev_grad_w,prev_grad_b=nag(grad_w,w,grad_b,b,lr,iter_,prev_grad_w,prev_grad_b,m,w_d)
                
            elif optimizer=='rmsprop':
                w,b,vt_w,vt_b=rmsprop(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta,w_d)
            elif optimizer=='adam':
                w,b,mt_w,mt_b,vt_w,vt_b=ADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d)
                
            elif optimizer=='nadam':
                w,b,mt_w,mt_b,vt_w,vt_b=NADAM(grad_w,w,grad_b,b,lr,iter_,vt_w,vt_b,beta1,mt_w,mt_b,beta2,ep,w_d)
                
                
            else:
                raise ValueError('Wrong Optimizer Given')

```
* All of these are included in a main function called  train_NN
```
train_NN(args.optimizer,args.learning_rate,args.weight_decay,args.num_layers,args.hidden_size,args.batch_size,args.loss,args.weight_init,args.activation,args.epochs,args.logs)
```
* Sweep functionality provided by wandb was used to find the best values of hyperparameters.
   - X_train: Input training data
   - Y_train: Output training data
   - activation: activation function
   - n_epoch: number of epochs to be run
   - learning_rate: learning rate of the algorithm
   - regulization term
   - weight initialization function: Xavier or Random
   - Type of loss: Cross entropy or mean squared error
   - Minibatch size
   - eg: ```def stochastic_GD(X_train,Y_train,activation,n_epoch,sizes,lr,reg,w_init,loss_type,minibatch_size=1,log=False)```
- The weights and biases are updated and reach optimum values.
- Sweep functionality provided by wandb was used to find the best values of hyperparameters.


## Activation functions
- Relu
- Sigmoid
- Tanh
- Identity

## Optimizers used:
- Sgd
- RmsProp
- Momentum
- Nestesrov
- Adam
- Nadam

## Loss functions
- Cross entropy
- Mean squared error

## Code specifications
A python script train.py was created that accepts the following command line arguments with the specified values -

### Arguments to be supported

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

<br>





  


