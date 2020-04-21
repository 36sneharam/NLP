import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    
    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1) + b1
    
    h = sigmoid(data.dot(W1) + b1) 
    z2 = h.dot(W2) + b2
    yhat = softmax(h.dot(W2) + b2)

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    
    #Computing Cost 
    
    cost = -np.sum(np.multiply(labels, np.log(yhat)), axis = 1) #sum all rows 
    #print(cost)
    
    gradW2 = np.zeros((H, Dy))
    gradb2 = np.zeros((1,Dy))
    gradW1 = np.zeros((Dx, H))
    gradb1 = np.zeros((1,H))
    for j in range(0, H):
        for k in range(0, Dy): 
        	gradW2[j,k] = h[0,j]*(yhat[0,k]-labels[0,k])
    #print(gradW2)
    for j in range(0, Dy):
    	gradb2[0,j] = (yhat[0,j]-labels[0,j])
    #print(gradb2)

    for j in range(0, Dx):
    	for k in range(0, H):
    		p = np.dot(W2[k,0:Dy], np.transpose(yhat-labels))
    		gradW1[j,k] = sigmoid_grad(z1[0,k])*data[0,j]*p

    for j in range(0, H):
    	p = np.dot(W2[j,:], np.transpose(yhat-labels))
    	gradb1[0,j] = p*sigmoid_grad(z1[0,j])

    #print(gradW1)
    

    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    N = 1 
    dimensions = [10, 5, 10] 
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    for i in range(0,N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1
    params = np.random.rand((dimensions[0]+1)*dimensions[1] + (dimensions[1]+1)*dimensions[2],)
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)
    #gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
    #    dimensions), params)

    
    ### END YOUR CODE

if __name__ == "__main__":
    #sanity_check()
    your_sanity_checks()
