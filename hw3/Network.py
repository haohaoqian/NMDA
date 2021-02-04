import numpy as np
import matplotlib.pyplot as plt
import math

class Network(object):

    def __init__(self, hidden_size, input_size = 256, output_size = 10, std = 1e-4):
        
        self.params = {}
            
        self.params['W1'] = std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        
        return
    
    def forward_pass(self, X, y = None, wd_decay = 0.0):
    
        loss = None
        predict = None
        
        self.hidden_out = np.clip(np.dot(X,self.params['W1'])+self.params['b1'],0,np.inf)
        self.class_out = np.dot(self.hidden_out,self.params['W2'])+self.params['b2']
        self.softmax_out = np.exp(self.class_out)/np.sum(np.exp(self.class_out),axis=-1).reshape(X.shape[0],1)
        
        if y is None:

            predict = np.argmax(self.softmax_out,axis=-1)           
            
            return predict
        else:

            loss = np.mean(-np.log(self.softmax_out[range(len(self.softmax_out)),y])) + wd_decay*(np.sum(self.params['W1']**2)+np.sum(self.params['W2']**2))/2
            
            return loss
        
    def back_prop(self, X, y, wd_decay = 0.0):
        grads = {}

        #grads should contain grads['W1'] grads['b1'] grads['W2'] grads['b2']
        delta = np.eye(self.params['b2'].shape[-1])[y]
        
        grads['W1'] = (1/X.shape[0])*np.dot(X.T, ((np.dot(X, self.params['W1']) + self.params['b1']) > 0) * np.dot((self.softmax_out - delta), self.params['W2'].T)) + wd_decay * self.params['W1']
        grads['b1'] = np.mean(((np.dot(X, self.params['W1']) + self.params['b1']) > 0) * np.dot((self.softmax_out - delta), self.params['W2'].T), axis=0)
        grads['W2'] = (1/X.shape[0])*np.dot(self.hidden_out.T, (self.softmax_out - delta)) + wd_decay * self.params['W2']
        grads['b2'] = np.mean(self.softmax_out - delta, axis=0)
        
        return grads
 
    def numerical_gradient(self, X, y, wd_decay = 0.0, delta = 1e-6):
        grads = {}
            
        for param_name in self.params:
            grads[param_name] = np.zeros(self.params[param_name].shape)
            itx = np.nditer(self.params[param_name], flags=['multi_index'], op_flags=['readwrite'])
            while not itx.finished:
                idx = itx.multi_index
                
                #This part will iterate for every params
                #You can use self.parmas[param_name][idx] and grads[param_name][idx] to access or modify params and grads
                self.params[param_name][idx]+=delta
                f1=self.forward_pass(X,y,wd_decay)
                self.params[param_name][idx]-=2*delta  
                f2=self.forward_pass(X,y,wd_decay)
                grads[param_name][idx]=(f1-f2)/2/delta
                self.params[param_name][idx]+=delta
                itx.iternext()
        return grads
    
    def get_acc(self, X, y):
        pred = self.forward_pass(X)
        return np.mean(pred == y)
    
    def train(self, X, y, X_val, y_val,
                learning_rate=0, 
                momentum=0, do_early_stopping=False, alpha = 0,
                wd_decay=0, num_iters=10,
                batch_size=4, verbose=False, print_every=10):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        acc_history = []
        val_acc_history = []
        val_loss_history = []
        
        velocity={}
        for param_name in self.params:
            velocity[param_name]=np.zeros(self.params[param_name].shape)
        best_acc=0
        best_params=self.params
        
        for it in range(num_iters):
            
            #Learning rate decay
            if alpha==0:
                learning_rate_a=learning_rate
            if alpha==1:
                if it<=30*iterations_per_epoch:
                    learning_rate_a=learning_rate
                elif it<=60*iterations_per_epoch:
                    learning_rate_a=0.5*learning_rate
                elif it<=90*iterations_per_epoch:
                    learning_rate_a=0.1*learning_rate
                else: 
                    learning_rate_a=0.01*learning_rate
            elif alpha==2:
                learning_rate_a=learning_rate*(1-it/num_iters)
            elif alpha==3:
                learning_rate_a=0.5*learning_rate*(1+math.cos(math.pi*it/num_iters))                
                
            index=np.random.choice(num_train,batch_size)
            X_batch = X[index]
            y_batch = y[index]
            loss = self.forward_pass(X_batch,y_batch,wd_decay=wd_decay)
            grads = self.back_prop(X_batch,y_batch,wd_decay=wd_decay)
            for param_name in self.params:
                velocity[param_name]=momentum*velocity[param_name]-learning_rate_a*grads[param_name]
                self.params[param_name]+=velocity[param_name]
            val_loss = self.forward_pass(X_val,y_val,wd_decay=wd_decay)
            loss_history.append(loss)
            val_loss_history.append(val_loss)
            
            if verbose and it % print_every == 0:
                print('iteration %d / %d: training loss %f val loss: %f' % (it, num_iters, loss, val_loss))
 
            if it % iterations_per_epoch == 0:                
                train_acc = self.get_acc(X_batch, y_batch)
                val_acc = self.get_acc(X_val, y_val)
                acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
            if do_early_stopping:
                if it % iterations_per_epoch == 0:
                    if best_acc-val_acc>0.1:
                        self.params=best_params
                        break
                    else:
                        if val_acc>best_acc:
                            best_acc=val_acc
                            best_params=self.params
                
        return {
          'loss_history': loss_history,
          'val_loss_history': val_loss_history,
          'acc_history': acc_history,
          'val_acc_history': val_acc_history,
        }