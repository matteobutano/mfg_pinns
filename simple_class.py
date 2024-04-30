from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import json
import os 
import shutil


class simple_mfg:
    
    def __init__(self, ts):
        
        self.dims = 2
        
        with open('config.json') as f:
            var_config = json.loads(f.read())
        
        self.lx = var_config['lx']
        
        self.N_points = var_config['N_points']
        self.N_b = self.N_points//4
        self.N_nodes = var_config['N_nodes']
        self.N_layers  = var_config['N_layers']
        self.learning_rate = var_config['learning_rate']
        self.training_steps = var_config['training_steps']
        
        self.weight_PDE = var_config['weight_PDE']
        self.weight_b = var_config['weight_b']
        self.weight_cyl = var_config['weight_cyl']
        
        self.pot = var_config['pot']
        self.mu = var_config['mu']
        self.m_0 = var_config['m_0']
        self.xi = var_config['xi']
        self.c_s = var_config['c_s']
        self.R = var_config['R']
        self.sigma = np.sqrt(2*self.xi*self.c_s)
        self.g = -(self.mu*self.sigma**4)/(2*self.m_0*self.xi**2)
        self.l = -self.g*self.m_0
        self.ts = ts
        
        if not os.path.exists('gfx/'+ str(self.ts)):
            
            os.mkdir('gfx/'+ str(self.ts))
            shutil.copyfile('config.json', 'gfx/'+ str(self.ts)+ '/config.json')
    
        else: 
            
            shutil.copyfile('config.json', 'gfx/'+ str(self.ts)+ '/config.json')
        
        self.loss_history = []
        
        self.model = Sequential()
        self.model.add(Dense(self.N_nodes, input_shape=(self.dims,), activation=None))

        for i in range(self.N_layers):
            self.model.add(Dense(self.N_nodes, activation='relu'))
            self.model.add(Dense(self.N_nodes, activation='elu'))
            self.model.add(Dense(self.N_nodes, activation='selu'))

        self.model.add(Dense(1, activation=None))
        if self.dims ==1:
            self.points = tf.Variable(tf.random.uniform((self.N_points, 1), -self.lx,self.lx)) 
            
            
            self.dx = 0.05
            self.nx = int(2*self.lx/self.dx) + 1 
            self.x = np.linspace(-self.lx,self.lx,self.nx)
            
            # Function that computes p given the rule (see pdf)
            def jacobi(p):
                
                def L2_error(p, pn):
                    return np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2)) 
                
                U = np.zeros(self.nx) 
                U[np.abs(self.x) < self.R] = -10e5
                
                p[0] = np.sqrt(self.m_0)
                p[-1] = np.sqrt(self.m_0)
                
                l2_error = 1
                
                m = np.zeros(self.nx) + self.m_0
    
                while l2_error > 10e-9:
                    
                    pn = p.copy()
    
                    p[1:-1] = 0.5*self.mu*self.sigma**4*(pn[2:] + pn[:-2])/(self.mu*self.sigma**4 - (U[1:-1] + self.g*m[1:-1] + self.l)*self.dx**2)
    
                    m = 0.01*p**2 + 0.99*pn
                    
                    l2_error = L2_error(p,pn)
                    
                return p
            
            self.p = np.zeros(self.nx) + np.sqrt(self.m_0)
            self.p = jacobi(self.p)
            
            print('Real solution is computed, learning begins.')
               
        else: 
            self.points = tf.Variable(tf.random.uniform((self.N_points, 2), -self.lx,self.lx))
            x_b1 = 2*self.lx*tf.keras.backend.random_bernoulli((int(self.N_b/2), 1), 0.5)-self.lx
            y_b1 = tf.random.uniform((int(self.N_b/2), 1), -self.lx, self.lx)
            y_b2 = 2*self.lx*tf.keras.backend.random_bernoulli((int(self.N_b/2), 1), 0.5)-self.lx
            x_b2 = tf.random.uniform((int(self.N_b/2), 1), -self.lx, self.lx)
            x_b = tf.concat([x_b1, x_b2], axis=0)
            y_b = tf.concat([y_b1, y_b2], axis=0)
            self.points_b = tf.concat([x_b, y_b], axis=1)
            
            
    def get_L2_loss(self,verbose):
        
        with tf.GradientTape() as phi_tape_1:
            with tf.GradientTape() as phi_tape_2:
                phi = self.model(self.points)
            grad_phi = phi_tape_2.gradient(phi,self.points)      
        
        jac_phi = phi_tape_1.gradient(grad_phi, self.points)
        lap_phi = tf.math.reduce_sum(jac_phi,axis = 1,keepdims=True)
         
        U0 = np.zeros(shape = (self.points.shape[0],1))
        U0[np.linalg.norm(self.points,axis=1) < self.R] = self.pot
        U0 = tf.constant(U0,dtype='float32')
        m = phi**2 
        V_pot = U0 + self.g * m
        
        res_PDE = tf.reduce_mean((0.5*self.mu*self.sigma**4*lap_phi + V_pot*phi+self.l*phi)**2)
        
        if self.dims == 1:
            res_b = tf.reduce_mean(tf.abs(self.model(tf.constant([-self.lx,self.lx],dtype = 'float32'))-[np.sqrt(self.m_0),np.sqrt(self.m_0)])**2)
        else: 
            res_b = tf.reduce_mean(tf.abs(self.model(self.points_b)-np.sqrt(self.m_0)))
        
        res_cyl = tf.reduce_mean((self.model(self.points[np.linalg.norm(self.points,axis=1) < self.R]))**2)
        
        if verbose: 
            print('        {:10.3e}       {:10.3e}       {:10.3e}       |  {:10.3e}'.format(res_PDE,res_b,res_cyl,res_PDE + res_b + res_cyl))
        
        return self.weight_PDE*res_PDE + self.weight_b*res_b + self.weight_cyl*res_cyl
    
    
    def train_step(self,verbose,learning_rate):
        
        lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,decay_steps=10000, decay_rate=0.9)
        
        optimizer = tf.optimizers.Adam(learning_rate = lr_schedule)
        
        with tf.GradientTape() as f_tape:
            f_loss = self.get_L2_loss(verbose)
            f_grad = f_tape.gradient(f_loss,self.model.trainable_variables)
            
        optimizer.apply_gradients(zip(f_grad, self.model.trainable_variables))
        
        return f_loss
     
    def train(self,frequency = 50):
        
        training_steps = self.training_steps
        learning_rate = self.learning_rate
        
        print('    #iter         res_PDE          res_b          res_cyl        |   Loss_total')
        print('--------------------------------------------------------------')
        print('    ',end="")
        
        step = 1
        loss = 100
        
        while (step < training_steps and loss > 10e-6):
            if step % frequency == 0: 
                print('{:6d}'.format(step),end="")                        
                loss = self.train_step(True,learning_rate)
                self.loss_history.append(loss)
                print('    ',end="")
            else:
                self.train_step(False,learning_rate)
            step = step + 1
        
        self.model.save_weights('gfx/'+ str(self.ts) + '/model.h5')
    
    def warmstart(self,training_steps = 1000,learning_rate = 10e-5):

        print('      #iter         |   Loss_total')
        print('----------------------------------')
        print('    ',end="")
        step = 1
        loss = 100
        
        while (step <= training_steps and loss > 10e-3):
            
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
         
            with tf.GradientTape() as f_tape:
                res_1 = tf.reduce_mean((self.model(self.points) - 1)**2)
                res_2 = tf.reduce_mean((self.model(self.points[np.linalg.norm(self.points,axis=1) < self.R]))**2)
                f_loss = res_1 + res_2
                f_grad = f_tape.gradient(f_loss,self.model.trainable_variables)
                
            optimizer.apply_gradients(zip(f_grad, self.model.trainable_variables))
            
            if step % 50 == 0: 
                print('{:6d}'.format(step),end="")                        
                print('        {:10.3e}'.format(f_loss))
                print('    ',end="")
           
            step = step + 1
  
    
    def draw(self):
        plt.figure(figsize=(8,8))
        plt.scatter(self.points[:,0], self.points[:,1], s=13, c=self.model(self.points)**2, cmap='hot_r')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=30) 
        plt.savefig('gfx/'+ str(self.ts) + '/model.png')
        
    def draw_history(self):
        plt.figure(figsize=(8,8))
        plt.plot(self.loss_history)
        plt.title('Loss')
        plt.savefig('gfx/'+ str(self.ts) + '/history.png')
        
        
        
