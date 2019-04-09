#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:44:09 2019

@author: charu
"""

from __future__ import print_function

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import  Model
from keras.layers import Dense, Activation, Input
from keras.layers import Conv2D, Conv2DTranspose, Concatenate
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from datetime import datetime
import matplotlib.pyplot as plt

#class ElapsedTimer(object):
#    def __init__(self):
#        self.start_time = time.time()
#    def elapsed(self,sec):
#        if sec < 60:
#            return str(sec) + " sec"
#        elif sec < (60 * 60):
#            return str(sec / 60) + " min"
#        else:
#            return str(sec / (60 * 60)) + " hr"
#    def elapsed_time(self):
#        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN1(object):
    def __init__(self, img_rows=64, img_cols=64, channel=3):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.dropout = 0.4
        self.depth = 3
        self.dim = 64
        self.Input_shape=(self.dim, self.dim, self.depth,)
        self.file_name=str(datetime.now())
        self.f=open('gan_%s.txt'% (self.file_name),'w+') 
    # (W−F+2P)/S+1
    def build_discriminator(self):
        if self.D:
            return self.D

        # In: dim x dim x depth
        # Out: 1        
        l0= Input(self.Input_shape)
        
        l1=Conv2D(64,(3,3), strides=1,padding='same')(l0)
        l1=LeakyReLU(alpha=0.2)(l1)
        
        l2=Conv2D(128,(3,3), strides=2,padding='same')(l1)
        l2=BatchNormalization(momentum=0.9)(l2)        
        l2=LeakyReLU(alpha=0.2)(l2)
        
        l3=Conv2D(128,(3,3), strides=1,padding='same')(l2)
        l3=BatchNormalization(momentum=0.9)(l3)        
        l3=LeakyReLU(alpha=0.2)(l3)
        
        l4=Conv2D(256,(3,3), strides=2,padding='same')(l3)
        l4=BatchNormalization(momentum=0.9)(l4)        
        l4=LeakyReLU(alpha=0.2)(l4)
        
        l5=Conv2D(256,(3,3), strides=1,padding='same')(l4)
        l5=BatchNormalization(momentum=0.9)(l5)       
        l5=LeakyReLU(alpha=0.2)(l5) 
        
        l6=Conv2D(512,(3,3), strides=2,padding='same')(l5)
        l6=BatchNormalization(momentum=0.9)(l6)
        l6=LeakyReLU(alpha=0.2)(l6)
        
        l7=Conv2DTranspose(512,(3,3), strides=1,padding='same')(l6)
        l7=BatchNormalization(momentum=0.9)(l7)
        l7=LeakyReLU(alpha=0.2)(l7)
        
        l8=Conv2DTranspose(512,(3,3), strides=2,padding='same')(l7)
        l8=BatchNormalization(momentum=0.9)(l8)
        l8=LeakyReLU(alpha=0.2)(l8)
        
        l9=Conv2DTranspose(128,(3,3), strides=2,padding='same')(l8)
        l9=LeakyReLU(alpha=0.2)(l9)
        
        l10=Dense(1)(l9)
        l11=Activation('sigmoid')(l10)
        
        model = Model(inputs=l0, outputs=l11)
        model.summary(print_fn=self.myprint)
        orgnl_img=Input(self.Input_shape)
        classify = model(orgnl_img)
        return Model(orgnl_img,classify)



    def build_generator(self):
        if self.G:
            return self.G

        # In: dim x dim x depth
        # Out: dim x dim x depth
        l0= Input(self.Input_shape)
        
        l1=Conv2D(64,(3,3), strides=2,padding='same')(l0)
        l1=LeakyReLU(alpha=0.2)(l1)
        
        l2=Conv2D(128,(3,3), strides=2,padding='same')(l1)
        l2=BatchNormalization(momentum=0.9)(l2)
        
        l3=LeakyReLU(alpha=0.2)(l2)
        l3=Conv2D(256,(3,3), strides=2,padding='same')(l3)
        l3=BatchNormalization(momentum=0.9)(l3)
        
        l4=LeakyReLU(alpha=0.2)(l3)
        l4=Conv2D(512,(3,3), strides=2,padding='same')(l4)
        l4=BatchNormalization(momentum=0.9)(l4)
        
        l5=LeakyReLU(alpha=0.2)(l4)
        l5=Conv2D(512,(3,3), strides=2,padding='same')(l5)
        l5=BatchNormalization(momentum=0.9)(l5)
        
        l6=LeakyReLU(alpha=0.2)(l5) 
        l6=Conv2D(512,(3,3), strides=2,padding='same')(l6)
        l6=BatchNormalization(momentum=0.9)(l6)
        l6=LeakyReLU(alpha=0.2)(l6)
        
        l7=Conv2DTranspose(512,(3,3), strides=2,padding='same')(l6)
        l7=BatchNormalization(momentum=0.9)(l7)
        l7=Concatenate(axis=-1)([l7,l5])
        l7=Activation('relu')(l7)
        
        l8=Conv2DTranspose(256,(3,3), strides=2,padding='same')(l7)
        l8=BatchNormalization(momentum=0.9)(l8)
        l8=Concatenate(axis=-1)([l8,l4])
        l8=Activation('relu')(l8)
        
        l9=Conv2DTranspose(128,(3,3), strides=2,padding='same')(l8)
        l9=BatchNormalization(momentum=0.9)(l9)
        l9=Concatenate(axis=-1)([l9,l3])
        l9=Activation('relu')(l9)
        
        l10=Conv2DTranspose(64,(3,3), strides=2,padding='same')(l9)
        l10=BatchNormalization(momentum=0.9)(l10)
        l10=Concatenate(axis=-1)([l10,l2])
        l10=Activation('relu')(l10)
        
        l11=Conv2DTranspose(64,(3,3), strides=2,padding='same')(l10)
        l11=BatchNormalization(momentum=0.9)(l11)
        l11=Activation('relu')(l11)
        
        l12=Conv2DTranspose(3,(3,3), strides=2,padding='same')(l11)
        l13=Activation('tanh')(l12)
        
        model = Model(inputs=l0, outputs=l13)
        model.summary(print_fn=self.myprint)
        orgnl_img=Input(self.Input_shape)
        trfmd_img = model(orgnl_img)
        return Model(orgnl_img,trfmd_img)


    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.discriminator1 = self.build_discriminator()
        self.discriminator1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        return self.discriminator1

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
     
       
        # Build and compile the discriminator0
        self.discriminator0 = self.build_discriminator()
        self.discriminator0.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        # Build the generator
        self.generator0 = self.build_generator()
        self.generator0.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        z = Input(self.Input_shape)
        img0 = self.generator0(z)

#         The discriminator takes generated images as input and determines validity
        validity0 = self.discriminator0(img0)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.adversarial1 = Model(z, validity0)
        self.adversarial1.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                               metrics=['accuracy'])

        return self.adversarial1

    def myprint(self,s):
      print(s, file=self.f)

class MNIST_DCGAN1(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN1()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.build_generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
#        noise_input = None
#        if save_interval>0:
#            noise_input = np.random.uniform(-1.0, 1.0,size=[16,64,64,3])
        for i in range(train_steps):
            images_train =np.random.uniform(-1.0, 1.0, size=[batch_size,64,64,3])#fix it later
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size,64,64,3])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 32,32,1])
            y[batch_size:, :] = 0
            
            d_loss = self.discriminator.train_on_batch(x,y)#fix it now
            y = np.ones([batch_size,32,32,1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 64,64,3])
            a_loss = self.adversarial.train_on_batch(noise, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise.shape[0],\
                        noise=noise, step=(i+1))                  
                    
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples,64,64,3])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            
            images =  np.random.uniform(-1.0, 1.0, size=[samples,64,64,3])

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(5, 5, i+1)
            image = images[i, :, :, :]
#            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN1()
#    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10, batch_size=25, save_interval=1)
#    timer.elapsed_time()
#    mnist_dcgan.plot_images(fake=True)
#mnist_dcgan.plot_images(fake=False, save2file=True)