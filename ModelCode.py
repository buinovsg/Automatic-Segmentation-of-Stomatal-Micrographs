# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:51:53 2019

@author: Greta_Buinovskaja_and_David_McKenna
"""

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import data, model
model_obj = model.unet()
data_gen_args = dict(rotation_range=180,
                     width_shift_range=0,
                     height_shift_range=0,
                     shear_range=45,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='reflect')
myGene = data.trainGenerator(1,'/content/drive/My Drive/Stomata_Project/training_data/train','image','label',data_gen_args,save_to_dir = '/content/drive/My Drive/Stomata_Project/training_data/train/aug')

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Stomata_Project/training_data/unet_stomata_steps_100_epochs_6.hdf5', monitor='loss',verbose=1, save_best_only=True)
history=model_obj.fit_generator(myGene,steps_per_epoch=100,epochs=6,callbacks=[model_checkpoint])

import data as data3
import model
num_images=2
model_obj = model.unet(pretrained_weights = '/content/drive/My Drive/Stomata_Project/training_data/unet_stomata_1.hdf5')
testGene = data3.testGenerator('/content/drive/My Drive/Stomata_Project/training_data/train/test', num_images)
results = model_obj.predict_generator(testGene,num_images,verbose=1)
data3.saveResult('/content/drive/My Drive/Stomata_Project/training_data/train/test_output_1',results)

import cv2
import os
def image_resizer(file_name):
  print (file_name)
  if ('true' in file_name) or ('resize0 in file_name'): return
  
  image=cv2.imread(file_name)
  original_image=cv2.imread(file_name.replace('_output', '').replace('_predict.png', '.jpg'))
  size=original_image.shape
  
  print(size)
  image_resized=cv2.resize(image, dsize=(size[1], size[0]))
  cv2.imwrite(file_name.replace('.png', '.resize.png'), image_resized)
folder='/content/drive/My Drive/Stomata_Project/training_data/train/test_output_1/'
files_to_process=[image_resizer(folder+file_name) for file_name in os.listdir(folder)]

import numpy as np
import matplotlib.pyplot as plt
def accuracy(predicted_file, true_file, threshold=0.35, plotting = False):
  predicted_image=cv2.imread(predicted_file)
  true_image=cv2.imread(true_file)
  
  print (predicted_image.shape, true_image.shape)
  predicted_image=np.sum(predicted_image, axis=-1) 
  true_image=np.sum(true_image, axis=-1)
  
  print (np.max(predicted_image), np.max(true_image))
  predicted_image = np.divide(predicted_image, np.max(predicted_image))
  true_image = np.divide(true_image, np.max(true_image))
  filtered_image=predicted_image.copy()
  filtered_image[filtered_image<threshold]=0.
  filtered_image[~(filtered_image<threshold)]=1.
  cv2.imwrite(predicted_file.replace('.', '.thresholded_{0}.'.format(threshold)), filtered_image)
  
  true_image_bool=true_image==1.
  filtered_image_bool = filtered_image == 1.
  true_positive=np.sum(filtered_image_bool[true_image_bool])
  false_positive=np.sum(filtered_image_bool[~true_image_bool])
  
  total_true=np.sum(true_image_bool)
  false_negative=np.sum(~filtered_image_bool[true_image_bool])
  true_negative=np.sum(~filtered_image_bool[~true_image_bool])
  
  total_false=np.sum(~true_image_bool)
 
  #accuracy
  accuracy=(true_positive+true_negative)/np.prod(true_image.shape)
  #precision
  precision=(true_positive)/(true_positive+false_positive)
  #recall
  recall=true_positive/(true_positive+false_negative)
  #F1
  f1=2*(precision*recall)/(precision+recall)
  
  print ('accuracy: {}, precision: {}, recall: {}, f1:{}'.format(accuracy, precision, recall, f1))
  
  probability_delta=predicted_image-true_image
  probability_delta_2 = filtered_image - true_image
  if plotting:
    plt.imshow(probability_delta, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.grid(False)
    plt.savefig('/content/drive/My Drive/Stomata_Project/probability_diff.png', dpi=340)
    plt.show()
    
    plt.imshow(filtered_image_bool)
    plt.gcf().set_facecolor('gray')
    plt.axis('off')
    plt.savefig('/content/drive/My Drive/Stomata_Project/filtered_image_{0}.png'.format(threshold), dpi=340)
    plt.show()
    
    plt.hist(probability_delta.reshape(-1), bins=16, edgecolor='black', linewidth=0.5)
    plt.ylabel('Nr. of pixels')
    plt.xlabel('Δ probability from reference mask')
    plt.yscale('log')
    plt.savefig('/content/drive/My Drive/Stomata_Project/probability_histgr.png', dpi=340)
    plt.show()

    plt.imshow(probability_delta_2, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.axis('off')
    plt.grid(False)
    plt.savefig('/content/drive/My Drive/Stomata_Project/probability_diff_delta_2.png', dpi=340)
    plt.show()
    
    plt.hist(probability_delta_2.reshape(-1), edgecolor='black', linewidth=0.5)
    plt.ylabel('Nr. of pixels')
    plt.xlabel('Δ probability from reference mask')
    plt.yscale('log')
    plt.savefig('/content/drive/My Drive/Stomata_Project/probability_diff_delta_2_diff.png', dpi=340)
    plt.show()

  true_positive/=total_true
  false_positive/=total_false
  false_negative/=total_true
  true_negative/=total_false
  
  print('TP: {}, FP: {}, FN: {}, TN: {}'.format(true_positive, false_positive, false_negative, true_negative))
  
  return [true_positive, true_negative, false_positive, false_negative, accuracy, precision, recall, f1]

accuracy('/content/drive/My Drive/Stomata_Project/training_data/train/test_output/0_predict.resize.png', '/content/drive/My Drive/Stomata_Project/training_data/train/test_output/0true.png',plotting = True)

input_file = '/content/drive/My Drive/Stomata_Project/training_data/train/test_output/0_predict.resize.png'
true_file = '/content/drive/My Drive/Stomata_Project/training_data/train/test_output/0true.png'
testSpace = np.linspace(0.001, 0.9, 600 )
testScanArr = [accuracy(input_file, true_file, threshold = testVal) for testVal in testSpace]
testScanArr = np.vstack(testScanArr)

labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative', 'Accuracy', 'Precision', 'Recall', 'F1']

plt.figure(1)
for idx, [label, data] in enumerate(zip(labels, testScanArr.T)):
  if idx == 4:
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Proportion')
    plt.savefig('/content/drive/My Drive/Stomata_Project/performance.png', dpi=340)
    
    plt.figure(2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Performance proportion')
 
  plt.plot(testSpace, data, label = label)
  plt.legend()
  plt.savefig('/content/drive/My Drive/Stomata_Project/performance2.png', dpi=340)
plt.show()


