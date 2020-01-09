import time
import random
import numpy as np
from scipy import misc
import scipy.io as scio
#import cv2
from PIL import Image 
#read file with corresponding file names
File=np.genfromtxt('img_names.txt',delimiter=' ',dtype=None)
File=np.asarray(File)
print File.shape
filename=np.empty([420,20],dtype="S10")
for i in range(0,File.shape[0]):
   file_name=File[i][2]
   act_name=File[i][0]
   sample_no=File[i][1]
   filename[File[i][0]-1,File[i][1]-1]=File[i][2]


#print filename 

# import code
# code.interact(local=dict(globals(), **locals()))
# assert False
TIMIT_IMAGES=np.load('mpii_hpe_inceptionprob.npy')
print TIMIT_IMAGES.shape
TOTAL_CLASSES=TIMIT_IMAGES.shape[0]
samples_per_class=TIMIT_IMAGES.shape[1]

def Normalize(data):
    INPUT_SIZE=data.shape[2]
    data_reshape=np.reshape(data,(TOTAL_CLASSES*samples_per_class,INPUT_SIZE))
    mean=np.mean(data_reshape,axis=0)
    print mean
    mean_clone=np.tile(mean,(TOTAL_CLASSES*samples_per_class,1))
    print mean_clone.shape
    deviation = np.std(data_reshape,axis=0,dtype=float)
    deviation_clone = np.tile(deviation,(TOTAL_CLASSES*samples_per_class,1))
    Normalized_data_reshape=np.divide((data_reshape-mean_clone),deviation_clone)
    Normalized_data=np.reshape(Normalized_data_reshape,(TOTAL_CLASSES,samples_per_class,INPUT_SIZE))
    return Normalized_data
#TIMIT_IMAGES[:,:,32:2080]=Normalize(TIMIT_IMAGES[:,:,32:2080])
timit_train_images = TIMIT_IMAGES[0:350,:,:]
test_set=[20,8,39,45,51]
test_set=[x+351 for x in test_set]
#timit_test_images = TIMIT_IMAGES[350:420,:,:]
timit_test_images=TIMIT_IMAGES[test_set,:,:]
print timit_train_images[0,0,:]
print timit_train_images[0,1,:]


NUM_CLASSES = timit_train_images.shape[0]
INPUT_SIZE=1032

def get_episode(time_steps, classes_per_episode, num_labels, use_test_data):
  timit_images = timit_test_images if use_test_data else timit_train_images
  num_classes, examples_per_class, raw_image_height = timit_images.shape

  # #print use_test_data
  # if use_test_data:
  #   assert np.all(timit _images == timit _test_images)
  # else:
  #   assert np.all(timit _images == timit _train_images)

  # choose classes
  classes = random.sample(range(num_classes), classes_per_episode)
  
  # choose labels
  class_labels = random.sample(range(classes_per_episode), classes_per_episode)
  
  # choose rotation for each class
  #class_rotation = np.random.choice(range(4), classes_per_episode)

  # choose images
  # NOTE: this is actually slower than data_mnist, which it shouldn't be, too much sampling I think
  samples_per_class = random.sample(range(classes_per_episode)*examples_per_class, time_steps)
  indices = [random.sample(range(examples_per_class), samples_per_class.count(i)) for i in range(classes_per_episode)]
  labels = [[class_labels[c]]*len(cs) for c, cs in enumerate(indices)]
  labels = [item for sublist in labels for item in sublist]

  indices = [zip([classes[c]]*len(cs), cs) for c, cs in enumerate(indices)]
  indices = [item for sublist in indices for item in sublist]

  shuffled_order = random.sample(range(time_steps), time_steps)
  labels = [labels[i] for i in shuffled_order]
  indices = [indices[i] for i in shuffled_order]


  indices = zip(*indices)
 # print len(indices[1])
  images_raw = timit_images[indices[0], indices[1], :]
    

  # apply perturbations to each image
  #images = np.zeros([time_steps, IMAGE_HEIGHT*IMAGE_WIDTH,1], dtype=np.float32)
  images = np.zeros([time_steps, INPUT_SIZE], dtype=np.float32)
  for i in range(time_steps):
    #255 - images_raw[i].astype(np.uint8)*255
    im = images_raw[i]
    
      
    images[i] = im
   # images[i]=im
   # print images[i]

  # insert extra labels that are never used
  if classes_per_episode < num_labels:
    mapping = random.sample(range(num_labels), classes_per_episode)
    labels = [mapping[label] for label in labels]

  # convert labels to one-hot
  labels = np.eye(num_labels)[labels]
  last_labels = np.zeros([time_steps, num_labels], dtype=np.float32)
  #labels = np.eye(classes_per_episode)[labels]
  #last_labels = np.zeros([time_steps, classes_per_episode], dtype=np.float32)
  last_labels[1:,:] = labels[:-1,:]

  return images, labels, last_labels

def get_testepisode(time_steps, classes_per_episode, num_labels, use_test_data):
  timit_images = timit_test_images 
  num_classes, examples_per_class, raw_image_height = timit_images.shape
  # choose classes
  classes = random.sample(range(num_classes), classes_per_episode)
   
  # choose labels
  class_labels = random.sample(range(classes_per_episode), classes_per_episode)
  
  # choose rotation for each class
  #class_rotation = np.random.choice(range(4), classes_per_episode)

  # choose images
  # NOTE: this is actually slower than data_mnist, which it shouldn't be, too much sampling I think
  samples_per_class = random.sample(range(classes_per_episode)*examples_per_class, time_steps)
  indices = [random.sample(range(examples_per_class), samples_per_class.count(i)) for i in range(classes_per_episode)]
  labels = [[class_labels[c]]*len(cs) for c, cs in enumerate(indices)]
  labels = [item for sublist in labels for item in sublist]

  indices = [zip([classes[c]]*len(cs), cs) for c, cs in enumerate(indices)]
  indices = [item for sublist in indices for item in sublist]

  shuffled_order = random.sample(range(time_steps), time_steps)
  labels = [labels[i] for i in shuffled_order]
  indices = [indices[i] for i in shuffled_order]
  indices = zip(*indices)
  images_raw = timit_images[indices[0], indices[1], :]
  indices_n=[]
  indices_n[:]=[x+350 for x in indices[0]] 
  org_filenames=filename[indices_n,indices[1]]
  if classes_per_episode < num_labels:
    mapping = random.sample(range(num_labels), classes_per_episode)
    labels = [mapping[label] for label in labels]
  # convert labels to one-hot
  labels = np.eye(num_labels)[labels]
  last_labels = np.zeros([time_steps, num_labels], dtype=np.float32)
  #labels = np.eye(classes_per_episode)[labels]
  #last_labels = np.zeros([time_steps, classes_per_episode], dtype=np.float32)
  last_labels[1:,:] = labels[:-1,:]

  return images_raw, labels, last_labels, org_filenames, indices

def get_batch_of_testepisodes(batch_size, time_steps, classes_per_episode=5, num_labels=5, use_test_data=False):
  images, labels, last_labels,filenames, indices = zip(*[get_testepisode(time_steps, classes_per_episode, num_labels, use_test_data) for _ in range(batch_size)])
  return np.array(images), np.array(labels), np.array(last_labels),np.array(filenames), np.array(indices)

def get_batch_of_episodes(batch_size, time_steps, classes_per_episode=5, num_labels=5, use_test_data=False):
  images, labels, last_labels = zip(*[get_episode(time_steps, classes_per_episode, num_labels, use_test_data) for _ in range(batch_size)])
  return np.array(images), np.array(labels), np.array(last_labels)

############################

# images, labels, last_labels = get_perturbed_batch_of_episodes(25, 50)

# print "starting test"
# start_time = time.time()
# for i in range(1000):
#   ims, lbls, last_lbls = get_perturbed_batch_of_episodes(25,50) # 10.3s for all 1000=25000 episodes
# duration_s = time.time() - start_time
# print "finished test"
# print duration_s

