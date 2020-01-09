from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import os.path
from extractor import Extractor
import os 
# get the model.

os.environ['CUDA_VISIBLE_DEVICES']='1'

def extract(self, image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model = InceptionV3(weights='imagenet',include_top=True)
    model = Model(inputs=base_model.input,outputs=base_model.get_layer('predictions').output)
    features = model.predict(x)
    features = features[0]
    return features


mpii_datapath = '/mnt/qnap/malredd/workspace/active-oneshot-learning/mpii_human_pose_v1/images'
# Loop through data.
for frames in os.listdir(mpii_datapath):

    # Get the path to the sequence for this video.
    features_path = '/mnt/qnap/malredd/workspace/active-oneshot-learning/mpii_human_pose_v1/inception_features/'+frames.split('.')[0]+'-features.txt'
    features_path_npy = '/mnt/qnap/malredd/workspace/active-oneshot-learning/mpii_human_pose_v1/inception_features/'+frames.split('.')[0]+'-features.npy'
    image_path=mpii_datapath+'/'+frames
  # for image in frames:
    features = extract(image_path)
    features = np.array([features])
    #sequence.append(features)

    # Save the sequence.
    
    np.savetxt(features_path, features, delimiter=',',fmt='%1.3f')


