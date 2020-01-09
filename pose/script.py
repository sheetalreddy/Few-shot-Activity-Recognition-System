import scipy.io as scio
import numpy as np




#load mat file 
annot=scio.loadmat('mpii_human_pose_v1_u12_1.mat')


#.annolist(imgidx) - annotations for image imgidx
#.image.name - image filename
#.annorect(ridx) - body annotations for a person ridx
#.x1, .y1, .x2, .y2 - coordinates of the head rectangle
#.scale - person scale w.r.t. 200 px height
#.objpos - rough human position in the image
#.annopoints.point - person-centric body joint annotations
#.x, .y - coordinates of a joint
#id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
#is_visible - joint visibility
#.vidx - video index in video_list
#.frame_sec - image position in video, in seconds
#img_train(imgidx) - training/testing image assignment
#single_person(imgidx) - contains rectangle id ridx of sufficiently separated individuals
#act(imgidx) - activity/category label for image imgidx
#act_name - activity name
#cat_name - category name
#act_id - activity id
#video_list(videoidx) - specifies video id as is provided by YouTube. To watch video on youtube

no_of_activities=410
mpii_data=[[]]*410
data=annot['RELEASE'] #shape (1,24987)
a=data['annolist'][0][0]
activity=data['act'][0][0]
#img_id=5
for img_id in range(0,a.shape[1]):
     #loop through no of images 
  try:
    data_persons=a['annorect'][0][img_id] # shape w to no_of_persons
    act_info=activity[img_id][0]
    act=act_info['act_name']
    category=act_info['cat_name']
    act_id=act_info['act_id']
    if act_id<0:
      continue 
    print act_id
    print act 
    for r_idx in range(0,data_persons.shape[1]):
      #looping through 1 to person_points size
      points_tf=np.full((1,32),-1,dtype=int)
      single_person=data_persons[0][r_idx]
      scale=single_person['scale']
      center=single_person['objpose']
      points=single_person['annopoints']['point'][0][0]
      array_ids=points['id']  #shape(1,ids of joints available)
     # print array_ids.shape
      points_x=points['x']
      points_y=poits['y']
      points_x_tf=(points_x-center[0])/scale
      points_y_tf=(points_y-center[1])/scale
      set=[]
      for i in range(0,array_ids.shape[1]):
        points_tf[0,2*array_ids[0][i]]=points_x_tf[0][i]
        points_tf[0,2*array_ids[0][i]+1]=points_y_tf[0][i]
     
      set.append(points_tf)
     
    mpii_data[act_id].append(set)   
  except ValueError:
    print "Entered exception"
np.save('mpii.npy',np.asarray(mpii_data))             
           
       






