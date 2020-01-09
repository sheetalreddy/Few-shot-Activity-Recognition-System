import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model_act_sup as model
from helpers import accuracies
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', action='store', choices=['mnist', 'omniglot','timit','ucf','mpii','mpii_f'], default='mpii_f', help='the dataset to use')
parser.add_argument('-c', '--cell_type', action='store', choices=['lstm', 'ntm'], default='lstm', help='the type of rnn cell to use')
parser.add_argument('-cpe', '--classes_per_episode', type=int, default=5, help='the number of classes to use per episode')
args = parser.parse_args()

model.init(args)

if args.dataset == 'mnist':
  import data_mnist as data
elif args.dataset == 'omniglot':
  import data_omniglot as data
elif args.dataset=='timit' :
  import data_timit as data
elif args.dataset=='ucf':
  import data_ucf as data
  INPUT_SIZE=2048
elif args.dataset=='mpii':
  import data_mpii_hpe_in as data
  INPUT_SIZE=2080
elif args.dataset=='mpii_f':
  import data_mpii_hpe_inception as data
  INPUT_SIZE=1032


NUM_EPISODES = 300000#100000
CLASSES_PER_EPISODE = args.classes_per_episode#5#2
NUM_LABELS = CLASSES_PER_EPISODE
EPISODES_PER_LOG = 100
BATCH_SIZE = 25#25
TIME_STEPS = 10*CLASSES_PER_EPISODE#50
RUN_NAME = '%s_%s_%dc_%02dbatch_sup'%(args.dataset, model.CELL_TYPE, CLASSES_PER_EPISODE, BATCH_SIZE)
n_runs = max([0]+[int(d.split('_')[-1]) for d in glob.glob('logs_experiments/'+RUN_NAME+'_*')])
RUN_NAME += '_%03d'%(n_runs+1)

images_t = tf.placeholder(tf.float32, (None, TIME_STEPS, INPUT_SIZE), "images")
labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, CLASSES_PER_EPISODE), "labels")
last_labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, CLASSES_PER_EPISODE), "last_labels")

logits_t = model.inference(images_t, last_labels_t)

#logits_t = tf.reshape(logits_t, (-1, 5))
loss_t = model.loss(logits_t, labels_t)
#num_labels = labels_t.shape[2]
#print'num_labels=', num_labels
#with tf.variable_scope("loss"):
#labels_t = tf.reshape(labels_t, (-1, 5))
#loss_t=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_t,labels=labels_t))


train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_t)
dummy_op = tf.no_op()
init_op = tf.initialize_all_variables()

loss_summary_t = tf.placeholder("float", [])
accuracy_summary_t = tf.placeholder("float", [])
accuracy_1st_summary_t = tf.placeholder("float", [])
accuracy_2nd_summary_t = tf.placeholder("float", [])
accuracy_5th_summary_t = tf.placeholder("float", [])
accuracy_10th_summary_t = tf.placeholder("float", [])
tf.summary.scalar('loss', loss_summary_t)
tf.summary.scalar('accuracy', accuracy_summary_t)
tf.summary.scalar('accuracy_01st', accuracy_1st_summary_t)
tf.summary.scalar('accuracy_02nd', accuracy_2nd_summary_t)
tf.summary.scalar('accuracy_05th', accuracy_5th_summary_t)
tf.summary.scalar('accuracy_10th', accuracy_10th_summary_t)
merged_summaries_op = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=1000)
sess = tf.Session()
sess.run(init_op)

train_writer = tf.summary.FileWriter("logs_experiments/%s/train"%(RUN_NAME), sess.graph)
test_writer = tf.summary.FileWriter("logs_experiments/%s/test"%(RUN_NAME), sess.graph)

# x = []
# y_1st = []
# y_2nd = []
# y_5th = []
# y_10th = []
for e in range(NUM_EPISODES):

  images, labels, last_labels = data.get_batch_of_episodes(BATCH_SIZE, TIME_STEPS, CLASSES_PER_EPISODE, use_test_data=False)
  logits, loss, _ = sess.run((logits_t, loss_t, train_op), {images_t: images, labels_t: labels, last_labels_t: last_labels})
  if not (e+1)%EPISODES_PER_LOG:
    if not (e+1)%(EPISODES_PER_LOG*10):
      saver.save(sess, 'logs_experiments/%s/model_%d.ckpt'%(RUN_NAME, e+1))
    true_labels = np.argmax(labels, axis=2)
    pred_labels = np.argmax(logits, axis=2)
    accuracy = np.mean(true_labels == pred_labels)
    accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th = accuracies(true_labels, pred_labels, labels.shape[2])
    train_writer.add_summary(sess.run(merged_summaries_op, {loss_summary_t: loss, accuracy_summary_t: accuracy, accuracy_1st_summary_t: accuracy_1st, accuracy_2nd_summary_t: accuracy_2nd, accuracy_5th_summary_t: accuracy_5th, accuracy_10th_summary_t: accuracy_10th, }), e+1)
    train_writer.flush()
    print e+1, loss, accuracy, accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th
    
    
