'''
Title           :make_predictions_2.py
Description     :This script makes predictions using the 2nd trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_2.py
python_version  :2.7.11
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import time

caffe.set_mode_cpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img
start = time.time()

'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/media/ryan/HDD/deeplearning-cats-dogs-tutorial/input/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/media/ryan/HDD/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt',
                '/media/ryan/HDD/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))


'''
Making predicitions
'''
##Reading image paths

# test_ids = []
# preds = []

# totalTimeSingle = 0
# totalTimeMulti = 0
#Making predictions
MAXI = 20
for imgi in range(1, MAXI+1):
    # net = caffe.Net('/media/ryan/HDD/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt',
    #             '/media/ryan/HDD/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1_iter_10000.caffemodel',
    #             caffe.TEST)
    img_path = "/media/ryan/HDD/deeplearning-cats-dogs-tutorial/input/test2/{}.jpg".format(imgi)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    t0 = time.time()
    # out = net.forward()
    out = net.forward_threaded()
    pred_probas = out['prob']
    # test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    # preds = preds + [pred_probas.argmax()]
    print img_path
    print pred_probas.argmax()
    print '-------\n\n\n'
end = time.time()

print "exec time: {}s" .format(end - start)
# print "\n\n\n total single thread time = {}s \n multi-thread time = {}s".format(totalTimeSingle, totalTimeMulti)

# '''
# Making submission file
# '''
# with open("../caffe_models/caffe_model_2/submission_model_2.csv","w") as f:
#     f.write("id,label\n")
#     for i in range(len(test_ids)):
#         f.write(str(test_ids[i])+","+str(preds[i])+"\n")
# f.close()
