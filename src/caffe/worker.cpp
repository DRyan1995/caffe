#include <vector>
#include <thread>
#include "iostream"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/worker.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe{

Worker::Worker(int x){
    this->x = x;
}

void Worker::Display(){
    using namespace std;
    cout << this->x << endl;
}
}
