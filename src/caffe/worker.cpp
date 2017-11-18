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

//class Worker
Worker::Worker(int x){
    this->number_of_threads = x;
}

void Worker::get_threads_num(){
    using namespace std;
    cout << this->number_of_threads << endl;
}

void Worker::create_threads(){
    std::thread t[this->number_of_threads];

    for (int i = 0; i < this->number_of_threads; ++i){
        workloads[i] = NULL;
        t[i] = std::thread(&Worker::worker_thread, this, i);
        t[i].detach();
        threads.push_back(&t[i]);
        std::cout << "thread " << i << "creating" << std::endl;
    }

}

void Worker::destroy_threads(){
    for (int i = 0; i < this->number_of_threads; ++i){
        (threads[i])->join();
        std::cout << "thread " << i << "finished" << std::endl;
    }
    threads.clear();
}

void Worker::worker_thread(int tid){
    cpu_set_t my_set;        /* Define your cpu_set bit mask. */
    CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
    CPU_SET(tid, &my_set);     /* set the bit that represents core 7. */
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

    using namespace std;
    std::cout << "thread " << tid << "running" << std::endl;
    while (1){
        if (workloads[tid] == NULL){
            usleep(1); // TODO: if delete, program stuck.
            // cout << tid << "waiting" << endl;
            continue;
        }

        cout << "executing thread " << tid << " of " << this->number_of_threads-1<< endl;
        //start computing
        for (int i = this->workloads[tid]->start[tid]; i <= this->workloads[tid]->end[tid]; ++i){
            float layer_loss = workloads[tid]->myNet->layers_[i]->Forward(workloads[tid]->myNet->bottom_vecs_[i], workloads[tid]->myNet->top_vecs_[i]);
            workloads[tid] -> loss += layer_loss;
        }
        //end computing

        if (tid == this->number_of_threads - 1){
            this->workloads[tid]->finished = 1;
            this->workloads[tid] = NULL;
        }else{
            while(this->workloads[tid+1] != NULL)usleep(1);
            this->workloads[tid+1] = this->workloads[tid];
            this->workloads[tid] = NULL;
        }
    }
}

void Worker::assign_workload(Workload &w){
    // using namespace std;
    while(this->workloads[0] != NULL)usleep(1);
    this->workloads[0] = &w;
    // cout << "aaaaaa" << (this->workloads[0]->finished) << endl;
    // workloads[1] = workloads[0];
}


//class Workload
Workload::Workload(int x){
    memset(this->start, 0, sizeof(this->start));
    memset(this->end, 0, sizeof(this->end));
    this->num = x;
    this->finished = -1;
    this-> myNet = NULL;
    this->loss = 0.0f;
}

void Workload::set_net(Net<float>&n){
    this->myNet = &n;
}

void Workload::testNet(){
    using namespace std;
    if (this->myNet == NULL){
        cout << "mynet null error" << endl;
        exit(-1);
    }
    cout << this->myNet->debug_info_ << endl;
}

int Workload::get_start(int index){
    using namespace std;
    if (index >= num){
        cout << "index error!" << endl;
        exit(-1);
    }
    return start[index];
}

int Workload::get_end(int index){
    using namespace std;
    if (index >= num){
        cout << "index error!" << endl;
        exit(-1);
    }
    return end[index];
}

void Workload::set_start(int index, int val){
    using namespace std;
    if (index >= num){
        cout << "index error!" << endl;
        exit(-1);
    }
    start[index] = val;
}

void Workload::set_end(int index, int val){
    using namespace std;
    if (index >= num){
        cout << "index error!" << endl;
        exit(-1);
    }
    end[index] = val;
}


}
