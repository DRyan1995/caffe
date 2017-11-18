#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "thread"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define MAX_THREAD_NUM 10

namespace caffe{

class Workload{
public:
    Workload(int);
    int num; // thread number
    int get_start(int index);
    int get_end(int index);
    void set_start(int index, int val);
    void set_end(int index, int val);
    int finished;
private:
    int start[MAX_THREAD_NUM];
    int end[MAX_THREAD_NUM];
// friend class Worker;
};

class Worker{
public:
    Worker(int);
    void get_threads_num();
    void create_threads();
    void destroy_threads();
    void worker_thread(int tid);
    void assign_workload(Workload &);
    Workload * workloads[MAX_THREAD_NUM];
private:
    std::vector<std::thread * > threads;
    int number_of_threads;

};
}
