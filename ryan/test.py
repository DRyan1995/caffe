
import caffe
import time

w = caffe.Workload(4)
w2 = caffe.Workload(4)
# a.set_start(4,2)
# print a.get_start(1)


worker = caffe.Worker(4)
worker.get_threads_num()
worker.create_threads()

worker.assign_workload(w)
worker.assign_workload(w2)
# worker.destroy_threads()
while w2.finished < 0:
    time.sleep(.01)
    pass
