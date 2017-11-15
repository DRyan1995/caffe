// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include "vector"

using namespace std;

int RUN_LOCK = 0;

void foo(int tid)
{
    while(tid != RUN_LOCK);
    cout << "foo : "   << tid << endl;
    cout << "thread " << tid << "finished" << endl;

    RUN_LOCK ++;
}


int main()
{
    vector<thread> thread_list;
  for (int i = 0; i < 4; ++i)
  {
      thread_list.push_back(thread(foo, i));
  }
    for (int i = 3; i >= 0; --i)
  {
      thread_list[i].join();
      cout << "thread " << i << "exiting ..." << endl;
  }
  return 0;
}
