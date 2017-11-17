#include "cstdio"
#include "iostream"

using namespace std;

class myClass
{
// private:

public:
    myClass(int t);
    int x;    
};

myClass::myClass(int t){
    this->x = t;
}

void print(myClass * c){
    cout << (c->x);
}

int main(int argc, char* argv[]){
    myClass test(1);
    print(&test);

    return 0;
}
