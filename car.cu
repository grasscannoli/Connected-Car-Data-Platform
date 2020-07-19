#include <unistd.h>
#include <iostream>
#include <schema.cu>
#include <sys/stat.h>
#include <cmath>
int write_fd;
void run_state()
{
    while(true)
    {
        Schema obj();
        obj.vehicle_id =  getpid();
        obj.
        
        write(write_fd,&bj,sizeof(obj));
    }
}

void initialize(int numberOfCars,int* file_descriptor)
{
    int i;
    srand(time(NULL));
    write_fd = file_descriptor[1];
    for(i=0;i<numberOfCars;i++)
    {
        int pid = fork();
        if(pid < 0)
        {
            cout<<"Error creating Car# "<<i<<'\n';
            return;
        }
        else if(pid == 0)
        {
            run_state();
            exit(0);
        }
    }
}