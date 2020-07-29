#include <unistd.h>
#include <limits.h>
#include <iostream>
#include <schema.h>
#include <sys.types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <cmath>
#include <random>
std::mt19937 generator (123);
std::uniform_real_distribution<double> distribution(0.0, 1.0);
int write_fd;
int count = 0;
Limits limit_object();
static struct sigaction sa;
void warning_handler(int sig, struct siginfo_t* sig_details,void* context)
{
    //handles the signal 
}
void run_state()
{
    double prev_speed = 0.0;//starting from scratch
    int prev_gear = 0;
    int pid = getpid();
    while(true)
    {
        Schema obj();
        obj.vehicle_id =  pid;
        obj.database_index = count;
        obj.oil_life_pct = limit_object.min_oil_level + (1-limit_object.min_oil_level)*distribution(generator);
        obj.tire_p_rl = limit_object.min_pressure + (limit_object.max_pressure + limit_object.min_pressure)*distribution(generator);
        obj.tire_p_rr = limit_object.min_pressure + (limit_object.max_pressure + limit_object.min_pressure)*distribution(generator);
        obj.tire_p_fl = limit_object.min_pressure + (limit_object.max_pressure + limit_object.min_pressure)*distribution(generator);
        obj.tire_p_fr = limit_object.min_pressure + (limit_object.max_pressure + limit_object.min_pressure)*distribution(generator);
        obj.batt_volt = limit_object.min_voltage + (limit_object.max_voltage-limit_object.min_voltage)*distribution(generator);
        obj.fuel_percentage = limit_object.min_fuel_percentage + (1-limit_object.min_fuel_percentage)*distribution(generator);
        double new_speed = 0;
        if(prev_speed < limit_object.max_speed * 0.9)
        {
            new_speed = prev_speed + distribution(generator)*2;
            int new_gear;
            if(new_speed < 10)
                new_gear = 1;
            else if(new_speed < 30)
                new_gear = 2;
            else if(new_speed < 50)
                new_gear = 3;
            else if(new_speed < 70)
                new_gear = 4;
            else
                new_gear = 5;
            obj.gear_toggle = (prev_gear != new_gear);
            obj.clutch = (old_gear != new_gear); 
            obj.speed = new_speed;
            obj.accel = true;
            prev_gear = new_gear;
            prev_speed = new_speed;
        }
        else
        {
            new_speed = prev_speed - distribution(generator)*2;
            int new_gear;
            if(new_speed < 10)
                new_gear = 1;
            else if(new_speed < 30)
                new_gear = 2;
            else if(new_speed < 50)
                new_gear = 3;
            else if(new_speed < 70)
                new_gear = 4;
            else
                new_gear = 5;
            obj.gear_toggle = (prev_gear != new_gear);
            obj.clutch = (old_gear != new_gear); 
            obj.speed = new_speed;
            obj.accel = true;
            prev_gear = new_gear;
            prev_speed = new_speed;   
        }
        obj.hard_brake = false;
        obj.door_lock = true;
        write(write_fd,&bj,sizeof(obj));//option 1- using pipes for inter process communication.
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
            std::cout<<"Error creating Car #"<<i<<'\n';
            return;
        }
        else if(pid == 0)
        {//child process
            sa.sa_sigaction = *warning_handler;
            sa.sa_flags |= SA_SIGINFO;
            if(signal(SIGUSR1,&sa,NULL) != 0)
            {
                std::cout<<"Error while initializing the signal handler!\n";
            }
            run_state();
            exit(0);
        }
        else
            count++;//parent process 
    }
    wait(NULL);//wait for all cars to finish.
}