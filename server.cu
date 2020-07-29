#include <string>
#include <vector>
#include <map>
#include <cuda.h>
#include <limits.h>
#include <schema.h>
#include <thrust/device_vector.h>
Limits l();
int nt, nb;
void init_bt(int val){
    // initialize the num thread blks and grids:
    if(val <= 1024){
        nt = val;
        nb = 1;
    }
    else{
        nt = 1024;
        nb = (val+1024-1)/1024;
    }
}

__global__ void changeRowsKernel(int numberOfRowsToBeModified,Schema* deviceRowsToBeModified, Schema* StateDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfRowsToBeModified)
    {
        StateDatabase[deviceRowsToBeModified[id].database_index] = deviceRowsToBeModified[id];
    }
}

__global__ void selectKernel(
    int* StateDatabase,
    int numberOfRows,
    int numberOfAttributes,
    int* selectedValues,
    int selectionCol,
    int conditionCol,
    int conditionValue,
    int* endIndexSelectedValues
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfAttributes && StateDatabase[id*numberOfAttributes+conditionCol] == conditionValue){
        int i = atomicAdd(endIndexSelectedValues, 1);
        selectedValues[i] = StateDatabase[id*numberOfAttributes+selectionCol];
    }
}

class DatabaseManagementSystem{
private:
    Schema* StateDatabase;//The table is represented by an object array. Primary keys are as intended.
    int* anomaly_states;//This table is to track state transitions for anomaly detection.
    int num_states;
    map<int,Schema> work_list;//stores a worklist of vaious queries to lazily update. State updates happen here.
    const int numberOfRows;
public:
    DatabaseManagementSystem(
        int numRows, 
        Schema* initTable
        int* anomaly_states;
    ):
        numberOfRows(numRows)
    {
        cudaMalloc(&StateDatabase, numberOfRows*sizeof(Schema));
        num_states = 10;
        anomaly_states = (int*)calloc(num_states * numberOfRows,sizeof(int));
        cudaMemcpy(StateDatabase, initTable, numberOfRows*sizeof(Schema), cudaMemcpyHostToDevice);        
    }

    void state_update(Schema& s)
    {
        int ind = s.database_index;
        int* row = (anomaly_states + num_states*ind);
        if(s.oil_life_pct < l.min_oil_level)
        {
            row[0] = min(row[0]+1,l.oil_violation_time);
            if(row[0] == l.oil_violation_time)
                //anomaly
        }
        else
            row[0] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[1] = min(row[1]+1,l.pressure_violation_time);
            if(row[1] == l.pressure_violation_time)
                //anomaly
        }
        else
            row[1] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[2] = min(row[1]+1,l.pressure_violation_time);
            if(row[2] == l.pressure_violation_time)
                //anomaly
        }
        else
            row[2] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[3] = min(row[1]+1,l.pressure_violation_time);
            if(row[3] == l.pressure_violation_time)
                //anomaly
        }
        else
            row[3] = 0;
        if(s.tire_p_rl < l.min_pressure)
        {
            row[4] = min(row[1]+1,l.pressure_violation_time);
            if(row[4] == l.pressure_violation_time)
                //anomaly
        }
        else
            row[4] = 0;
        if(s.batt_volt < s.min_voltage)
        {
            row[5] = min(row[5]+1,l.voltage_violation_time);
            if(row[5] == l.voltage_violation_time)
                //anomaly
        }
        else
            row[5] = 0;
        if(s.fuel_percentage < l.min_fuel_percentage)
        {
            row[6] = min(row[6]+1,l.fuel_violation_time);
            if(row[6] == l.fuel_violation_time)
                //anomaly
        }
        else
            row[6] = 0;
        if(s.hard_brake)
        {
            row[7] = min(row[7]+1,l.brake_violation_time);
            if(row[1] == l.brake_violation_time)
                //anomaly
        }
        else
            row[7] = 0;
        if(!s.door_lock)
        {
            row[8] = min(row[8]+1,l.door_violation_time);
            if(row[1] == l.door_violation_time)
                //anomaly
        }
        else
            row[8] = 0;
        if(s.hard_steer)
        {
            row[9] = min(row[9]+1,l.steer_violation_time);
            if(row[1] == l.steer_violation_time)
                //anomaly
        }
    }
    void update_worklist(Schema& s)
    {
        state_update(s);
        work_list[s.database_index] = s;//update the schema object being stored.
    }

    vector<Schema> get_pending_writes()
    {
        vector<Schema> v;
        for(auto it: work_list)
            v.push_back(it.second);
        return v;
    }

    void WriteRows(vector<Schema> RowsToBeWritten){
        // Find the row numbers to be modified in the database
        // This is done parallely, using the fact that the primary key of each row
        // in the argument rowsToBeModified uniquely defines a row in the actual database.
        Schema* devicerowsToBeModified;
        cudaMemcpy(deviceRowsToBeModified, hostRowsToBeModified, RowsToBeWritten.size()*sizeof(Schema), cudaMemcpyHostToDevice);
        init_bt(RowsToBeWritten.size());//linear number of threads are enough.
        // The next task, using the row numbers acquired with the above kernel, 
        // fire numberOfAttributes*numRowsToBeModified threads to modify the cells 
        // of the actual database
        changeRowsKernel<<<nb, nt>>>(
            numberOfRowsToBeModified,
            deviceRowsToBeModified, 
            StateDatabase 
        );
        cudaDeviceSynchronize();
        //std::cout << "Write Completed!!" << endl;
    }

    map<int,Schema> Select(vector<std::string> columns,std::string conditionAttribute,int conditionValue)
    {
        int* selectedValues;
        int* endIndexSelectedValues;
        int* retArr;
        int temp = 0;
        int size;
        int selectionCol = attributesToCol[selectionAttribute];
        int conditionCol = attributesToCol[conditionAttribute];
        cudaMalloc(&selectedValues, numberOfRows*sizeof(int));//row indices that were selected 
        cudaMalloc(&endIndexSelectedValues, sizeof(int));        
        cudaMemcpy(selectedValues, &temp, sizeof(int), cudaMemcpyHostToDevice);
        init_bt(numberOfRows);
        selectKernel<<<nb, nt>>>(
            StateDatabase,
            numberOfRows,
            numberOfAttributes,
            selectedValues,
            selectionCol,
            conditionCol,
            conditionValue,
            endIndexSelectedValues
        );
        cudaMemcpy(&size, endIndexSelectedValues, sizeof(int), cudaMemcpyDeviceToHost);
        retArr = new int[size];
        cudaMemcpy(retArr, selectedValues, size*sizeof(int), cudaMemcpyDeviceToHost);
        set<int> ret(retArr, retArr+size);
        return ret;
    }

    void PrintDatabase(){
        int* hostDatabase = new int[numberOfAttributes*numberOfRows];
        cudaMemcpy(hostDatabase, StateDatabase, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < numberOfRows; i ++){
            for(int j = 0; j < numberOfAttributes; j ++){
                std::cout << hostDatabase[i*numberOfAttributes+j] << " ";
            }
            std::cout << endl;
        }
        std::cout << endl;
    }
};

__global__ void DropVerticesKernel(int numberOfVertices,int numberOfDroppedVertices,int* deviceAdjacencyMatrix,int* deviceDroppedVertices)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/(numberOfVertices);
    int j = id % numberOfVertices;
    if(id < numberOfDroppedVertices*numberOfVertices)
        atomicAnd(deviceAdjacencyMatrix + numberOfVertices*deviceDroppedVertices[i] + j,INT_MAX);
}
}

__global__ void FindMinDistance(int numberOfVertices,int* deviceUsed,int* deviceDistance,int* minDistance)
{   
    // printf("init mindist = %d\n", *minDistance);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id]){
        atomicMin(minDistance, deviceDistance[id]);
    }
}

__global__ void FindArgMin(int numberOfVertices,int* deviceUsed,int* deviceDistance,int* minDistance,int* argMinVertex)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id] && *minDistance == deviceDistance[id]){
        *argMinVertex = id;
    }
}

__global__ void Relax(
    int numberOfVertices,
    int* deviceAdjacencyMatrix,
    int* deviceUsed,
    int* deviceDistance,
    int* deviceParent,
    int* minDistance,
    int* argMinVertex
)
{
    deviceUsed[*argMinVertex] = 1;
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id] && deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id] != INT_MAX){
        // printf("argMinVertex = %d\n", *argMinVertex);
        if(deviceDistance[id] > deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id]+*minDistance){
            // printf("%d %d\n", deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id], *minDistance);
            deviceDistance[id] = deviceAdjacencyMatrix[(*argMinVertex)*numberOfVertices+id]+*minDistance;
            deviceParent[id] = *argMinVertex;
        }
    }
}

// Aux function to print cuda error:
void cudaErr(){
    // Get last error:
    cudaError_t err = cudaGetLastError();
    printf("error=%d, %s, %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
}

class GPSsystem{
private:
    int numberOfVertices;
    int* hostAdjacencyMatrix;    
public:
    GPSsystem(int numVert, int* initMat){
        numberOfVertices = numVert;
        hostAdjacencyMatrix = new int[numberOfVertices*numberOfVertices];
        for(int i = 0; i < numberOfVertices*numberOfVertices; i++){
            hostAdjacencyMatrix[i] = initMat[i];
        }
    }
    vector<int> PathFinder(int source, int destination, set<int> setDroppedVertices){
        // Phase one, make a new device Adjacency Matrix using the old one and
        // the set of dropped vertices
        int numberOfDroppedVertices = setDroppedVertices.size();
        int* hostDroppedVertices = new int[numberOfDroppedVertices];
        int* deviceDroppedVertices;
        int* deviceAdjacencyMatrix;
        int idx = 0;
        for(auto vertex: setDroppedVertices)
        {
            hostDroppedVertices[idx++] = vertex;
        }
        cudaMalloc(&deviceDroppedVertices, numberOfDroppedVertices*sizeof(int));
        cudaMemcpy(deviceDroppedVertices, hostDroppedVertices, numberOfDroppedVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&deviceAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int));
        cudaMemcpy(deviceAdjacencyMatrix, hostAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
        if(numberOfDroppedVertices != 0)
        {
            init_bt(numberOfVertices*numberOfDroppedVertices);
            DropVerticesKernel<<<nb, nt>>>(numberOfVertices,numberOfDroppedVertices,deviceAdjacencyMatrix,deviceDroppedVertices);
            cudaDeviceSynchronize();
        }
        // Phase two, Implement Dijkstra:
        int  hostNumberOfUsedVertices = 0;
        int* minDistance;
        int* hostMinDistance = new int;
        *hostMinDistance = INT_MAX;
        cudaMalloc(&minDistance, sizeof(int));        
        int* argMinVertex;
        cudaMalloc(&argMinVertex, sizeof(int));
        
        int* deviceUsed;
        int* hostUsed = new int[numberOfVertices];
        for(int i = 0; i < numberOfVertices; i++){
            hostUsed[i] = 0;
        }
        int* deviceDistance;
        int* hostDistance = new int[numberOfVertices];
        for(int i = 0; i < numberOfVertices; i ++){
            hostDistance[i] = ((i == source)?0:INT_MAX);
        }
        int* deviceParent;
        int* hostParent = new int[numberOfVertices];
        cudaMalloc((void**)&deviceUsed, numberOfVertices*sizeof(int));
        cudaMalloc((void**)&deviceDistance, numberOfVertices*sizeof(int));
        cudaMalloc((void**)&deviceParent, numberOfVertices*sizeof(int));
        cudaMemcpy(deviceUsed, hostUsed, numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceDistance, hostDistance, numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
        
        while(hostNumberOfUsedVertices < numberOfVertices){
            cudaMemcpy(minDistance, hostMinDistance, sizeof(int), cudaMemcpyHostToDevice);            
            init_bt(numberOfVertices);
            FindMinDistance<<<nb, nt>>>(
                numberOfVertices,
                deviceUsed,
                deviceDistance,
                minDistance
            );
            cudaDeviceSynchronize();
            FindArgMin<<<nb, nt>>>(
                numberOfVertices,
                deviceUsed,
                deviceDistance,
                minDistance,
                argMinVertex
            );
            cudaDeviceSynchronize();
            Relax<<<nb, nt>>>(
                numberOfVertices,
                deviceAdjacencyMatrix,
                deviceUsed,
                deviceDistance,
                deviceParent,
                minDistance,
                argMinVertex
            );
            cudaDeviceSynchronize();
            hostNumberOfUsedVertices++;
        }
        cudaErr();
        cudaMemcpy(hostParent, deviceParent, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDistance, deviceDistance, numberOfVertices*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < numberOfVertices; i ++){
            std::cout << hostDistance[i] << " ";
        }
        std::cout << endl;
        for(int i = 0; i < numberOfVertices; i ++){
            std::cout << hostParent[i] << " ";
        }
        std::cout << endl;        
        std::cout << endl;
        
        if(hostDistance[destination] == INT_MAX) return vector<int>();
        vector<int> path;
        int currentVertex = destination;
        while(currentVertex != source){
            path.push_back(currentVertex);
            currentVertex = hostParent[currentVertex];
        }
        path.push_back(source);
        reverse(path.begin(), path.end());
        
        return path;
    }
};

int main(){
    int numberOfVertices;
    cin >> numberOfVertices;
    int* hostAdjacencyMatrix = new int[numberOfVertices*numberOfVertices];
    for(int i = 0; i < numberOfVertices; i ++){
        for(int j = 0; j < numberOfVertices; j ++){
            cin >> hostAdjacencyMatrix[i*numberOfVertices+j];
            if(hostAdjacencyMatrix[i*numberOfVertices+j] < 0) hostAdjacencyMatrix[i*numberOfVertices+j] = INT_MAX;
        }
    }

    GPSsystem obj(numberOfVertices, hostAdjacencyMatrix);
    int t;
    cin >> t;
    while(t--){
        int source, destination;
        cin >> source >> destination;
        set<int> setOfDroppedVertices;
        int numberOfDroppedVertices;
        cin >> numberOfDroppedVertices;
        for(int i = 0; i < numberOfDroppedVertices; i ++){
            int v;
            cin >> v;
            setOfDroppedVertices.insert(v);
        }
        auto res = obj.PathFinder(source, destination, setOfDroppedVertices);
        for(auto v: res){
            std::cout << v << " ";
        }
        std::cout << endl;
    }
}
int main(int argc, char* argv[])
{
    int num_cars = 100;
    int messages_per_second = 0;
}
// int main(){
//     int numberOfRows, numberOfAttributes, pKey;
//     cin >> numberOfRows >> numberOfAttributes >> pKey;
//     vector<std::string> attList(numberOfAttributes);
//     for(int i = 0; i < numberOfAttributes; i ++){
//         cin >> attList[i];
//     }
//     int* initTable = new int[numberOfRows*numberOfAttributes];
//     for(int i = 0; i < numberOfRows; i ++){
//         for(int j = 0; j < numberOfAttributes; j ++){
//             cin >> initTable[i*numberOfAttributes+j];
//         }
//     }
//     DatabaseManagementSystem dbms(
//         numberOfAttributes,
//         numberOfRows,
//         pKey,
//         attList, 
//         initTable
//     );

//     dbms.PrintDatabase();

//     int q;
//     cin >> q;
//     while(q--){
//         int numberOfRowsToBeModified;
//         cin >> numberOfRowsToBeModified;
//         int* hostRowsToBeModified = new int[numberOfRowsToBeModified*numberOfAttributes];
//         for(int i = 0; i < numberOfRowsToBeModified; i ++){
//             for(int j = 0; j < numberOfAttributes; j ++){
//                 cin >> hostRowsToBeModified[i*numberOfAttributes+j];
//             }
//         }
//         dbms.WriteRows(
//             numberOfRowsToBeModified,
//             hostRowsToBeModified
//         );
//         dbms.PrintDatabase();
//     }

//     int t;
//     cin >> t;
//     while(t--){
//         std::std::string selectionAttribute, conditionAttribute;
//         int conditionValue;
//         cin >> selectionAttribute >> conditionAttribute >> conditionValue;
//         auto Values = dbms.Select(
//             selectionAttribute,
//             conditionAttribute,
//             conditionValue
//         );
//         for(auto val: Values){
//             std::cout << val << " ";
//         }
//         std::cout << endl;
//     }
//     return 0;
// }