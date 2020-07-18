#include<bits/stdc++.h>
#include<cuda.h>
#include<thrust/device_vector.h>

using namespace std;


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

__global__ void setRowMapKernel(
    int numberOfRows,
    int numberOfRowsToBeModified,
    int numberOfAttributes,
    int primaryKey,
    int* deviceMapToRows,
    int* deviceRowsToBeModified,
    int* deviceDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int j = id/numberOfRows;
    int i = id%numberOfRows;
    if(j < numberOfRowsToBeModified && deviceDatabase[numberOfAttributes*i+primaryKey] == deviceRowsToBeModified[numberOfAttributes*j+primaryKey]){
        deviceMapToRows[j] = i;
    }
}

__global__ void changeRowsKernel(
    int numberOfRowsToBeModified,
    int numberOfAttributes,
    int* deviceMapToRows,
    int* deviceRowsToBeModified, 
    int* deviceDatabase
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int j = id/numberOfRowsToBeModified;
    int k = id%numberOfRowsToBeModified;
    if(j < numberOfAttributes){
        int i = deviceMapToRows[k];
        deviceDatabase[i*numberOfAttributes+j] = deviceRowsToBeModified[k*numberOfAttributes+j];
    }
}

__global__ void selectKernel(
    int* deviceDatabase,
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
    if(id < numberOfAttributes && deviceDatabase[id*numberOfAttributes+conditionCol] == conditionValue){
        int i = atomicAdd(endIndexSelectedValues, 1);
        selectedValues[i] = deviceDatabase[id*numberOfAttributes+selectionCol];
    }
}

class DatabaseManagementSystem{
private:
    int* deviceDatabase;
    const int numberOfAttributes;
    const int numberOfRows;
    const int primaryKey;
    map<string, int> attributesToCol;
public:

    DatabaseManagementSystem(
        int numAtt,
        int numRows, 
        int pKey,
        vector<string> attList, 
        int* initTable
    ):
        numberOfAttributes(numAtt),
        numberOfRows(numRows),
        primaryKey(pKey)
    {
        for(int i = 0; i < numberOfAttributes; i ++) attributesToCol[attList[i]] = i;
        cudaMalloc(&deviceDatabase, numberOfAttributes*numberOfRows*sizeof(int));
        cudaMemcpy(deviceDatabase, initTable, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyHostToDevice);        
    }

    
    void WriteRows(int numberOfRowsToBeModified, int* hostRowsToBeModified){
        // Find the row numbers to be modified in the database
        // This is done parallely, using the fact that the primary key of each row
        // in the argument rowsToBeModified uniquely defines a row in the actual database.
        int* deviceRowsToBeModified;
        int* deviceMapToRows;
        cudaMalloc(&deviceRowsToBeModified, numberOfRowsToBeModified*numberOfAttributes*sizeof(int));
        cudaMalloc(&deviceMapToRows, numberOfRowsToBeModified*sizeof(int));
        cudaMemcpy(deviceRowsToBeModified, hostRowsToBeModified, numberOfRowsToBeModified*numberOfAttributes*sizeof(int), cudaMemcpyHostToDevice);
        
        cout << "Starting map compute" << endl;
        init_bt(numberOfRows*numberOfRowsToBeModified);
        setRowMapKernel<<<nb, nt>>>(
            numberOfRows,
            numberOfRowsToBeModified,
            numberOfAttributes,
            primaryKey,
            deviceMapToRows,
            deviceRowsToBeModified,
            deviceDatabase
        );
        cudaDeviceSynchronize();
        cout << "Map Computed!!" << endl;
        // The next task, using the row numbers acquired with the above kernel, 
        // fire numberOfAttributes*numRowsToBeModified threads to modify the cells 
        // of the actual database
        init_bt(numberOfRowsToBeModified*numberOfAttributes);
        changeRowsKernel<<<nb, nt>>>(
            numberOfRowsToBeModified,
            numberOfAttributes,
            deviceMapToRows,
            deviceRowsToBeModified, 
            deviceDatabase 
        );
        cudaDeviceSynchronize();
        cout << "Write Completed!!" << endl;
    }

    set<int> Select(
        string selectionAttribute,
        string conditionAttribute,
        int conditionValue
    ){
        int* selectedValues;
        int* endIndexSelectedValues;
        int* retArr;
        int temp = 0;
        int size;
        int selectionCol = attributesToCol[selectionAttribute];
        int conditionCol = attributesToCol[conditionAttribute];
        cudaMalloc(&selectedValues, numberOfRows*sizeof(int));
        cudaMalloc(&endIndexSelectedValues, sizeof(int));        
        cudaMemcpy(selectedValues, &temp, sizeof(int), cudaMemcpyHostToDevice);
        init_bt(numberOfRows);
        selectKernel<<<nb, nt>>>(
            deviceDatabase,
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
        cudaMemcpy(hostDatabase, deviceDatabase, numberOfAttributes*numberOfRows*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < numberOfRows; i ++){
            for(int j = 0; j < numberOfAttributes; j ++){
                cout << hostDatabase[i*numberOfAttributes+j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};

__global__ void DropVerticesKernel(
    int numberOfVertices,
    int numberOfDroppedVertices,
    int* deviceAdjacencyMatrix,
    int* deviceDroppedVertices
)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/(numberOfVertices*numberOfVertices);
    if(i < numberOfDroppedVertices){
        int u = id%(numberOfVertices*numberOfVertices)/numberOfVertices;
        int v = id%(numberOfVertices*numberOfVertices)%numberOfVertices;
        if(deviceDroppedVertices[i] == u || deviceDroppedVertices[i] == v){
            deviceAdjacencyMatrix[u*numberOfVertices+v] = INT_MAX;
        }
    }
}

__global__ void FindMinDistance(
    int numberOfVertices,
    int* deviceUsed,
    int* deviceDistance,
    int* minDistance
)
{   
    // printf("init mindist = %d\n", *minDistance);
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < numberOfVertices && !deviceUsed[id]){
        atomicMin(minDistance, deviceDistance[id]);
    }
}

__global__ void FindArgMin(
    int numberOfVertices,
    int* deviceUsed,
    int* deviceDistance,
    int* minDistance,
    int* argMinVertex
)
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
        for(auto vertex: setDroppedVertices){
            hostDroppedVertices[idx++] = vertex;
        }
        cudaMalloc(&deviceDroppedVertices, numberOfDroppedVertices*sizeof(int));
        cudaMemcpy(deviceDroppedVertices, hostDroppedVertices, numberOfDroppedVertices*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&deviceAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int));
        cudaMemcpy(deviceAdjacencyMatrix, hostAdjacencyMatrix, numberOfVertices*numberOfVertices*sizeof(int), cudaMemcpyHostToDevice);
        if(numberOfDroppedVertices != 0){
            init_bt(numberOfVertices*numberOfVertices*numberOfDroppedVertices);
            DropVerticesKernel<<<nb, nt>>>(
                numberOfVertices,
                numberOfDroppedVertices,
                deviceAdjacencyMatrix,
                deviceDroppedVertices
            );
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
            cout << hostDistance[i] << " ";
        }
        cout << endl;
        for(int i = 0; i < numberOfVertices; i ++){
            cout << hostParent[i] << " ";
        }
        cout << endl;        
        cout << endl;
        
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
            cout << v << " ";
        }
        cout << endl;
    }
}

// int main(){
//     int numberOfRows, numberOfAttributes, pKey;
//     cin >> numberOfRows >> numberOfAttributes >> pKey;
//     vector<string> attList(numberOfAttributes);
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
//         string selectionAttribute, conditionAttribute;
//         int conditionValue;
//         cin >> selectionAttribute >> conditionAttribute >> conditionValue;
//         auto Values = dbms.Select(
//             selectionAttribute,
//             conditionAttribute,
//             conditionValue
//         );
//         for(auto val: Values){
//             cout << val << " ";
//         }
//         cout << endl;
//     }
//     return 0;
// }