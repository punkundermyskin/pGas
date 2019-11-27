#include <iostream>
#include <armadillo>

#include "mpi.h"
#include <stdio.h>

#include <vector>

#include <random>
#include <ctime>

using namespace std;
using namespace arma;

double* splitIntoSubcubes (cube basicCube, int numParts, int subCubeSize) {
  double* subCubes;
  int numPartsDimm = cbrt(numParts);
  subCubes = (double *) malloc(numParts*subCubeSize*subCubeSize*subCubeSize*sizeof(double));
  int num=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*subCubeSize, 0 + j*subCubeSize, 0 + k*subCubeSize, subCubeSize - 1  + i * subCubeSize, subCubeSize - 1  + j * subCubeSize, subCubeSize - 1  + k * subCubeSize );
        memcpy(subCubes + subCube.size() * cout , subCube.begin(), subCube.size() * sizeof(double));
        num++;
  }
  return subCubes;
}

void combineSubcubes(cube &basicCube, double* arraySendCubes, int numParts, int nodeCellsNum) {
  for (int i=0; i< numParts; i++) memcpy(basicCube.begin() + nodeCellsNum * i , arraySendCubes + nodeCellsNum * i, nodeCellsNum * sizeof(double));
}

int main(int argc, char *argv[]) {
  int id, worldSize;
  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  srand(time(NULL) + id);

  // Initialize the random generator
  arma_rng::set_seed_random();

  int cellsNum = 64; // 8000000
  int numParts = 8; // 64
  int nodeCellsNum = cellsNum/numParts;
  int cubeSize = cbrt(cellsNum); // 6
  int numPartsDimm = cbrt(numParts); // 2
  int subCubeSize = cubeSize / numPartsDimm; // 6/3

  cube nodeCube(subCubeSize, subCubeSize, subCubeSize);
  cube basicCube(cubeSize, cubeSize, cubeSize);
  double* arraySendCubes;
  vector<cube> subCubes;

  if (id == 0) {
    int num = 0;
    for (value:basicCube) { // set cells value from 0 to 63
      value = num;
      num++;
    }
    arraySendCubes = splitIntoSubcubes(basicCube, numParts, subCubeSize);
  }
  MPI_Scatter( arraySendCubes, nodeCellsNum, MPI_DOUBLE, nodeCube.begin(), nodeCellsNum, MPI_DOUBLE, 0, comm );
  cout << nodeCube << endl;
  MPI_Gather( nodeCube.begin(), nodeCellsNum, MPI_DOUBLE, arraySendCubes, nodeCellsNum, MPI_DOUBLE, 0, comm );

  if (id == 0) {
    combineSubcubes(basicCube, arraySendCubes, numParts, nodeCellsNum);
    delete [] arraySendCubes;
    arraySendCubes = NULL;
    // cout << basicCube << endl;
  }

  MPI_Finalize();
  return 0;
}
