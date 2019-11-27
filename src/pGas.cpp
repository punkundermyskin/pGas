#include <iostream>
#include <armadillo>

#include "mpi.h"
#include <stdio.h>

#include <vector>

#include <random>
#include <ctime>

using namespace std;
using namespace arma;

int taskP; // p - processors count
int taskN = 64; // n - number of cells
int taskm = 8; // m - number of blocks
int taskK = 1; // K - number of parameters
int taskt = 1; // t - shadow area size

double* splitIntoSubcubes (cube basicCube, int numParts, int subCubeSize) {
  double* subCubes;
  int numPartsDimm = cbrt(numParts);
  subCubes = (double *) malloc(numParts*subCubeSize*subCubeSize*subCubeSize*sizeof(double));
  int num=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*subCubeSize, 0 + j*subCubeSize, 0 + k*subCubeSize, subCubeSize - 1  + i * subCubeSize, subCubeSize - 1  + j * subCubeSize, subCubeSize - 1  + k * subCubeSize );
        memcpy(subCubes + subCube.size() * num , subCube.begin(), subCube.size() * sizeof(double));
        num++;
  }
  return subCubes;
}

cube combineSubcubes(double* &arraySendCubes, int numParts, int subCubeSize, int cubeSize, int nodeCellsNum) {
  int numPartsDimm = cbrt(numParts);
  vector<cube> subCubes;
  for (int i=0; i<numParts; i++) {
    cube cube(subCubeSize, subCubeSize, subCubeSize);
    memcpy(cube.begin() + i*nodeCellsNum, arraySendCubes + i*nodeCellsNum, nodeCellsNum * sizeof(double));
    subCubes.push_back(cube);
  }
  delete [] arraySendCubes;
  arraySendCubes = NULL;
  cube newCube;
  for (int i=0; i < numPartsDimm; i++) {
    cube cubeY;
    for (int j=0; j < numPartsDimm; j++) {
      cube cubeZ;
      for (int k=0; k < numPartsDimm; k++) {
        cubeZ = join_slices( cubeZ, subCubes[i * numPartsDimm * numPartsDimm + j * numPartsDimm + k]);
      }
      cubeY.resize(subCubeSize, subCubeSize*(j+1), cubeSize);
      for (uword k = 0; k < cubeY.n_slices; k++) {
          mat cubeYMat = cubeY.slice(k);
          cubeYMat.shed_cols(subCubeSize*j, subCubeSize*(j+1)-1);
          mat cubeZMat = cubeZ.slice(k);
          cubeYMat.insert_cols(cubeYMat.n_cols, cubeZMat);
          cubeY.slice(k) = cubeYMat;
      }
    }
    newCube.resize(subCubeSize*(i+1), cubeSize, cubeSize);
    for (uword j = 0; j < newCube.n_slices; j++)
    {
        mat newCubeMat = newCube.slice(j);
        mat cubeYMat = cubeY.slice(j);
        newCubeMat.shed_rows(subCubeSize*i, subCubeSize*(i+1)-1);
        newCubeMat.insert_rows(newCubeMat.n_rows, cubeYMat);
        newCube.slice(j) = newCubeMat;
    }
  }
  return newCube;
}

void updateParametetsValue(cube &cube, int subCubeSize) {
  int dimSize = cbrt(cube.size());
  for (int i=0+taskt; i<dimSize; i++)
    for (int j=0+taskt; j<dimSize; j++)
      for (int k=0+taskt; k<dimSize; k++) {
        // for (int m=1; m<=taskt; m++) cube(i,j,k) += cube(i-m,j,k) + cube(i+m,j,k) + cube(i,j-m,k) + cube(i,j+m,k) + cube(i,j,k-m) + cube(i,j,k+m);
        // cube(i,j,k) = taskt*cube(i,j,k)*1/6;
      }
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

  int cellsNum = 64;
  int numParts = 8;
  int nodeCellsNum = cellsNum/numParts;
  int cubeSize = cbrt(cellsNum);
  int numPartsDimm = cbrt(numParts);
  int subCubeSize = cubeSize / numPartsDimm;

  cube nodeCube(subCubeSize, subCubeSize, subCubeSize);
  cube basicCube(cubeSize, cubeSize, cubeSize);
  double* arraySendCubes;
  vector<cube> subCubes;

  if (id == 0) {
    int num = 0;
    for (value:basicCube) {
      value = num;
      num++;
    }
    arraySendCubes = splitIntoSubcubes(basicCube, numParts, subCubeSize);
  }
  MPI_Scatter( arraySendCubes, nodeCellsNum, MPI_DOUBLE, nodeCube.begin(), nodeCellsNum, MPI_DOUBLE, 0, comm );

  updateParametetsValue(nodeCube, subCubeSize);

  MPI_Gather( nodeCube.begin(), nodeCellsNum, MPI_DOUBLE, arraySendCubes, nodeCellsNum, MPI_DOUBLE, 0, comm );

  if (id == 0) {
    basicCube = combineSubcubes(arraySendCubes, numParts, subCubeSize, cubeSize, nodeCellsNum);

    cout << basicCube << endl;
  }

  MPI_Finalize();
  return 0;
}
