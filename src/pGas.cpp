#include <iostream>
#include <armadillo>
#include <cmath>
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
// int taskN = 216;
// int taskm = 27;
int taskK = 1; // K - number of parameters
int taskt = 1; // t - shadow area size

void extendCubeToShadowArea(cube &cube) {
  int extendedSize = cube.n_slices + taskt;
  cube.insert_slices(0, taskt);
  cube.insert_slices(cube.n_slices, taskt);
  cube.resize(extendedSize + taskt, extendedSize + taskt, extendedSize + taskt);
  for (uword k = 0; k < cube.n_slices; k++) {
      mat cubeMat = cube.slice(k);
      cubeMat.shed_rows(extendedSize, extendedSize + taskt - 1);
      cubeMat.shed_cols(extendedSize, extendedSize + taskt - 1);
      cubeMat.insert_cols(0, taskt);
      cubeMat.insert_rows(0, taskt);
      cube.slice(k) = cubeMat;
  }
}

double* splitIntoSubcubes (cube basicCube, int numParts, int subCubeSize) {

  double* subCubes;
  int numPartsDimm = cbrt(numParts);
  int side = subCubeSize;
  subCubes = (double *) malloc(numParts*pow((side+taskt*2), 3)*sizeof(double));
  int num=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*subCubeSize, 0 + j*subCubeSize, 0 + k*subCubeSize, subCubeSize - 1  + i * subCubeSize + taskt*2, subCubeSize - 1  + j * subCubeSize + taskt*2, subCubeSize - 1  + k * subCubeSize + taskt*2);
        memcpy(subCubes + subCube.size() * num , subCube.begin(), subCube.size() * sizeof(double));
        num++;
  }
  return subCubes;
}

cube combineSubcubes(double* &arraySendCubes, int numParts, int subCubeSize, int cubeSize, int nodeCellsNum) {
  int numPartsDimm = cbrt(numParts);
  vector<cube> subCubes;
  for (int i=0; i<numParts; i++) {
    cube cube(arraySendCubes + i*nodeCellsNum, subCubeSize, subCubeSize, subCubeSize);
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
  for (int i=0+taskt; i<dimSize-taskt; i++)
    for (int j=0+taskt; j<dimSize-taskt; j++)
      for (int k=0+taskt; k<dimSize-taskt; k++) {
        for (int m=1; m<=taskt; m++) cube(i,j,k) += cube(i-m,j,k) + cube(i+m,j,k) + cube(i,j-m,k) + cube(i,j+m,k) + cube(i,j,k-m) + cube(i,j,k+m);
        cube(i,j,k) = cube(i,j,k)/(6*taskt);
      }
}

cube cropCubeBack(cube &nodeCube, int subCubeSize) {
  nodeCube.shed_slices(subCubeSize+1, subCubeSize+1+taskt-1);
  nodeCube.shed_slices(0, taskt-1);
  cube cube(subCubeSize, subCubeSize, subCubeSize);
  for (uword i = 0; i < nodeCube.n_slices; i++) {
      mat nodeCubeMat = nodeCube.slice(i);
      nodeCubeMat.shed_cols(subCubeSize+1, subCubeSize+1+taskt-1);
      nodeCubeMat.shed_cols(0, taskt-1);
      nodeCubeMat.shed_rows(subCubeSize+1, subCubeSize+1+taskt-1);
      nodeCubeMat.shed_rows(0, taskt-1);
      cube.slice(i) = nodeCubeMat;
  }
  return cube;
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

  int cellsNum = taskN;
  int numParts = taskm;
  int cubeSize = cbrt(cellsNum);
  int numPartsDimm = cbrt(numParts);
  int subCubeSize = cubeSize / numPartsDimm;

  cube extendedNodeCube(subCubeSize+taskt*2, subCubeSize+taskt*2, subCubeSize+taskt*2);
  cube nodeCube(subCubeSize, subCubeSize, subCubeSize);
  cube basicCube(cubeSize, cubeSize, cubeSize);
  double* arraySendCubes;
  double* arrayRecvCubes;
  vector<cube> subCubes;

  if (id == 0) {
    basicCube = randu<cube>(cubeSize, cubeSize, cubeSize);
    extendCubeToShadowArea(basicCube);
    arraySendCubes = splitIntoSubcubes(basicCube, numParts, subCubeSize);
  }
  MPI_Scatter( arraySendCubes, extendedNodeCube.size(), MPI_DOUBLE, extendedNodeCube.begin(), extendedNodeCube.size(), MPI_DOUBLE, 0, comm );
  if (id==0) {
    arrayRecvCubes = (double *) malloc(pow(cubeSize, 3)*sizeof(double));
  }

  updateParametetsValue(extendedNodeCube, cubeSize);

  for (int i=0; i<100000; i++) {
    updateParametetsValue(extendedNodeCube, cubeSize);
  }

  nodeCube = cropCubeBack(extendedNodeCube, subCubeSize);

  MPI_Gather( nodeCube.begin(), nodeCube.size(), MPI_DOUBLE, arrayRecvCubes, nodeCube.size(), MPI_DOUBLE, 0, comm );

  if (id == 0) {
    basicCube = randu<cube>(cubeSize, cubeSize, cubeSize);
    basicCube = combineSubcubes(arrayRecvCubes, numParts, subCubeSize, cubeSize, nodeCube.size());
    cout << basicCube << endl;
  }

  MPI_Finalize();
  return 0;
}
