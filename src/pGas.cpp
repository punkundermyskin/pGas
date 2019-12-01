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
// int taskN = 64; // n - number of cells
int taskN = 216;
int taskm; // m - number of blocks
int taskK = 1; // K - number of parameters
int taskt = 1; // t - shadow area size

double* splitIntoSubcubes (cube basicCube, int numParts, int internalCubeSide, int numPartsDimm) {
  double* subCubes;
  subCubes = (double *) malloc(numParts*pow((internalCubeSide+taskt*2), 3)*sizeof(double));
  int side = internalCubeSide/cbrt(numParts);
  int num=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*side, 0 + j*side, 0 + k*side, side - 1 + i * side + taskt*2, side - 1  + j * side + taskt*2, side - 1 + k * side + taskt*2);
        memcpy(subCubes + subCube.size() * num , subCube.begin(), subCube.size() * sizeof(double));
        num++;
  }
  return subCubes;
}

cube combineSubcubes(double* &arrayCubes, int numParts, int numPartsDimm, int subCubeSide, int basicCubeSide, int subCubeNum) {

  vector<cube> subCubes;

  for (int i=0; i<numParts; i++) {
    cube cube(arrayCubes + i*subCubeNum, subCubeSide, subCubeSide, subCubeSide);
    subCubes.push_back(cube);
  }
  delete [] arrayCubes;
  arrayCubes = NULL;
  cube newCube;
  for (int i=0; i < numPartsDimm; i++) {
    cube cubeY;
    for (int j=0; j < numPartsDimm; j++) {
      cube cubeZ;
      for (int k=0; k < numPartsDimm; k++) {
        cubeZ = join_slices( cubeZ, subCubes[i * numPartsDimm * numPartsDimm + j * numPartsDimm + k]);
      }
      cubeY.resize(subCubeSide, subCubeSide*(j+1), basicCubeSide);
      for (uword k = 0; k < cubeY.n_slices; k++) {
          mat cubeYMat = cubeY.slice(k);
          cubeYMat.shed_cols(subCubeSide*j, subCubeSide*(j+1)-1);
          mat cubeZMat = cubeZ.slice(k);
          cubeYMat.insert_cols(cubeYMat.n_cols, cubeZMat);
          cubeY.slice(k) = cubeYMat;
      }
    }
    newCube.resize(subCubeSide*(i+1), basicCubeSide, basicCubeSide);
    for (uword j = 0; j < newCube.n_slices; j++)
    {
        mat newCubeMat = newCube.slice(j);
        mat cubeYMat = cubeY.slice(j);
        newCubeMat.shed_rows(subCubeSide*i, subCubeSide*(i+1)-1);
        newCubeMat.insert_rows(newCubeMat.n_rows, cubeYMat);
        newCube.slice(j) = newCubeMat;
    }
  }
  return newCube;
}

void updateParametetsValue(cube &cube, int nodeCubeSide) {
  int dimSize = cbrt(cube.size());
  for (int i=0+taskt; i<dimSize-taskt; i++)
    for (int j=0+taskt; j<dimSize-taskt; j++)
      for (int k=0+taskt; k<dimSize-taskt; k++) {
        for (int m=1; m<=taskt; m++) cube(i,j,k) += cube(i-m,j,k) + cube(i+m,j,k) + cube(i,j-m,k) + cube(i,j+m,k) + cube(i,j,k-m) + cube(i,j,k+m);
        cube(i,j,k) = cube(i,j,k)/(6*taskt);
        // cube(i,j,k) = 0;
      }
}

cube extractInternalPart(cube &nodeCube, int internalCubeSide) {

  nodeCube.shed_slices(internalCubeSide+1, internalCubeSide+1+taskt-1);
  nodeCube.shed_slices(0, taskt-1);
  cube cube(internalCubeSide, internalCubeSide, internalCubeSide);
  for (uword i = 0; i < nodeCube.n_slices; i++) {
      mat nodeCubeMat = nodeCube.slice(i);
      nodeCubeMat.shed_cols(internalCubeSide+1, internalCubeSide+1+taskt-1);
      nodeCubeMat.shed_cols(0, taskt-1);
      nodeCubeMat.shed_rows(internalCubeSide+1, internalCubeSide+1+taskt-1);
      nodeCubeMat.shed_rows(0, taskt-1);
      cube.slice(i) = nodeCubeMat;
  }
  return cube;
}

void updateInternalPart(cube &basicCube, cube &internalCube, int internalCubeSide) {

  // TO-DO !

  // internalCube.resize(internalCubeSide + taskt*2, internalCubeSide + taskt*2, internalCubeSide + taskt*2);
  // for (uword k = taskt; k < internalCube.n_slices - taskt; k++) {
  //     mat internalCubeMat = internalCube.slice(k);
  //     internalCubeMat.shed_rows(internalCubeSide, internalCubeSide + taskt*2 - 1);
  //     internalCubeMat.shed_cols(internalCubeSide, internalCubeSide + taskt*2 - 1);
  //     mat basicCubeMat = cube.slice(k);
  //     internalCubeMat.insert_cols(0, taskt);
  //     internalCubeMat.insert_rows(0, taskt);
  //     internalCubeMat.insert_cols(0, taskt);
  //     internalCubeMat.insert_rows(0, taskt);
  //     cube.slice(k) = cubeMat;
  // }
  // for (int i=0; i<taskt; i++) {
  //   mat basicCubeSlice = basicCube.slice(i);
  //   internalCube.insert_slices(0+i, basicCubeSlice);
  // }
  // for (int i=internalCubeSide; i<internalCubeSide+taskt; i++) {
  //   mat basicCubeSlice = basicCube.slice(i);
  //   internalCube.insert_slices(internalCubeSide, basicCubeSlice);
  // }

}

int main(int argc, char *argv[]) {
  int id, worldSize;
  MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  taskm = worldSize;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  srand(time(NULL) + id);

  // Initialize the random generator
  arma_rng::set_seed_random();

  int numParts = taskm;
  int numPartsDimm = cbrt(taskm);

  int basicCubeNum = taskN;
  int basicCubeSide = cbrt(basicCubeNum);

  int internalCubeSide = basicCubeSide - 2*taskt;
  int internalCubeNum = pow(internalCubeSide, 3);

  int internalSubCubeSide = (basicCubeSide - 2*taskt)/numPartsDimm;
  int internalSubCubeNum = pow(internalSubCubeSide, 3);

  int nodeCubeSide = (internalCubeSide/numPartsDimm) + 2*taskt;
  int nodeCubeNum = pow(nodeCubeSide, 3);

  cube nodeCube(nodeCubeSide, nodeCubeSide, nodeCubeSide);
  cube basicCube(basicCubeSide, basicCubeSide, basicCubeSide);
  double* arraySendCubes;
  double* arrayRecvCubes;
  vector<cube> subCubes;

  if (id == 0) {
    // basicCube = randu<cube>(basicCubeSide, basicCubeSide, basicCubeSide);
    int num = 0;
    for (int i=0; i<basicCubeNum; i++) {
      basicCube[i] = num;
      num++;
    }
    arraySendCubes = splitIntoSubcubes(basicCube, numParts, internalCubeSide, numPartsDimm);
  }
  MPI_Scatter( arraySendCubes, nodeCubeNum, MPI_DOUBLE, nodeCube.begin(), nodeCubeNum, MPI_DOUBLE, 0, comm );
  if (id==0) {
    delete [] arraySendCubes;
    arraySendCubes = NULL;
    arrayRecvCubes = (double *) malloc(pow(internalCubeSide, 3)*sizeof(double));
  }
  updateParametetsValue(nodeCube, nodeCubeNum);

  // for (int i=0; i<100000; i++) {
  //   updateParametetsValue(nodeCube, nodeCubeNum);
  // }

  cube internalSubCube = extractInternalPart(nodeCube, internalSubCubeSide);

  MPI_Gather( internalSubCube.begin(), internalSubCubeNum, MPI_DOUBLE, arrayRecvCubes, internalSubCubeNum, MPI_DOUBLE, 0, comm );

  if (id == 0) {
    // basicCube = randu<cube>(basicCubeSide, basicCubeSide, basicCubeSide);
    cube internalCube = combineSubcubes(arrayRecvCubes, numParts, numPartsDimm, internalSubCubeSide, internalCubeSide, internalSubCubeNum);
    updateInternalPart(basicCube, internalCube, internalCubeSide);
  }

  MPI_Finalize();
  return 0;
}
