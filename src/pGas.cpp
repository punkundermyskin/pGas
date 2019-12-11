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
int taskN = 729000; // n - number of cells
int taskm; // m - number of blocks
int taskK = 1; // K - number of parameters
int taskt = 15; // t - shadow area size

double* splitIntoSubcubes (cube basicCube, int numParts, int internalCubeSide, int numPartsDimm, int nodeCubeNum) {
  double* subCubes;
  subCubes = (double *) malloc(numParts*nodeCubeNum*sizeof(double));
  int side = internalCubeSide/round(cbrt(numParts));
  int num=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*(side), 0 + j*(side), 0 + k*(side), side - 1 + i * (side) + taskt*2, side - 1 + j * (side) + taskt*2, side - 1 + k * (side) + taskt*2);
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

void updateParametetsValue(cube &cube, int nodeCubeSide, int dimSize) {
  for (int i=taskt; i<dimSize-taskt; i++)
    for (int j=taskt; j<dimSize-taskt; j++)
      for (int k=taskt; k<dimSize-taskt; k++) {
        for (int m=1; m<=taskt; m++) cube(i,j,k) += cube(i-m,j,k) + cube(i+m,j,k) + cube(i,j-m,k) + cube(i,j+m,k) + cube(i,j,k-m) + cube(i,j,k+m);
        cube(i,j,k) = cube(i,j,k)/(6*taskt);
      }
}

cube extractInternalPart(cube &nodeCube, int internalCubeSide, int id) {
  nodeCube.shed_slices(0, taskt-1);
  nodeCube.shed_slices(internalCubeSide, internalCubeSide+taskt-1);
  cube cube(internalCubeSide, internalCubeSide, internalCubeSide);
  for (uword i = 0; i < nodeCube.n_slices; i++) {
      mat nodeCubeMat = nodeCube.slice(i);
      nodeCubeMat.shed_cols(0, taskt-1);
      nodeCubeMat.shed_cols(internalCubeSide, internalCubeSide+taskt-1);
      nodeCubeMat.shed_rows(0, taskt-1);
      nodeCubeMat.shed_rows(internalCubeSide, internalCubeSide+taskt-1);
      cube.slice(i) = nodeCubeMat;
  }
  return cube;
}

void updateInternalPart(cube &basicCube, cube &internalCube, int internalCubeSide) {
  internalCube.resize(internalCubeSide + taskt*2, internalCubeSide + taskt*2, internalCubeSide);
  internalCube.insert_slices(0, taskt);
  internalCube.insert_slices(taskt + internalCubeSide, taskt);
  for (uword k = taskt; k < internalCube.n_slices - taskt; k++) {
    mat internalCubeMat = internalCube.slice(k);
    mat basicCubeMat = basicCube.slice(k);
    internalCubeMat.shed_cols(internalCubeSide, internalCubeSide + taskt - 1);
    internalCubeMat.insert_cols(0, taskt);
    internalCubeMat.shed_rows(internalCubeSide, internalCubeSide + taskt*2 - 1);
    mat topRows = basicCubeMat;
    topRows.shed_rows(taskt, internalCubeSide + taskt*2 - 1);
    mat bottomRows = basicCubeMat;
    bottomRows.shed_rows(0, internalCubeSide + taskt - 1);
    internalCubeMat.insert_rows(0, topRows);
    internalCubeMat.insert_rows(internalCubeSide + taskt, bottomRows);
    internalCubeMat.shed_cols(taskt + internalCubeSide, taskt + internalCubeSide + taskt - 1);
    internalCubeMat.shed_cols(0, taskt-1);
    mat leadColums = basicCubeMat;
    leadColums.shed_cols(taskt, internalCubeSide + taskt*2 - 1);
    mat tailColums = basicCubeMat;
    tailColums.shed_cols(0, internalCubeSide + taskt - 1);
    internalCubeMat.insert_cols(0, leadColums);
    internalCubeMat.insert_cols(taskt + internalCubeSide, tailColums);
    internalCube.slice(k) = internalCubeMat;
  }
  for (int i=0; i<taskt; i++) {
    mat basicCubeSlice = basicCube.slice(i);
    internalCube.slice(i) = basicCubeSlice;
  }
  for (int i=taskt+internalCubeSide; i<internalCubeSide+taskt*2; i++) {
    mat basicCubeSlice = basicCube.slice(i);
    internalCube.slice(i) = basicCubeSlice;
  }
  basicCube = internalCube;
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
  int numPartsDimm = round(cbrt(taskm));

  int basicCubeNum = taskN;
  int basicCubeSide = round(cbrt(basicCubeNum)); // cout points

  int internalCubeSide = basicCubeSide - 2*taskt;
  int internalCubeNum = pow(internalCubeSide, 3);

  int internalSubCubeSide = internalCubeSide/numPartsDimm;
  int internalSubCubeNum = pow(internalSubCubeSide, 3);

  int nodeCubeSide = internalSubCubeSide + 2*taskt;
  int nodeCubeNum = pow(nodeCubeSide, 3);

  cube basicCube(basicCubeSide, basicCubeSide, basicCubeSide);

  if (id==0) {
    basicCube.load("input.txt");
    // basicCube = randu<cube>(basicCubeSide, basicCubeSide, basicCubeSide);
  }
  double start = MPI_Wtime();
  for (int i=0; i<100; i++) {
    cube nodeCube(nodeCubeSide, nodeCubeSide, nodeCubeSide);
    double* arraySendCubes;
    double* arrayRecvCubes;
    vector<cube> subCubes;
    if (id == 0) {
      arraySendCubes = splitIntoSubcubes(basicCube, numParts, internalCubeSide, numPartsDimm, nodeCubeNum);
    }
    MPI_Scatter( arraySendCubes, nodeCubeNum, MPI_DOUBLE, nodeCube.begin(), nodeCubeNum, MPI_DOUBLE, 0, comm );

    if (id==0) {
      delete [] arraySendCubes;
      arraySendCubes = NULL;
      arrayRecvCubes = (double *) malloc(numParts*nodeCubeNum*sizeof(double));
    }
    updateParametetsValue(nodeCube, nodeCubeNum, nodeCubeSide);
    cube internalSubCube = extractInternalPart(nodeCube, internalSubCubeSide, id);

    MPI_Gather( internalSubCube.begin(), internalSubCubeNum, MPI_DOUBLE, arrayRecvCubes, internalSubCubeNum, MPI_DOUBLE, 0, comm );

    if (id == 0) {
      cube internalCube = combineSubcubes(arrayRecvCubes, numParts, numPartsDimm, internalSubCubeSide, internalCubeSide, internalSubCubeNum);
      updateInternalPart(basicCube, internalCube, internalCubeSide);
      delete [] arrayRecvCubes;
      arrayRecvCubes = NULL;
    }
  }
  double end = MPI_Wtime();
  if (id==0) cout << "The process 0 took " << end - start << " seconds to run." << endl;
  basicCube.save("result.txt", arma_ascii);
  MPI_Finalize();
  return 0;
}
