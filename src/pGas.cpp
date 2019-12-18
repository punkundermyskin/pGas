#include <iostream>
#include <armadillo>
#include <cmath>
#include "mpi.h"
#include <stdio.h>

#include <vector>

#include <random>
#include <ctime>

#include <unistd.h>
#include <fstream>
#include <bits/stdc++.h>
#include <math.h>

using namespace std;
using namespace arma;

int taskP; // p - processors count
int taskN; // n - number of cells
int taskm; // m - number of blocks
int taskK; // K - number of parameters
int taskt; // t - shadow area size
int taskI;
bool testMode = true;
vector<string> inputFilesPaths;
vector<string> outputFilesPaths;

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

bool is_digits(const std::string &str) {
    return str.find_first_not_of("0123456789. -e") == std::string::npos;
}

void findInputError(string path) {
  std::ifstream file(path);
  if (file.is_open() == false) cerr << "File " << path << " can't be opened!" << endl;
  else {
    std::string str;
    int num = 0;
    while (std::getline(file, str)) {
      if (num==0 && str != "ARMA_CUB_TXT_FN008") cerr << path << ":" << num << " - wrong line !" << endl;
      if (is_digits(str) == false) cerr << path << ":" << num << " - wrong line !" << endl;
      num++;
    }
  }
}

vector<string> splitFilePaths(string str) {
  vector<string> filePaths;
  string path = "";
  for (auto x : str) {
      if (x == ' ') {
          filePaths.push_back(path);
          path = "";
      } else path = path + x;
  }
  filePaths.push_back(path);
  return filePaths;
}

int parserRun(int argc, char *argv[]) {
  int opt;
  if ((argc < 10) || (argc > 13)) {
      cerr << "Wrong arguments number!" << endl;
      return 1;
  }
  opterr = 0;

  while ( (opt = getopt(argc, argv, "n:k:t:i:f:o:e")) != -1 ) {
    switch ( opt ) {
      case 'n':
        taskN = atoi(optarg);
        if (taskN == 0) {
          std::cerr << "error in parameter: -n" << endl;
          return 1;
        }
        break;
      case 'k':
        taskK = atoi(optarg);
        if (taskK == 0) {
          std::cerr << "error in parameter: -k"  << endl;
          return 1;
        }
        break;
      case 't':
        taskt = atoi(optarg);
        if (taskt == 0) {
          std::cerr << "error in parameter: -t" << endl;
          return 1;
        }
        break;
      case 'i':
        taskI = atoi(optarg);
        if (taskI == 0) {
          std::cerr << "error in parameter: -i"  << endl;
          return 1;
        }
        break;
      case 'e':
        testMode = false;
        break;
      case 'f':
        if (testMode == true) {
          inputFilesPaths = splitFilePaths(optarg);
        } else break;
      case 'o':
        if (testMode == true) {
          outputFilesPaths = splitFilePaths(optarg);
        } else break;
      case '?':
        if (optopt != 0) cerr << "Unknown option: '" << char(optopt) << "'!" << endl;
        break;
      }
  }
  return 0;
  // To-Do check values!
}

int main(int argc, char *argv[]) {
  int id, worldSize;
  MPI_Init(&argc,&argv);
  if (parserRun(argc, argv) == 1) MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  srand(time(NULL) + id);

  // Initialize the random generator
  arma_rng::set_seed_random();

  int numParts = worldSize;
  int numPartsDimm = round(cbrt(worldSize));

  int basicCubeNum = taskN/taskK;
  int basicCubeSide = round(cbrt(basicCubeNum)); // cout points

  int internalCubeSide = basicCubeSide - 2*taskt;
  int internalCubeNum = pow(internalCubeSide, 3);

  int internalSubCubeSide = internalCubeSide/numPartsDimm;
  int internalSubCubeNum = pow(internalSubCubeSide, 3);

  int nodeCubeSide = internalSubCubeSide + 2*taskt;
  int nodeCubeNum = pow(nodeCubeSide, 3);

  checkInputParameters();
  vector<cube> basicCubes;
  cube basicCube(basicCubeSide, basicCubeSide, basicCubeSide);

  // load input data
  if (id==0) {
    if (testMode==false) {
      for (int i=0; i<taskK; i++) basicCubes.push_back(randu<cube>(basicCubeSide, basicCubeSide, basicCubeSide));
    } else {
      for (int i=0; i<taskK; i++) {
        string path = inputFilesPaths[i];
        basicCube.load(path);
        if (basicCube.is_empty()) {
          findInputError(path);
          MPI_Abort(MPI_COMM_WORLD, 1);
        } else if (basicCube.size() != basicCubeNum) cerr << path << ": Wrong cube size" << endl;
        basicCubes.push_back(basicCube);
      }
    }
  }

  double start = MPI_Wtime();

  for (int i=0; i<taskK; i++) {
    if (id==0) basicCube = basicCubes[i];
    for (int i=0; i<taskI; i++) {
      if (worldSize==1) updateParametetsValue(basicCube, basicCubeNum, basicCubeSide);
      else {
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
    }
    if (id==0) basicCubes[i] = basicCube;
  }

  double end = MPI_Wtime();
  if (id==0) {
    if (testMode == true) {
      for (int i=0; i<taskK; i++) {
        string path = outputFilesPaths[i];
        basicCubes[i].save(path, arma_ascii);
      }
    } else
    cout << "The process 0 took " << end - start << " seconds to run." << endl;
  }

  MPI_Finalize();
  return 0;
}
