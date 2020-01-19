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
int taskI; // number of iterations
bool testMode = true; // need some output?

vector<string> inputFilesPaths;
vector<string> outputFilesPaths;

double* splitIntoSubcubes (cube basicCube, int numParts, int internalCubeSide, int numPartsDimm, int nodeCubeNum) {
  double* subCubes;
  // used dynamic array instead of STL things
  // cause MPI works only with sequential memory
  subCubes = (double *) malloc(numParts*nodeCubeNum*sizeof(double));
  int side = internalCubeSide/round(cbrt(numParts));
  int num=0;
  int cubesSize=0;
  for (int i=0; i < numPartsDimm; i++)
    for (int j=0; j < numPartsDimm; j++)
      for (int k=0; k < numPartsDimm; k++) {
        cube subCube = basicCube.subcube( 0 + i*(side), 0 + j*(side), 0 + k*(side), side - 1 + i * (side) + taskt*2, side - 1 + j * (side) + taskt*2, side - 1 + k * (side) + taskt*2);
        memcpy(subCubes + subCube.size() * num , subCube.begin(), subCube.size() * sizeof(double));
        num++;
        cubesSize += subCube.size();
  }
  return subCubes;
}

cube combineSubcubes(double* &arrayCubes, int numParts, int numPartsDimm, int subCubeSide, int basicCubeSide, int subCubeNum) {

  // so u just give this func the array
  // with ur cubes and it combine they into one
  // if ur arg is ok

  vector<cube> subCubes;

  for (int i=0; i<numParts; i++) {
    cube cube(arrayCubes + i*subCubeNum, subCubeSide, subCubeSide, subCubeSide);
    subCubes.push_back(cube);
  }
  // when all cubes loaded - cleaning memory
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
  // using the new cube
  // cause arma wan't resize existing cube colm and rows
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

  // the outer "2t" slices (rows, colums) of the original cube
  // will be restored by this func

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

bool is_perfect_cube(int n) {
    int root(round(cbrt(n)));
    return n == root * root * root;
}

int checkInputParameters(int basicCubeSide, int internalCubeSide, int internalSubCubeSide, int nodeCubeSide) {
  if (is_perfect_cube(taskN) == false) {
    cerr << "Invalid parameter N!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (is_perfect_cube(taskP) == false) {
    cerr << "Invalid parameter -np !" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if ((taskt <= 0) || (basicCubeSide < 1) || (internalCubeSide < 1) || (internalSubCubeSide < 1) || (nodeCubeSide < 1) ) {
    cerr << "Wrong grid size. Please check the input parameters: -n, -t  !" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (inputFilesPaths.size() != outputFilesPaths.size()) {
    cerr << "Check numbers input/output files!" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

bool is_digits(const std::string &str) {
    return str.find_first_not_of("0123456789. -e") == std::string::npos;
}

void findInputError(string path) {
  std::ifstream file(path);
  if (file.is_open() == false) cerr << "Input file " << path << " not found or couldn't be opened: " << strerror(errno) << endl;
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
    cerr << endl;
      cerr << "Wrong number of parameters!" << endl;
      cerr << endl;
      cerr << "Please use mpirun -np <np> ./pGas -n <N> -k <K> -t <T> -i <I>" << endl;
      cerr << "{-e | -f '<input_file> ... <input_file>' -o '<input_file> ... <input_file>'}" << endl;
      cerr << endl;
      return 1;
  }
  opterr = 0;

  while ( (opt = getopt(argc, argv, "n:k:t:i:f:o:e")) != -1 ) {
    switch ( opt ) {
      case 'n':
        taskN = atoi(optarg);
        if (taskN == 0) {
          std::cerr << "Invalid parameter -n" << endl;
          return 1;
        }
        break;
      case 'k':
        taskK = atoi(optarg);
        if (taskK == 0) {
          std::cerr << "Invalid parameter -k"  << endl;
          return 1;
        }
        break;
      case 't':
        taskt = atoi(optarg);
        if (taskt == 0) {
          std::cerr << "Invalid parameter -t" << endl;
          return 1;
        }
        break;
      case 'i':
        taskI = atoi(optarg);
        if (taskI == 0) {
          std::cerr << "Invalid parameter -i"  << endl;
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
  // check numbers
}

int main(int argc, char *argv[]) {
  int id; // process number
  int worldSize; // number of process
  MPI_Init(&argc,&argv);

  // Get input arguments with some check
  if (parserRun(argc, argv) == 1) MPI_Abort(MPI_COMM_WORLD, 1);

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  taskP = worldSize;

  // Initialize the random generator
  srand(time(NULL) + id);
  arma_rng::set_seed_random();

  int numParts = worldSize; // number of parts into which the cube (grid) will be divided
  int numPartsDimm = round(cbrt(worldSize)); // number of parts in 1D

  int basicCubeNum = taskN;
  int basicCubeSide = round(cbrt(basicCubeNum));

  int internalCubeSide = basicCubeSide - 2*taskt;
  int internalCubeNum = pow(internalCubeSide, 3);

  int internalSubCubeSide = internalCubeSide/numPartsDimm;
  int internalSubCubeNum = pow(internalSubCubeSide, 3);

  int nodeCubeSide = internalSubCubeSide + 2*taskt; //  size of the cube to be sent to the node
  int nodeCubeNum = pow(nodeCubeSide, 3);

  // check cube size
  if ((is_perfect_cube(internalCubeNum) == false ) || (is_perfect_cube(internalSubCubeNum) == false ) || (is_perfect_cube(nodeCubeNum) == false )) {
    cerr << "Wrong grid size! Please check input parameters: -n, -t !" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // one more check
  checkInputParameters(basicCubeSide, internalCubeSide, internalSubCubeSide, nodeCubeSide);

  cube basicCube(basicCubeSide, basicCubeSide, basicCubeSide);

  double total = 0; // timer

  for (int i=0; i<taskK; i++) {
    if (id==0) {
      // if u don't have any input/output, then it's just about
      // checking time so cube will be init by random float numbers
      if (testMode == false) basicCube = randu<cube>(basicCubeSide, basicCubeSide, basicCubeSide);
      else {
        // else we load your cube from file
        string path = inputFilesPaths[i];
        basicCube.load(path);
        // and do some check
        if (basicCube.is_empty()) {
          findInputError(path);
          MPI_Abort(MPI_COMM_WORLD, 1);
        } else if (basicCube.size() != basicCubeNum) {
          cerr << path << ": Wrong grid size" << endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }
    double start = MPI_Wtime();
    for (int i=0; i<taskI; i++) {
      if (worldSize==1) updateParametetsValue(basicCube, basicCubeNum, basicCubeSide); // serial algorithm
      else {
        cube nodeCube(nodeCubeSide, nodeCubeSide, nodeCubeSide);
        double* arraySendCubes;
        double* arrayRecvCubes;
        // On the root process split the input (basic) cube into
        // subcubes to be able to send it by MPI thing
        if (id == 0) {
          arraySendCubes = splitIntoSubcubes(basicCube, numParts, internalCubeSide, numPartsDimm, nodeCubeNum);
        }
        // Send data to nodes
        MPI_Scatter( arraySendCubes, nodeCubeNum, MPI_DOUBLE, nodeCube.begin(), nodeCubeNum, MPI_DOUBLE, 0, comm );
        if (id==0) {
          // cleaning memory and prepare array for recv data
          delete [] arraySendCubes;
          arraySendCubes = NULL;
          arrayRecvCubes = (double *) malloc(numParts*nodeCubeNum*sizeof(double));
        }
        // main thing
        updateParametetsValue(nodeCube, nodeCubeNum, nodeCubeSide);
        // trimming the cube to the original internal part
        cube internalSubCube = extractInternalPart(nodeCube, internalSubCubeSide, id);
        // recv results on root process from nodes
        MPI_Gather( internalSubCube.begin(), internalSubCubeNum, MPI_DOUBLE, arrayRecvCubes, internalSubCubeNum, MPI_DOUBLE, 0, comm );

        if (id == 0) {
          // restore the internal part
          cube internalCube = combineSubcubes(arrayRecvCubes, numParts, numPartsDimm, internalSubCubeSide, internalCubeSide, internalSubCubeNum);
          // restore the original cube with updated internal part
          updateInternalPart(basicCube, internalCube, internalCubeSide);
          // cleaning memory
          delete [] arrayRecvCubes;
          arrayRecvCubes = NULL;
        }
      }
    }
    if (id==0) {
      if (testMode == true) {
        for (int i=0; i<taskK; i++) {
          string path = outputFilesPaths[i];
          basicCube.save(path, arma_ascii);

          ifstream ifs (path);

          if (ifs.is_open()) {
            cout << "the file: " << path << " was successfully written" << endl;
          }
          else {
            cerr << "File " << path << " coudn't be used for write data: " << strerror(errno) << endl;;
          }
        }
      }
      double stop = MPI_Wtime();
      total += stop - start;
    }
  }

  if (id==0) {
    if (testMode == false) cout << "Time: " << total << " seconds." << endl;
  }

  MPI_Finalize();
  return 0;
}
