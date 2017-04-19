
#ifndef LP_CPP
#define LP_CPP

#include "lp.hpp"


LP::LP() {
    numVariables = -1;
    numConstraints = -1;
    numLHSElements = -1;
    lhsMatrixDense = NULL;
    rhsVector = NULL;
    objVector = NULL;
    simplexModel = NULL;
}


LP::~LP() {
    delete [] lhsMatrixDense;
    delete [] rhsVector;
    delete [] objVector;
    delete simplexModel;
}


void LP::constructDenseCanonical(int nv, int nc, double* A, double* b, double* c) {
    // Construction method by copying dense arrays.

    numVariables = nv;
    numConstraints = nc;
    numLHSElements = 0;

    lhsMatrixDense = new double[nv * nc];
    rhsVector = new double[nc];
    objVector = new double[nv];

    for (int col = 0; col < nv; col++) {
        objVector[col] = c[col];
    }

    int index = 0;
    for (int row = 0; row < nc; row++) {
        rhsVector[row] = b[row];
        for (int col = 0; col < nv; col++) {
            index = row * nv + col;
            if (A[index] != 0) {
                numLHSElements++;
            }
            lhsMatrixDense[index] = A[index];
        }
    }
}


CoinPackedMatrix* LP::getCoinPackedMatrix() {
    // Allocate and return pointer to a COIN matrix.

    int* rowIndices = new int[numLHSElements];
    int* colIndices = new int[numLHSElements];
    double* elements = new double[numLHSElements];

    int index;
    int elem = 0;
    for (int row = 0; row < numConstraints; row++) {
        for (int col = 0; col < numVariables; col++) {
            index = row * numVariables + col;
            if (lhsMatrixDense[index] != 0) {
                rowIndices[elem] = row;
                colIndices[elem] = col;
                elements[elem] = lhsMatrixDense[index];
                elem++;
            }
        }
    }

    CoinPackedMatrix* matrix = new CoinPackedMatrix(
        true, rowIndices, colIndices, elements, numLHSElements);

    delete[] rowIndices;
    delete[] colIndices;
    delete[] elements;

    return matrix;
}


ClpModel* LP::getClpModel() {
    // Allocate and return pointer to a COIN LP model
    //
    // Problem is stored in canonical form
    //  max  cTx
    //  s.t. Ax <= b
    //       x >= 0
    //
    // Clp model is created as a minimisation model
    //  min  -cTx
    //  s.t. Ax <= b
    //       x >= 0

    CoinPackedMatrix* matrix = getCoinPackedMatrix();

    double* lowerColumn = new double[numVariables];
    double* upperColumn = new double[numVariables];
    double* objective = new double[numVariables];
    double* lowerRow = new double[numConstraints];
    double* upperRow = new double[numConstraints];

    for (int col = 0; col < numVariables; col++) {
        lowerColumn[col] = 0;
        upperColumn[col] = COIN_DBL_MAX;
        objective[col] = objVector[col] * -1;
    }

    for (int row = 0; row < numConstraints; row++) {
        lowerRow[row] = -COIN_DBL_MAX;
        upperRow[row] = rhsVector[row];
    }

    ClpModel* model = new ClpModel();
    model->loadProblem(
        *matrix, lowerColumn, upperColumn, objective, lowerRow, upperRow);

    delete[] lowerColumn;
    delete[] upperColumn;
    delete[] objective;
    delete[] lowerRow;
    delete[] upperRow;
    delete matrix;

    return model;
}


OsiClpSolverInterface* LP::getOsiClpModel() {
    // Allocate and return pointer to a COIN LP model
    //
    // Problem is stored in canonical form
    //  max  cTx
    //  s.t. Ax <= b
    //       x >= 0
    //
    // Clp model is created as a minimisation model
    //  min  -cTx
    //  s.t. Ax <= b
    //       x >= 0

    CoinPackedMatrix* matrix = getCoinPackedMatrix();

    double* lowerColumn = new double[numVariables];
    double* upperColumn = new double[numVariables];
    double* objective = new double[numVariables];
    double* lowerRow = new double[numConstraints];
    double* upperRow = new double[numConstraints];

    for (int col = 0; col < numVariables; col++) {
        lowerColumn[col] = 0;
        upperColumn[col] = COIN_DBL_MAX;
        objective[col] = objVector[col] * -1;
    }

    for (int row = 0; row < numConstraints; row++) {
        lowerRow[row] = -COIN_DBL_MAX;
        upperRow[row] = rhsVector[row];
    }

    OsiClpSolverInterface* model = new OsiClpSolverInterface();
    model->loadProblem(
        *matrix, lowerColumn, upperColumn, objective, lowerRow, upperRow);

    delete[] lowerColumn;
    delete[] upperColumn;
    delete[] objective;
    delete[] lowerRow;
    delete[] upperRow;
    delete matrix;

    return model;
}


void LP::writeMps(string fileName) {
    // Generate a COIN model for writing to MPS file.
    OsiClpSolverInterface* model;
    model = getOsiClpModel();
    model->writeMps(fileName.c_str(), "", 0.0);
    delete model;
}


void LP::writeMpsIP(string fileName) {
    // Generate a COIN model for writing to MPS file.
    OsiClpSolverInterface* model;
    model = getOsiClpModel();
    for (int i = 0; i < numVariables; i++) {
        model->setInteger(i);
    }
    model->writeMps(fileName.c_str(), "", 0.0);
    delete model;
}


void LP::getLhsMatrixDense(double* buffer) {
    // Copy stored dense constraints to an array.
    // Input array must have getNumVariables() * getNumConstraints() elements.
    for (int i=0; i<numConstraints*numVariables; i++) {
        buffer[i] = lhsMatrixDense[i];
    }
}


void LP::getRhsVector(double* buffer) {
    // Copy stored constraint upper bounds to an array.
    // Input array must have getNumConstraints() elements.
    for (int i=0; i<numConstraints; i++) {
        buffer[i] = rhsVector[i];
    }
}


void LP::getObjVector(double* buffer) {
    // Copy stored objective function coefficients to an array.
    // Input array must have getNumVariables() elements.
    for (int i=0; i<numVariables; i++) {
        buffer[i] = objVector[i];
    }
}


void LP::solve() {
    // Build and solve model, store it for retrieving solution values.
    delete simplexModel;
    ClpModel* model;
    model = getClpModel();
    simplexModel = new ClpSimplex(*model);
    simplexModel->setLogLevel(0);
    simplexModel->dual();
    delete model;
}


int LP::getSolutionStatus() {
    return simplexModel->status();
}


void LP::getSolutionPrimals(double* buffer) {
    // Copy stored primal solution for solved model to an array.
    // Input array must have getNumVariables() elements.
    const double* primalValues = simplexModel->getColSolution();
    for (int i=0; i<numVariables; i++) {
        buffer[i] = primalValues[i];
    }
}


void LP::getSolutionSlacks(double* buffer) {
    // Copy stored constraint slacks for solved model to an array.
    // Input array must have getNumConstraints() elements.
    const double* rowValues = simplexModel->getRowActivity();
    const double* rhsValues = simplexModel->getRowUpper();
    for (int i=0; i<numConstraints; i++) {
        buffer[i] = rhsValues[i] - rowValues[i];
    }
}


void LP::getSolutionDuals(double* buffer) {
    const double* dualValues = simplexModel->getRowPrice();
    for (int i=0; i<numConstraints; i++) {
        buffer[i] = dualValues[i] * -1;
    }
}


void LP::getSolutionReducedCosts(double* buffer) {
    const double* reducedCosts = simplexModel->getReducedCost();
    for (int i=0; i<numVariables; i++) {
        buffer[i] = reducedCosts[i];
    }
}


void LP::getSolutionBasis(double* buffer) {

    for (int i = 0; i < numVariables; i++) {
        buffer[i] = double(simplexModel->getColumnStatus(i) == ClpSimplex::Status::basic);
    }

    for (int i = 0; i < numConstraints; i++) {
        buffer[i + numVariables] = double(simplexModel->getRowStatus(i) == ClpSimplex::Status::basic);
    }

}

#endif
