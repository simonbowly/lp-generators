// Wrapper class around COIN CLP callable library to solve instances and
// write MPS files.

#ifndef LP_HPP
#define LP_HPP

#include <string>
using std::string;
#include <cstddef>

#include "ClpModel.hpp"
#include "ClpSimplex.hpp"
#include "OsiClpSolverInterface.hpp"


class LP {

 public:
    LP();
    ~LP();

    // Construction methods (overwrite existing data)
    void constructDenseCanonical(int nv, int nc, double* A, double* b, double* c);

    // Conversion to COIN models (return pointers requiring cleanup)
    ClpModel* getClpModel();
    OsiClpSolverInterface* getOsiClpModel();
    CoinPackedMatrix* getCoinPackedMatrix();

    // Model writers
    void writeMps(string fileName);
    void writeMpsIP(string fileName);
    void writeMpsMIP(string fileName, string vtypes);

    // Property accessors
    int getNumVariables() { return numVariables; }
    int getNumConstraints() { return numConstraints; }
    int getNumLHSElements() { return numLHSElements; }

    // Read data into arrays
    void getLhsMatrixDense(double* buffer);
    void getRhsVector(double* buffer);
    void getObjVector(double* buffer);

    // Solution
    void solve();
    int getSolutionStatus();
    void getSolutionPrimals(double* buffer);
    void getSolutionSlacks(double* buffer);
    void getSolutionDuals(double* buffer);
    void getSolutionReducedCosts(double* buffer);
    void getSolutionBasis(double* buffer);

 private:
    int numVariables;
    int numConstraints;
    int numLHSElements;

    // Requiring cleanup
    double* lhsMatrixDense;
    double* rhsVector;
    double* objVector;
    ClpSimplex* simplexModel;

};

#endif
