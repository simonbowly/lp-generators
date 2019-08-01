// Very basic tests to check for successful compile.

#ifndef TESTS_CPP
#define TESTS_CPP

#include "lp.hpp"
#include <gtest/gtest.h>


TEST(LPTest, ConstructDense) {

    double A[] = {
        1,0,2,0,1,
        0,1,0,1,0,
        1,-1,0,1,0,
        0,0,-1,1,0,
        };
    double b[] = {1, 2, 3, 4};
    double c[] = {1, 2, 3, 4, 5};

    // Construct model
    LP lp;
    lp.constructDenseCanonical(5, 4, A, b, c);

    // Check accessors
    ASSERT_EQ(5, lp.getNumVariables());
    ASSERT_EQ(4, lp.getNumConstraints());
    ASSERT_EQ(10, lp.getNumLHSElements());

    // Check copiers
    double* buffer = new double[20];
    lp.getLhsMatrixDense(buffer);
    for (int i = 0; i < 20; i++) {
        ASSERT_EQ(A[i], buffer[i]);
    }
    delete[] buffer;

    buffer = new double[4];
    lp.getRhsVector(buffer);
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ(b[i], buffer[i]);
    }
    delete[] buffer;

    buffer = new double[5];
    lp.getObjVector(buffer);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(c[i], buffer[i]);
    }
    delete[] buffer;

}


TEST(LPTest, Model) {

    double A[] = {
        1,0,2,0,1,
        0,1,0,1,0,
        1,-1,0,1,0,
        0,0,-1,1,0,
        };
    double b[] = {1, 2, 3, 4};
    double c[] = {1, 2, 3, 4, 5};

    // Construct model
    LP lp;
    lp.constructDenseCanonical(5, 4, A, b, c);

    // COIN matrix
    CoinPackedMatrix* matrix = lp.getCoinPackedMatrix();
    ASSERT_EQ(5, matrix->getNumCols());
    ASSERT_EQ(4, matrix->getNumRows());
    ASSERT_EQ(10, matrix->getNumElements());
    delete matrix;

    // COIN clp model
    ClpModel* model = lp.getClpModel();
    ASSERT_EQ(5, model->getNumCols());
    ASSERT_EQ(4, model->getNumRows());
    ASSERT_EQ(10, model->getNumElements());
    delete model;

}


TEST(LPTest, Write) {

    double A[] = {1, 3, 3, 1};
    double b[] = {4, 4};
    double c[] = {1, 1};

    // Construct model
    LP lp;
    lp.constructDenseCanonical(2, 2, A, b, c);

    // Write to file
    lp.writeMps("cpp_test_lp.mps.gz");
    lp.writeMpsIP("cpp_test_ip.mps.gz");
    lp.writeMpsMIP("cpp_test_mip.mps.gz", "IC");

}


TEST(LPTest, Solve) {

    double A[] = {1, 3, 3, 1};
    double b[] = {4, 4};
    double c[] = {1, 1};

    // Construct model
    LP lp;
    lp.constructDenseCanonical(2, 2, A, b, c);

    // Solve and copy results
    lp.solve();

    double* primals = new double[2];
    double* slacks = new double[2];
    double* costs = new double[2];
    double* duals = new double[2];
    double* basis = new double[4];

    lp.getSolutionPrimals(primals);
    lp.getSolutionSlacks(slacks);
    lp.getSolutionReducedCosts(costs);
    lp.getSolutionDuals(duals);
    lp.getSolutionBasis(basis);

    ASSERT_EQ(1, primals[0]);   ASSERT_EQ(1, primals[1]);
    ASSERT_EQ(0, slacks[0]);    ASSERT_EQ(0, slacks[1]);
    ASSERT_EQ(0, costs[0]);     ASSERT_EQ(0, costs[1]);
    ASSERT_EQ(.25, duals[0]);   ASSERT_EQ(.25, duals[1]);

    ASSERT_EQ(1, basis[0]);
    ASSERT_EQ(1, basis[1]);
    ASSERT_EQ(0, basis[2]);
    ASSERT_EQ(0, basis[3]);

    delete[] primals;
    delete[] slacks;
    delete[] costs;
    delete[] duals;
    delete[] basis;

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif
