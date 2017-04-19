# This makefile and tests are not meant to do more then check the basics.
# Logic tests, performance tests, etc are performed via python.

CXX = g++
CXXFLAGS = -std=c++11 -Wunused-variable
CXXLINKFLAGS =

INCL = -I$(GTEST_HOME)/include -I$(COINOR_HOME)/include/coin
LIBS = -L$(GTEST_HOME)/build -lgtest -lpthread

LIBS += -L$(COINOR_HOME)/lib -lCbcSolver -lCbc -lCgl -lOsiClp -lClpSolver -lClp -lOsi -lCoinUtils -lz -lm


all: test

clean:
	@echo "-> cleaning objects and binary"
	@rm -f bin/tests obj/*.o

bin/tests: obj/tests.o obj/lp.o
	@echo "-> linking test program"
	@mkdir -p bin
	@$(CXX) $(CXXFLAGS) $(INCL) $(CXXLINKFLAGS) obj/lp.o obj/tests.o -o bin/tests $(LIBS)

obj/%.o: %.cpp
	@echo "-> compiling $@"
	@mkdir -p obj
	@$(CXX) $(CXXFLAGS) $(INCL) $(MACROS) $< -o $@ -c

test: bin/tests
	@echo "-> running tests"
	@bin/tests

obj/lp.o: lp.hpp
