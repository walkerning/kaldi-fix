

all:

include ../kaldi.mk

#CXXFLAGS += -std=gnu++11
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-pdf-prior.o nnet-randomizer.o nnet-nnet-fix.o

LIBNAME = kaldi-nnet

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
         ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../base/kaldi-base.a

include ../makefiles/default_rules.mk

