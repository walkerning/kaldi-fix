
all:

include ../kaldi.mk

#CXXFLAGS += -std=gnu++11

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = fix-strategy-test
OBJFILES = fix-strategy.o
LIBNAME = kaldi-fix

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../util/kaldi-util.a \
         ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../thread/kaldi-thread.a \
         ../base/kaldi-base.a

include ../makefiles/default_rules.mk
