# Coarse-grained design: one thread per signature.
#   make VARIANT=2 DESIGN=bitsliced        (VARIANT 1/2/3/5, DESIGN bitsliced|nibble)
CUDA_ROOT_DIR ?= /usr/local/cuda
NVCC    ?= nvcc
CC      ?= gcc
VARIANT ?= 2
DESIGN  ?= bitsliced
ARCH    ?= native

ifeq ($(DESIGN),bitsliced)
DESIGN_FLAGS = -DMAYO_GF16_MASK -DMAYO_BITSLICE_SPS -DMAYO_BITSLICE_HYBRID
endif

MAYO_C  = ../mayo-c
CC_FLAGS   = -O2 -DMAYO_$(VARIANT)
NVCC_FLAGS = -arch=$(ARCH) -O3 -DMAYO_$(VARIANT) -rdc=true $(DESIGN_FLAGS)
LIBS = $(MAYO_C)/build/src/libmayo_$(VARIANT)_nistapi.a $(MAYO_C)/build/src/libmayo.a $(MAYO_C)/build/src/libmayo_common_sys.a

OBJS = bin/main.o bin/cuda_kernel.o bin/inputs.o bin/fips202.o bin/host_helpers.o
EXE  = mayo-coarse

$(EXE): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) $(LIBS) -o $@ -L$(CUDA_ROOT_DIR)/lib64 -lcudart -lstdc++

bin:
	mkdir -p bin

bin/main.o: main.c | bin
	$(CC) $(CC_FLAGS) -c $< -o $@

bin/%.o: src/%.cu | bin
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf bin $(EXE)
