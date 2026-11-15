# Fine-grained design: one thread per output vector, warp-ballot elimination.
#   make VARIANT=2 BATCH=4096              (VARIANT 1/2/3/5; BATCH = signatures per launch)
CUDA_ROOT_DIR ?= /usr/local/cuda
NVCC    ?= nvcc
CC      ?= gcc
VARIANT ?= 2
BATCH   ?= 4096
ARCH    ?= native

MAYO_C  = ../mayo-c
NVCC_FLAGS = -arch=$(ARCH) -O3 -Xcompiler -O3 -DMAYO_VARIANT=$(VARIANT) -DBATCH=$(BATCH)
LIBS = $(MAYO_C)/build/src/libmayo_$(VARIANT)_nistapi.a $(MAYO_C)/build/src/libmayo.a $(MAYO_C)/build/src/libmayo_common_sys.a

OBJS = bin/main.o bin/mayo.o bin/inputs.o bin/fips202.o bin/aes_c.o bin/gf16_bitslice_selftest.o
EXE  = mayo-fine

$(EXE): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) $(LIBS) -o $@ -L$(CUDA_ROOT_DIR)/lib64 -lcudart -lstdc++

bin:
	mkdir -p bin

bin/main.o: main.cu | bin
	$(NVCC) $(NVCC_FLAGS) -Iinclude -c $< -o $@

bin/%.o: src/%.cu | bin
	$(NVCC) $(NVCC_FLAGS) -Iinclude -c $< -o $@

bin/%.o: src/%.c | bin
	$(CC) -O2 -Iinclude -c $< -o $@

clean:
	rm -rf bin $(EXE)
