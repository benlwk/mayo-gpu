#ifdef __cplusplus
extern "C" {
#endif

#define BATCH 131072        // total signatures.
#define BLOCK 256           // threads per block
#define GRID  BATCH / BLOCK // num blocks

// List wrapper function callable by .cpp file.
void kernel(double *A, double *B, double *C, int arraySize);
void run_mayo_sign();
void run_mayo_verify();

#ifdef __cplusplus
}
#endif
