#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef IS_DM_CORE
#error "Define IS_DM_CORE to use this template."
#endif

#ifndef BETA_NZ
#error "Define BETA_NZ to use this template."
#endif

#include "gemm_kernel.h"

__attribute__((flatten)) // Force inline called functions
void SNBLAS_GEMM_TILING(streambuffer, FLOAT_T, IS_DM_CORE, BETA_NZ) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl) {
    
    /**
     * Problem is double buffered in L1. The buffer that is used is toggled at
     * each iteration. The DMA cores are one index step ahead so they load the
     * data in advance into the buffer that will be used.
     */

    typedef SNBLAS_GEMM_TCDM(FLOAT_T) TcdmLayout;
    typedef SnblasGemmInfo GemmInfo;
    typedef SNBLAS_GEMM_ARGS(FLOAT_T) GemmArgs;

    if (impl.bench) snrt_mcycle();

    const uint32_t M   = info.M;
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.ta ? info.M : info.K;
    const uint32_t ldb = info.tb ? info.K : info.N;
    const uint32_t ldc = info.tc ? info.M : info.N;
    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;

    const FLOAT_T* const A = args.A;
    const FLOAT_T* const B = args.B;
          FLOAT_T* const C = args.C;
    const FLOAT_T alpha    = args.alpha;
    const FLOAT_T beta     = args.beta;

    uint32_t p[3] = {0, 0, 0};
    uint32_t P[3] = {0, 0, 0};
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    // Setup layout for TCDM L1
    // For double buffering l1 is a size 2 array
    TcdmLayout* l1 = (TcdmLayout*)snrt_l1_next();

    // Initialize indices
    const uint32_t PI = 1, PJ = 1;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;

    bool storeC = false;

    GemmInfo tileInfo = {0};
    tileInfo.M   = L1_M / 2;
    tileInfo.N   = L1_N / 2;
    tileInfo.K   = L1_K;
    tileInfo.ta  = info.ta ^ TA_TILE;
    tileInfo.tb  = info.tb ^ TB_TILE;
    tileInfo.tc  = info.tc ^ TC_TILE; // TODO: implement transposed blocking
    tileInfo.lda = tileInfo.ta ? L1_M : L1_K;
    tileInfo.ldb = tileInfo.tb ? L1_K : L1_N;
    tileInfo.ldc = tileInfo.tc ? L1_M : L1_N;

    // create function ptr for dma loading
    const snrt_dma_load_2d_tile_t load_tile_A = TA_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile;
    const snrt_dma_load_2d_tile_t load_tile_B = TB_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile;
    const snrt_dma_load_2d_tile_t load_tile_C = BETA_NZ ? 
                                                TC_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile
                                                : &load_zero_tile;
    const snrt_dma_load_2d_tile_t store_tile_C = TC_TILE ? &snrt_dma_store_2d_tile_transpose : &snrt_dma_store_2d_tile;
    
    if (!IS_DM_CORE) {
        SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(tileInfo, impl);
    }
    if (impl.bench) snrt_mcycle();

    for(ib = pi; ib <  M / L1_M; ib += PI) {
        for(jb = pj; jb < N / L1_N; jb += PJ) {
            if (IS_DM_CORE) {
                dump_ib(ib);
                dump_jb(jb);
                (*load_tile_C)(l1->C, (void*) C, ib, jb, L1_M, L1_N, ldc, FP64);
            }

            for(kb = 0; kb < K / L1_K; kb++) {

                // quad split dma load, dma two steps ahead
                uint32_t isb = 0;
                uint32_t jsb = 0;
                const uint32_t ksb = 0;
                // load    b0
                // compute (0,0)
                if (IS_DM_CORE) {
                    snrt_dma_load_2d_tile_to_tile(l1->B, (void*) B,
                                                info.tb ? jb : kb, 
                                                info.tb ? kb : jb, 
                                                tileInfo.tb ? jsb : ksb,
                                                tileInfo.tb ? ksb : jsb, 
                                                info.tb ? L1_N : L1_K,  
                                                info.tb ? L1_K : L1_N, 
                                                tileInfo.ldb, ldb, FP64);
                    snrt_dma_wait_all();
                    snrt_global_barrier();
                    if (impl.bench) snrt_mcycle();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1->A + isb * tileInfo.M * tileInfo.lda;
                    tileArgs.B     = l1->B + jsb * tileInfo.N;
                    tileArgs.C     = l1->C + isb * tileInfo.M * tileInfo.ldc + 
                                            jsb * tileInfo.N;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = 1;
                        
                    // hw barrier in kernel
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }

                jsb = 1;
                // compute (0,1)
                // load    a0
                if (IS_DM_CORE) {
                    snrt_dma_load_2d_tile_to_tile(l1->A, (void*) A,
                                                info.ta ? kb : ib, 
                                                info.ta ? ib : kb, 
                                                tileInfo.tb ? ksb : isb, 
                                                tileInfo.tb ? isb : ksb, 
                                                info.ta ? L1_K : L1_M, 
                                                info.ta ? L1_M : L1_K, 
                                                tileInfo.lda, lda, FP64);
                    snrt_dma_wait_all();
                    snrt_global_barrier();
                    if (impl.bench) snrt_mcycle();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1->A + isb * tileInfo.M * tileInfo.lda;
                    tileArgs.B     = l1->B + jsb * tileInfo.N;
                    tileArgs.C     = l1->C + isb * tileInfo.M * tileInfo.ldc + 
                                            jsb * tileInfo.N;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = 1;
                        
                    // hw barrier in kernel
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }


                isb = 1;
                // compute (1,1)
                // load    b1
                if (IS_DM_CORE) {
                    snrt_dma_load_2d_tile_to_tile(l1->B, (void*) B,
                                                info.tb ? jb : kb, 
                                                info.tb ? kb : jb, 
                                                tileInfo.tb ? jsb : ksb,
                                                tileInfo.tb ? ksb : jsb, 
                                                info.tb ? L1_N : L1_K,  
                                                info.tb ? L1_K : L1_N, 
                                                tileInfo.ldb, ldb, FP64);
                    snrt_dma_wait_all();
                    snrt_global_barrier();
                    if (impl.bench) snrt_mcycle();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1->A + isb * tileInfo.M * tileInfo.lda;
                    tileArgs.B     = l1->B + jsb * tileInfo.N;
                    tileArgs.C     = l1->C + isb * tileInfo.M * tileInfo.ldc + 
                                            jsb * tileInfo.N;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = 1;
                        
                    // hw barrier in kernel
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }

                jsb = 0;
                // compute (1,0)
                // load    a1
                if (IS_DM_CORE) {
                    snrt_dma_load_2d_tile_to_tile(l1->A, (void*) A,
                                                info.ta ? kb : ib, 
                                                info.ta ? ib : kb, 
                                                tileInfo.tb ? ksb : isb, 
                                                tileInfo.tb ? isb : ksb, 
                                                info.ta ? L1_K : L1_M, 
                                                info.ta ? L1_M : L1_K, 
                                                tileInfo.lda, lda, FP64);
                    snrt_dma_wait_all();
                    snrt_global_barrier();
                    if (impl.bench) snrt_mcycle();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1->A + isb * tileInfo.M * tileInfo.lda;
                    tileArgs.B     = l1->B + jsb * tileInfo.N;
                    tileArgs.C     = l1->C + isb * tileInfo.M * tileInfo.ldc + 
                                            jsb * tileInfo.N;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = 1;
                        
                    // hw barrier in kernel
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }

                isb = 0;
            }

            if (IS_DM_CORE) {
                (*store_tile_C)(C, l1->C, ib, jb, L1_M, L1_N, ldc, FP64);
            }
        }
    }

    if (IS_DM_CORE) {
        snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    } else {
        snrt_fpu_fence();
        SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(tileInfo, impl);
        snrt_cluster_hw_barrier();
    }
    snrt_fpu_fence();
    snrt_global_barrier();
}
