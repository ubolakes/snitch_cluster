#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef IS_DM_CORE
#error "Define IS_DM_CORE to use this template."
#endif

#include "gemm_kernel.h"

void SNBLAS_GEMM_TILING(baseline, FLOAT_T, IS_DM_CORE) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl) {
    
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
    const uint32_t lda = info.lda;
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc;
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
    TcdmLayout* l1 = snrt_l1_next();

    // Which buffer is the valid data in for computation
    bool l1Id_A = true;
    bool l1Id_B = true;
    bool l1Id_C = false;

    // Initialize indices
    const uint32_t PI = 2, PJ = 2;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev = -1, jb_prev = -1, kb_prev = -1;

    bool storeC = false;

    GemmInfo tileInfo = {0};
    tileInfo.M   = L1_M;
    tileInfo.N   = L1_N;
    tileInfo.K   = L1_K;
    tileInfo.lda = L1_LDA;
    tileInfo.ldb = L1_LDB;
    tileInfo.ldc = L1_LDC;
    tileInfo.ta  = false;
    tileInfo.tb  = false;

    // TODO: place memory barrier before sync
    if (impl.bench) snrt_mcycle();

    if (!IS_DM_CORE) {
        SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(tileInfo, impl);
        snrt_cluster_hw_barrier();  // DMA core is one index ahead
    }

    for(ib = pi; ib <  M / L1_M; ib += PI) {
        for(jb = pj; jb < N / L1_N; jb += PJ) {
            FLOAT_T* const l1_C = l1[l1Id_C].C;

            if (IS_DM_CORE) {
                dump_ib(ib);
                dump_jb(jb);
                snrt_dma_load_2d_tile(l1_C, (void*) C, ib, jb, L1_M, L1_N, ldc, FP64);
                if (ib_prev >= 0 && jb_prev >= 0) storeC = true;
            }

            for(kb = 0; kb < K / L1_K; kb++) {
                // Switch buffers when the indices have changed
                l1Id_A = !l1Id_A;
                l1Id_B = !l1Id_B;

                FLOAT_T* const l1_A = l1[l1Id_A].A;
                FLOAT_T* const l1_B = l1[l1Id_B].B;

                if (IS_DM_CORE) {
                    dump_kb(kb);
                    snrt_dma_load_2d_tile(l1_A, (void*) A, ib, kb, L1_M, L1_K, lda,
                                        FP64);
                    snrt_dma_load_2d_tile(l1_B, (void*) B, kb, jb, L1_K, L1_N, ldb,
                                        FP64);
                    snrt_dma_wait_all();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1_A;
                    tileArgs.B     = l1_B;
                    tileArgs.C     = l1_C;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = beta;
                    
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }
                
                // if (impl.bench) snrt_mcycle();
                snrt_cluster_hw_barrier();
                if (impl.bench) snrt_mcycle();

                if (IS_DM_CORE) {
                    if (storeC) {
                        storeC = false;
                        snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev,
                                               jb_prev, L1_M, L1_N, ldc, FP64);
                    }
                }
                kb_prev = kb;
            }

            l1Id_C = !l1Id_C;
            jb_prev = jb;
            ib_prev = ib;
        }
    }

    if (IS_DM_CORE) {
        snrt_cluster_hw_barrier();  // DMA core is one index ahead

        // store final tile
        // if (ib_prev >= 0 && jb_prev >= 0) {
            snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N,
                                ldc, FP64);
            snrt_dma_wait_all();
        // }
    } else {
        SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(tileInfo, impl);
    }

    if (impl.bench) snrt_mcycle();
}
