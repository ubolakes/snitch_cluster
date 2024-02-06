#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef IS_DM_CORE
#error "Define IS_DM_CORE to use this template."
#endif

#include "gemm_kernel.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void SNBLAS_GEMM_TILING(1dpipe, FLOAT_T, IS_DM_CORE) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const bool bench) {

    typedef SNBLAS_GEMM_TCDM(FLOAT_T) TcdmLayout;
    typedef SnblasGemmInfo GemmInfo;
    typedef SNBLAS_GEMM_ARGS(FLOAT_T) GemmArgs;

    if (bench) snrt_mcycle();

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
    const uint32_t PI = P[1], PJ = 1;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev = -1, jb_prev = -1, kb_prev = -1;
    bool ib_dir = false, jb_dir = false, kb_dir = false;

    bool storeC = false;

    // -- Compute C2C sources for 2D pipeline
    const uint32_t pk = 0;  // pipeline step
    const int PK      = 1;  // pipeline depth

    // Determine C2C source cluster index for each matrix, < 0 is from DRAM
    TcdmLayout* l1Ptr[P[1]];
    #if IS_DM_CORE
    for (int i = 0; i < P[1]; ++i)
        l1Ptr[i] = (TcdmLayout*)((uint32_t)l1 + cluster_offset * (i - p[1]));
    
    const int p_srcA = pj - 1;
    const int p_srcB = pi - 1;
    TcdmLayout* const c2cL1_A = p_srcA == 0 ? NULL : l1Ptr[p_srcA];
    TcdmLayout* const c2cL1_B = p_srcB == 0 ? NULL : l1Ptr[p_srcB];
    #endif

    GemmInfo tileInfo = {0};
    tileInfo.M   = L1_M;
    tileInfo.N   = L1_N;
    tileInfo.K   = L1_K;
    tileInfo.lda = L1_LDA;
    tileInfo.ldb = L1_LDB;
    tileInfo.ldc = L1_LDC;
    tileInfo.ta  = false;
    tileInfo.tb  = false;

    if (bench) snrt_mcycle();

    if (!IS_DM_CORE) {
        SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(tileInfo);
        // DMA core is one index ahead
        
        asm volatile ("" ::: "memory");
        snrt_global_barrier();
        if (bench) snrt_mcycle();
    }

    FOR_EACH(ib, pi, M / L1_M, PI, ib_dir, ib_prev) {
        FOR_EACH(jb, pj, N / L1_N, PJ, jb_dir, jb_prev) {
            FLOAT_T* const l1_C = l1[l1Id_C].C;

            if (IS_DM_CORE) {
                dump_ib(ib);
                dump_jb(jb);
                snrt_dma_load_2d_tile(l1_C, (void*) C, ib, jb, L1_M, L1_N, ldc, FP64);
                if (ib_prev >= 0 /* && jb_prev >= 0 */) storeC = true;
            }

            FOR_EACH(kb, 0, K / L1_K, 1, kb_dir, kb_prev) {

                // Only load if the indices have changed, otherwise data is already loaded
                const bool loadA = ib != ib_prev || kb != kb_prev;
                const bool loadB = kb != kb_prev || jb != jb_prev;

                // Switch buffers when the indices have changed
                if (loadA) l1Id_A = !l1Id_A;
                if (loadB) l1Id_B = !l1Id_B;

                FLOAT_T* const l1_A = l1[l1Id_A].A;
                FLOAT_T* const l1_B = l1[l1Id_B].B;

                if (IS_DM_CORE) {
                    dump_kb(kb);
                    if (loadA) {
                        snrt_dma_load_2d_tile(l1_A, (void*) A, ib, kb, L1_M, L1_K, lda, FP64);
                        // FLOAT_T* const c2c_A = c2cL1_A[l1Id_A].A;
                        // snrt_dma_start_1d(l1_A, c2c_A, L1_M * L1_K * FP64);
                    }
                    if (loadB) {
                        snrt_dma_load_2d_tile(l1_B, (void*) B, kb, jb, L1_K, L1_N, ldb, FP64);
                        if (p[1] == 0) {
                            // immediately broadcast to other clusters
                            for (int pt = 1; pt < P[1]; ++pt) {
                                FLOAT_T* const c2c_B = l1Ptr[pt][l1Id_B].B;
                                snrt_dma_start_1d(l1_B, c2c_B, L1_K * L1_N * FP64);
                            }
                        }
                    }

                    snrt_dma_wait_all();
                    if (bench) snrt_mcycle();
                } else {
                    // solve block already in l1, parallelize inside each cluster

                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1_A;
                    tileArgs.B     = l1_B;
                    tileArgs.C     = l1_C;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = beta;
                    
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, bench);
                }
                
                snrt_global_barrier();
                if (bench) snrt_mcycle();

                if (IS_DM_CORE) {
                    if (storeC) {
                        storeC = false;
                        snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev,
                                               jb_prev, L1_M, L1_N, ldc, FP64);
                    }
                }
                kb_prev = kb;
            }

            l1Id_C = !l1Id_C;  // switch buffers
            jb_prev = jb;
            ib_prev = ib;
        }
    }

    if (IS_DM_CORE) {
        // DMA core is one index ahead
        snrt_global_barrier();

        // store final tile
        // if (ib_prev >= 0 && jb_prev >= 0) {
            snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N,
                                ldc, FP64);
            snrt_dma_wait_all();
        // }
    } else {
        SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(tileInfo);
    }

    if (bench) snrt_mcycle();
}
