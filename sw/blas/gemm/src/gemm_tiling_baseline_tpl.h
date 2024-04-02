#ifndef FLOAT_T 
#error "Define FLOAT_T to use this template."
#endif

#ifndef IS_DM_CORE
#error "Define IS_DM_CORE to use this template."
#endif

#include "gemm_kernel.h"

__attribute__((flatten)) // Force inline called functions
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
    TcdmLayout* l1 = snrt_l1_next();

    // Which buffer is the valid data in for computation
    bool l1Id_A = true;
    bool l1Id_B = true;
    bool l1Id_C = false;

    // Initialize indices
    const uint32_t PI = 1, PJ = 1;
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev = -1, jb_prev = -1, kb_prev = -1;

    bool storeC = false;

    GemmInfo tileInfo = {0};
    tileInfo.M   = L1_M;
    tileInfo.N   = L1_N;
    tileInfo.K   = L1_K;
    tileInfo.ta  = info.ta ^ TA_TILE;
    tileInfo.tb  = info.tb ^ TB_TILE;
    tileInfo.tc  = info.tc ^ TC_TILE; // TODO: implement transposed blocking
    tileInfo.lda = tileInfo.ta ? tileInfo.M : tileInfo.K;
    tileInfo.ldb = tileInfo.tb ? tileInfo.K : tileInfo.N;
    tileInfo.ldc = tileInfo.tc ? tileInfo.M : tileInfo.N;

    // create function ptr for dma loading
    const snrt_dma_load_2d_tile_t load_tile_A = TA_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile;
    const snrt_dma_load_2d_tile_t load_tile_B = TB_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile;
    const snrt_dma_load_2d_tile_t load_tile_C = (args.beta == (FLOAT_T)0.0) ? &load_zero_tile : 
                                                TC_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile;
    const snrt_dma_load_2d_tile_t store_tile_C = TC_TILE ? &snrt_dma_store_2d_tile_transpose : &snrt_dma_store_2d_tile;

    if (!IS_DM_CORE) {
        SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(tileInfo, impl);
        if (impl.bench) snrt_mcycle();
    }

    for(ib = pi; ib <  M / L1_M; ib += PI) {
        for(jb = pj; jb < N / L1_N; jb += PJ) {
            FLOAT_T* const l1_C = l1[l1Id_C].C;

            if (IS_DM_CORE) {
                dump_ib(ib);
                dump_jb(jb);
                (*load_tile_C)(l1_C, (void*) C, ib, jb, L1_M, L1_N, ldc, FP64); // apply args.beta here
                if (ib_prev >= 0 && jb_prev >= 0) storeC = true; // store after k-accumulation is complete
            }

            for(kb = 0; kb < K / L1_K; kb++) {
                // Switch buffers when the indices have changed
                l1Id_A = !l1Id_A;
                l1Id_B = !l1Id_B;

                FLOAT_T* const l1_A = l1[l1Id_A].A;
                FLOAT_T* const l1_B = l1[l1Id_B].B;

                if (IS_DM_CORE) {
                    dump_kb(kb);
                    (*load_tile_A)(l1_A, (void*) A, info.ta ? kb : ib, 
                                                    info.ta ? ib : kb, 
                                                    info.ta ? L1_K : L1_M, 
                                                    info.ta ? L1_M : L1_K, lda, FP64);
                    (*load_tile_B)(l1_B, (void*) B, info.tb ? jb : kb, 
                                                    info.tb ? kb : jb, 
                                                    info.tb ? L1_N : L1_K, 
                                                    info.tb ? L1_K : L1_N, ldb, FP64);
                    snrt_dma_wait_all();
                    snrt_cluster_hw_barrier();
                    if (impl.bench) snrt_mcycle();
                } else {
                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1_A;
                    tileArgs.B     = l1_B;
                    tileArgs.C     = l1_C;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = 1; // always accumulate partial result, args.beta already applied by dma
                        
                    SNBLAS_GEMM_CLUSTER_KERNEL_COMPUTE(FLOAT_T)(tileInfo, tileArgs, impl);
                }
                
                // if (impl.bench) snrt_mcycle();
                // TODO: distinguish for compute and dma core, compute cores do indexing first and then barrier, better use of waiting time
                //       no need for extra barrier to offset dma cores

                if (IS_DM_CORE) {
                    if (storeC) {
                        storeC = false;
                        (*store_tile_C)(C, l1[!l1Id_C].C, ib_prev,
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
        snrt_cluster_hw_barrier();

        // store final tile
        // if (ib_prev >= 0 && jb_prev >= 0) {
            (*store_tile_C)(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N,
                                ldc, FP64);
            snrt_dma_wait_all();
        // }
    } else {
        snrt_fpu_fence();
        SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(tileInfo, impl);
        snrt_cluster_hw_barrier();
    }
    snrt_fpu_fence();
    snrt_global_barrier();
}
