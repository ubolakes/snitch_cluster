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

#if USE_C2C_TILES
#define SNRT_BARRIER(with_mcycle) \
snrt_global_barrier(); \
if (with_mcycle) snrt_mcycle(); 
#else
#define SNRT_BARRIER(with_mcycle) \
snrt_cluster_hw_barrier(); \
if (with_mcycle) snrt_mcycle(); 
#endif

__attribute__((flatten)) // Force inline called functions
void SNBLAS_GEMM_TILING(2dpipe, FLOAT_T, IS_DM_CORE, BETA_NZ) (const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl) {

    /**
     * Problem is double buffered in L1. The buffer that is used is toggled at
     * each iteration. The DMA cores are one index step ahead so they load the
     * data in advance into the buffer that will be used.
     */

    typedef SNBLAS_GEMM_TCDM(FLOAT_T) TcdmLayout;
    typedef SnblasGemmInfo GemmInfo;
    typedef SNBLAS_GEMM_ARGS(FLOAT_T) GemmArgs;

    // if (impl.bench) snrt_mcycle();

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
    const uint32_t PI = 2, PJ = 2; // 1Q4C
    // const uint32_t PI = 1, PJ = 1; // 1Q1C
    const uint32_t pi = p[1] / PJ;
    const uint32_t pj = p[1] % PJ;

    int ib, jb, kb;
    int ib_prev = -1, jb_prev = -1, kb_prev = -1;
    bool ib_dir = false, jb_dir = false, kb_dir = false;

    bool storeC = false;

    // -- Compute C2C sources for 2D pipeline
    const uint32_t pk = USE_C2C_TILES ? (PI + 2 * PJ - pi - pj - 1) % PJ : 0; // pipeline step
    const int PK      = USE_C2C_TILES ? PJ : 1;                               // pipeline depth

    // Determine C2C source cluster index for each matrix, < 0 is from DRAM
    TcdmLayout* c2cL1_A = NULL;
    TcdmLayout* c2cL1_B = NULL;
    if (IS_DM_CORE) {
        // -- Sync l1 pointers between clusters
        TcdmLayout* l1Ptr[P[1]];
        for (int i = 0; i < P[1]; ++i)
            l1Ptr[i] = (TcdmLayout*)((uint32_t)l1 + cluster_offset * (i - p[1]));

        // 2D pipeline indices, see notes or python notebook for details
        // Works for PI = PJ
        const uint32_t p_srcA = pi * PJ + ((2 * PJ - pi - pk) % PJ);
        const uint32_t p_srcB = pj + PJ * ((2 * PJ - pj - pk) % PJ);

        const bool fetch_dram = pk == 0;
        c2cL1_A = fetch_dram ? NULL : l1Ptr[p_srcA];
        c2cL1_B = fetch_dram ? NULL : l1Ptr[p_srcB];

        // dump_p_src(fetch_dram ? -1 : p_srcA);
        // dump_p_src(fetch_dram ? -1 : p_srcB);
    }

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
    const snrt_dma_load_2d_tile_t load_tile_C = BETA_NZ ? 
                                                TC_TILE ? &snrt_dma_load_2d_tile_transpose : &snrt_dma_load_2d_tile
                                                : &load_zero_tile;
    const snrt_dma_load_2d_tile_t store_tile_C = TC_TILE ? &snrt_dma_store_2d_tile_transpose : &snrt_dma_store_2d_tile;

    // if (impl.bench) snrt_mcycle();

    if (!IS_DM_CORE) {
        SNBLAS_GEMM_CLUSTER_KERNEL_INIT(FLOAT_T)(tileInfo, impl);
        // SNRT_BARRIER(false) // DMA core is one index ahead
    }

    // Wait for pipeline to be filled
    for (int pipeline = pk; pipeline > 0; --pipeline) {
        SNRT_BARRIER(false)
    }

    FOR_EACH(ib, pi, M / L1_M, PI, ib_dir) {
        FOR_EACH(jb, pj, N / L1_N, PJ, jb_dir) {
            FLOAT_T* const l1_C = l1[l1Id_C].C;

            if (IS_DM_CORE) {
                dump_ib(ib);
                dump_jb(jb);
                (*load_tile_C)(l1_C, (void*) C, ib, jb, L1_M, L1_N, ldc, FP64); // apply args.beta here
                if (ib_prev >= 0 /* && jb_prev >= 0 */) storeC = true; // store after k-accumulation is complete
            }

            FOR_EACH(kb, 0, K / L1_K, 1, kb_dir) {
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
                        if (c2cL1_A == NULL)
                            (*load_tile_A)(l1_A, (void*) A, info.ta ? kb : ib, 
                                                            info.ta ? ib : kb, 
                                                            info.ta ? L1_K : L1_M, 
                                                            info.ta ? L1_M : L1_K, lda, FP64);
                        else
                            snrt_dma_start_1d(l1_A, c2cL1_A[l1Id_A].A, L1_M * L1_K * FP64);
                    }
                    if (loadB) {
                        if (c2cL1_B == NULL)
                            (*load_tile_B)(l1_B, (void*) B, info.tb ? jb : kb, 
                                                            info.tb ? kb : jb, 
                                                            info.tb ? L1_N : L1_K, 
                                                            info.tb ? L1_K : L1_N, ldb, FP64);
                        else
                            snrt_dma_start_1d(l1_B, c2cL1_B[l1Id_B].B, L1_K * L1_N * FP64);
                    }

                    snrt_dma_wait_all();
                    SNRT_BARRIER(impl.bench)
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
                

                if (IS_DM_CORE) {
                    if (storeC) {
                        storeC = false;
                        (*store_tile_C)(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N, ldc, FP64);
                    }
                }
                kb_prev = kb;
            }

            l1Id_C = !l1Id_C;  // switch buffers
            jb_prev = jb;
            ib_prev = ib;
        }
    }

    if (IS_DM_CORE) { // DMA core is one index ahead
        SNRT_BARRIER(false) 

        // store final tile
        // if (ib_prev >= 0 && jb_prev >= 0) {
            (*store_tile_C)(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N, ldc, FP64);
            snrt_dma_wait_all();
        // }
    } else {
        snrt_fpu_fence();
        SNBLAS_GEMM_CLUSTER_KERNEL_DEINIT(FLOAT_T)(tileInfo, impl);
        SNRT_BARRIER(false)
    }

    // Wait for pipeline to be emptied
    for (int pipeline = pk; pipeline < PK - 1; ++pipeline) {
        SNRT_BARRIER(false)
    }
    SNRT_BARRIER(false)
}
