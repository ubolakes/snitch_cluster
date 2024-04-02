#include "gemm_decls.h"

#define FLOAT_T fp64

/**
 * Test DMA effective bandwidth.
 * Transfers an array from HBM to TCDM, 
 * then rotates between cluster TCDMs with C2C communication, 
 * then stores the result back to HBM.
*/
void dma_tile2tile_test(const SnblasGemmInfo info, const SNBLAS_GEMM_ARGS(FLOAT_T) args, const SnblasGemmImpl impl) {
    if (!snrt_is_dm_core()) return;

    typedef SNBLAS_GEMM_TCDM(FLOAT_T) TcdmLayout;
    typedef SnblasGemmInfo GemmInfo;
    typedef SNBLAS_GEMM_ARGS(FLOAT_T) GemmArgs;

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

    if (impl.bench) snrt_mcycle();

    uint32_t p[3] = {0, 0, 0};
    uint32_t P[3] = {0, 0, 0};
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

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


    int ib, jb, kb;
    ib = 0;
    jb = 1;
    kb = 1;

    int isb = 0, ksb = 1, jsb = 1;
    // B -> C
    snrt_dma_load_2d_tile_to_tile(args.C, (void*) args.B,
                                kb, 
                                jb, 
                                ksb,
                                jsb, 
                                tileInfo.K,  
                                tileInfo.N, 
                                tileInfo.ldb, ldb, FP64);
    snrt_dma_wait_all();

}