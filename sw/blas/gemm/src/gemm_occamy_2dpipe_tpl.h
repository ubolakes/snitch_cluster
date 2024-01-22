#ifndef IS_DM_CORE
#error "Define IS_DM_CORE to use this template."
#elif IS_DM_CORE==true
void gemm_oc_dm
#else
void gemm_oc_compute
#endif
(const GemmInfo info, const GemmArgs args, const bool bench) {
    
    /**
     * Problem is double buffered in L1. The buffer that is used is toggled at
     * each iteration. The DMA cores are one index step ahead so they load the
     * data in advance into the buffer that will be used.
     */

    const uint32_t M   = info.M;
    const uint32_t N   = info.N;
    const uint32_t K   = info.K;
    const uint32_t lda = info.lda;
    const uint32_t ldb = info.ldb;
    const uint32_t ldc = info.ldc;
    const uint32_t ta  = info.ta;
    const uint32_t tb  = info.tb;

    const double* const A = args.A;
    const double* const B = args.B;
          double* const C = args.C;
    const double alpha    = args.alpha;
    const double beta     = args.beta;

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
    bool ib_dir = false, jb_dir = false, kb_dir = false;

    bool storeC = false;

    // -- Compute C2C sources for 2D pipeline
    const uint32_t pk = (PI + 2 * PJ - pi - pj - 1) % PJ; // pipeline step
    const int PK      = PJ;                               // pipeline depth

    // Determine C2C source cluster index for each matrix, < 0 is from DRAM
    TcdmLayout* c2cL1_A = NULL;
    TcdmLayout* c2cL1_B = NULL;
    if (IS_DM_CORE) {
        // -- Sync l1 pointers between clusters
        TcdmLayout* l1Ptr[SNRT_CLUSTER_NUM];
        for (int i = 0; i < snrt_cluster_num(); ++i)
            l1Ptr[i] = (TcdmLayout*)((uint32_t)l1 + cluster_offset * (i - snrt_cluster_idx()));

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
    tileInfo.lda = L1_LDA;
    tileInfo.ldb = L1_LDB;
    tileInfo.ldc = L1_LDC;
    tileInfo.ta  = false;
    tileInfo.tb  = false;

    if (bench) snrt_mcycle();

    if (!IS_DM_CORE) {
        gemm_cluster_kernel_init(tileInfo);
        snrt_global_barrier();  // DMA core is one index ahead
    }

    // Wait for pipeline to be filled
    for (int pipeline = pk; pipeline > 0; --pipeline) {
        snrt_global_barrier();
    }

    FOR_EACH(ib, pi, M / L1_M, PI, ib_dir, ib_prev) {
        FOR_EACH(jb, pj, N / L1_N, PJ, jb_dir, jb_prev) {
            double* const l1_C = l1[l1Id_C].C;

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

                double* const l1_A = l1[l1Id_A].A;
                double* const l1_B = l1[l1Id_B].B;

                if (IS_DM_CORE) {
                    dump_kb(kb);
                    if (loadA) {
                        if (c2cL1_A == NULL)
                            snrt_dma_load_2d_tile(l1_A, (void*) A, ib, kb, L1_M, L1_K, lda,
                                                FP64);
                        else {
                            double* const c2c_A = c2cL1_A[l1Id_A].A;
                            snrt_dma_start_1d(l1_A, c2c_A, L1_M * L1_K * FP64);
                        }
                    }
                    if (loadB) {
                        if (c2cL1_B == NULL)
                            snrt_dma_load_2d_tile(l1_B, (void*) B, kb, jb, L1_K, L1_N, ldb,
                                                FP64);
                        else {
                            double* const c2c_B = c2cL1_B[l1Id_B].B;
                            snrt_dma_start_1d(l1_B, c2c_B, L1_K * L1_N * FP64);
                        }
                    }

                    snrt_dma_wait_all();
                } else {
                    // solve block already in l1, parallelize inside each
                    // cluster
                    // gemm_cluster_kernel_baseline(alpha, beta, L1_M, L1_N, L1_K, l1_A,
                    //                     l1_B, l1_C, L1_LDA, L1_LDB, L1_LDC);

                    // gemm(FP64, 0, true, false, false, L1_M, L1_N, L1_K, alpha,
                    //      l1_A, L1_LDA, l1_B, L1_LDB, beta, l1_C, L1_LDC);

                    GemmArgs tileArgs = {0};
                    tileArgs.A     = l1_A;
                    tileArgs.B     = l1_B;
                    tileArgs.C     = l1_C;
                    tileArgs.alpha = alpha;
                    tileArgs.beta  = beta;
                    
                    gemm_cluster_kernel(tileInfo, tileArgs);
                }

                snrt_global_barrier();

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
        snrt_global_barrier();  // DMA core is one index ahead

        // store final tile
        // if (ib_prev >= 0 && jb_prev >= 0) {
            snrt_dma_store_2d_tile(C, l1[!l1Id_C].C, ib_prev, jb_prev, L1_M, L1_N,
                                ldc, FP64);
            snrt_dma_wait_all();
        // }
    } else {
        gemm_cluster_kernel_deinit(tileInfo);
    }

    // Wait for pipeline to be emptied
    for (int pipeline = pk; pipeline < PK - 1; ++pipeline) {
        snrt_global_barrier();
    }
}
