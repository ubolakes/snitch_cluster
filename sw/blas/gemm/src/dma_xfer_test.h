#include "gemm_decls.h"

/**
 * Test DMA effective bandwidth.
 * Transfers an array from HBM to TCDM, 
 * then rotates between cluster TCDMs with C2C communication, 
 * then stores the result back to HBM.
*/
void dma_xfer_test(const double* A, const uint32_t N, const bool bench) {
    if (!snrt_is_dm_core()) return;

    if (bench) snrt_mcycle();

    uint32_t p[3] = {0, 0, 0};
    uint32_t P[3] = {0, 0, 0};
    ocrt_thread_idx(p);
    ocrt_compute_thread_num(P);

    double* l1 = snrt_l1_next();

    // -- Sync l1 pointers between clusters
    double* l1Ptr[P[1]];
    for (int i = 0; i < P[1]; ++i)
        l1Ptr[i] = (double*)((uint32_t)l1 + cluster_offset * (i - snrt_cluster_idx()));

    uint32_t n = N / P[1];

    // Rotate data between clusters
    const uint32_t p_c2c = (p[1] + 1) % P[1];
    double* l1_c2c = l1Ptr[p_c2c];
    asm volatile("" ::: "memory");
    snrt_global_dm_core_barrier();
    asm volatile("" ::: "memory");

    // HBM -> TCDM
    if (bench) snrt_mcycle();
    snrt_dma_start_1d((void*)l1, (void*)(A + p[1]*n), n * sizeof(double));
    snrt_dma_wait_all();
    if (bench) snrt_mcycle();
    snrt_global_dm_core_barrier();
    if (bench) snrt_mcycle();

    // TCDM -> TCDM
    snrt_dma_start_1d((void*)l1_c2c, (void*)l1, n * sizeof(double));
    snrt_dma_wait_all();
    if (bench) snrt_mcycle();
    snrt_global_dm_core_barrier();
    if (bench) snrt_mcycle();
    
    // TCDM -> HBM
    snrt_dma_start_1d((void*)(A + p[1]*n), (void*)l1, n * sizeof(double));
    snrt_dma_wait_all();
    if (bench) snrt_mcycle();
    snrt_global_dm_core_barrier();
    if (bench) snrt_mcycle();

}