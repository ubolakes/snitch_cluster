add_subdirectory(blas/gemm)
add_subdirectory(occamy/blas/gemm)

add_custom_target(all.test)
add_dependencies(all.test gemm.elf gemm_oc.elf)