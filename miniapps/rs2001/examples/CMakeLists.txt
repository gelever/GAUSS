add_executable(finitevolume finitevolume.cpp)
target_link_libraries(finitevolume ReservoirSimulator2000)

add_executable(timestep timestep.cpp)
target_link_libraries(timestep ReservoirSimulator2000)

add_executable(mltopo mltopo.cpp)
target_link_libraries(mltopo ReservoirSimulator2000)

add_executable(PDESamplerTest PDESamplerTest.cpp)
target_link_libraries(PDESamplerTest ReservoirSimulator2000)

add_executable(MLMC_SPE10 MLMC_SPE10.cpp)
target_link_libraries(MLMC_SPE10 ReservoirSimulator2000)

add_executable(dirichlet dirichlet.cpp)
target_link_libraries(dirichlet ReservoirSimulator2000)

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/test.py"
  "${PROJECT_BINARY_DIR}/examples/test.py" @ONLY)

add_test(rs2001_eigenvector1 python test.py eigenvector1)
add_test(rs2001_eigenvector4 python test.py eigenvector4)
add_test(rs2001_fv-hybridization python test.py fv-hybridization)
add_test(rs2001_slice19 python test.py slice19)
add_test(rs2001_timestep python test.py timestep)

add_test(rs2001_pareigenvector1 python test.py pareigenvector1)
add_test(rs2001_pareigenvector4 python test.py pareigenvector4)
add_test(rs2001_parfv-hybridization python test.py parfv-hybridization)
add_test(rs2001_parslice19 python test.py parslice19)
add_test(rs2001_partimestep python test.py partimestep)

if (NOT APPLE)
    add_test(rs2001_fv-metis python test.py fv-metis)

    # Takes a relatively long while to run:
    # add_test(rs2001_veigenvector python test.py veigenvector)
endif()
