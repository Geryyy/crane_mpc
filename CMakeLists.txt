cmake_minimum_required(VERSION 3.8)
project(crane_mpc)

find_package(ament_cmake REQUIRED)
# The node is Python. What is still C++ here is the solver acados generated and
# the C++ node the Python one is being shown equivalent to before it is deleted.
find_package(ament_cmake_python REQUIRED)
# acados, for the optimal control problem of `wiki/mpc.md` §1.
# `wiki/implementation/libraries.md` §1 lists it for "the optimal control problems
# of `mpc` and `trajectory_planning`" against "SQP loops, condensing, integrators,
# Riccati recursions", so none of those is written here.
#
# **How CI resolves it, since there is no installed `acadosConfig.cmake` where
# CMake's package search usually looks.** The image installs acados to
# `/usr/local` and the config lands at `/usr/local/cmake/acadosConfig.cmake` --
# not `lib/cmake/acados/` and not `share/acados*/cmake/`. It is found without a
# hint anyway, because `<prefix>/cmake/` is one of the directories the config-mode
# search walks and `/usr/local` is a default system prefix. **No `acados_DIR`, no
# `CMAKE_PREFIX_PATH` entry and no hint variable is set anywhere in this package
# or in CI.** The `ACADOS_SOURCE_DIR=/opt/acados` the image exports is for the
# Python templater and takes no part in the C++ find. If a future image drops the
# config, `acados_DIR` is the single variable to set and this is the line that
# would carry the hint. There is no rosdep key for acados, so `package.xml`'s
# `<depend>acados</depend>` resolves to nothing and is documentation; the image is
# what provides it. This is the same resolution `crane_planning` recorded for
# issue 043, re-verified here.
find_package(acados REQUIRED)
# **CasADi is not a dependency of this package any more.** It was, through
# `crane_model::casadi_graph`, which carried it as an INTERFACE usage
# requirement. Issue 071 moved the OCP into `scripts/export_ocp.py`, so what
# reaches the C++ is generated straight-line C: CasADi runs at export time, by
# hand, on a developer's machine. Do not add it back without a reason -- its
# absence from this file is the readable half of grill D1.
find_package(Eigen3 REQUIRED)
find_package(control_msgs REQUIRED)
find_package(crane_model REQUIRED)
find_package(crane_msgs REQUIRED)
# The acados layer, once: the capsule binding both this package and
# `crane_planning` open their generated solvers through.
find_package(crane_ocp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# The horizon as data: the resample of `wiki/mpc.md` §4's grid and the wire form
# `wiki/implementation/ros2_interfaces.md` §4 fixes. ROS-free -- no node, no
# clock, no graph (`wiki/implementation/style_guide.md` §3) -- so all of it is
# tested offline. It links the message package because the wire form is data.
#
# STATIC, for the reason `crane_planning` records:
# crane_model installs a static archive built without position-independent code,
# and the only consumer here is an executable rather than a pluginlib object, so
# no `--exclude-libs` workaround is needed.
add_library(horizon_source STATIC src/horizon_source.cpp)
add_library(${PROJECT_NAME}::horizon_source ALIAS horizon_source)
set_target_properties(horizon_source PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(horizon_source PUBLIC cxx_std_17)
target_include_directories(horizon_source
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(horizon_source
  PUBLIC
    crane_model::crane_model
    ${trajectory_msgs_TARGETS})

# The optimal control problem of `wiki/mpc.md` §1, as the solver
# `scripts/export_ocp.py` generated. ROS-free for the same reason
# `horizon_source` is, so all of it is tested offline.
#
# What is compiled here is **machine output that is checked in**, one tree, plus
# `src/ocp_solver.cpp` -- the binding that opens the generated solver, writes
# what stays runtime-settable, and reads acados' own answers back.
# There is no expression building at run time and no `casadi::Function` alive in
# the process.
#
# The file list is globbed rather than written out because the export decides its
# own file set: `nh_e` becoming non-zero, or a second cost stage appearing, adds
# a translation unit that no one should have to remember to list. `CONFIGURE_DEPENDS`
# makes a re-export re-run CMake rather than link yesterday's objects.
#
# Every generated `.c` is compiled with `-w`: it is CasADi and acados output, it
# is regenerated rather than edited, and a warning on it is a warning nobody can
# act on without breaking the byte-for-byte regeneration check
# (`generated/README.md`).
set(CRANE_MPC_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
# One solver: the PZS100's, constrained. The description is baked in, so an
# artifact is one machine's, and the PZS100 is the machine that has to run.
set(CRANE_MPC_SOLVER crane_mpc_pzs100)
file(GLOB_RECURSE CRANE_MPC_GENERATED_SOURCES CONFIGURE_DEPENDS
  "${CRANE_MPC_GENERATED_DIR}/${CRANE_MPC_SOLVER}/*.c")
if(NOT CRANE_MPC_GENERATED_SOURCES)
  message(FATAL_ERROR
    "`generated/${CRANE_MPC_SOLVER}/` is empty. It is checked in, not built: run "
    "`./scripts/export_ocp.py` from ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()
set(CRANE_MPC_GENERATED_INCLUDES
  "${CRANE_MPC_GENERATED_DIR}"
  "${CRANE_MPC_GENERATED_DIR}/${CRANE_MPC_SOLVER}")
set_source_files_properties(${CRANE_MPC_GENERATED_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")

add_library(ocp STATIC
  src/ocp_solver.cpp
  ${CRANE_MPC_GENERATED_SOURCES})
add_library(${PROJECT_NAME}::ocp ALIAS ocp)
set_target_properties(ocp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ocp PUBLIC cxx_std_17)
# The generated tree is PRIVATE, which is the same rule `crane_model`'s contract
# §9 applies to itself: `ocp_solver.hpp` names no acados type, so the solver
# backend is an implementation detail of this library and a consumer of the
# horizon producer compiles against neither acados nor the generated headers.
target_include_directories(ocp
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CRANE_MPC_GENERATED_INCLUDES})
target_link_libraries(ocp
  PUBLIC
    crane_model::crane_model
    Eigen3::Eigen
  PRIVATE
    acados
    crane_ocp::crane_ocp)

generate_parameter_library(crane_mpc_parameters
  src/crane_mpc_parameters.yaml)
# The same yaml again, for the Python node: one declaration, two languages, so
# a parameter cannot be added to one and missed by the other. The target name
# differs from the C++ one only because CMake target names are global.
generate_parameter_module(parameters
  src/crane_mpc_parameters.yaml)

# The adapter, as a library rather than as part of the executable: the stream
# test constructs this node object in process, so what is asserted is the node
# the deployment runs and not a second one written for the test.
#
# A horizon longer than `crane_mpc::kHorizonKnotCapacity` (horizon_source.hpp) is
# refused at construction rather than published as something no receiver can read.
add_library(mpc_node STATIC src/mpc_node.cpp)
add_library(${PROJECT_NAME}::mpc_node ALIAS mpc_node)
set_target_properties(mpc_node PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(mpc_node PUBLIC cxx_std_17)
target_include_directories(mpc_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_node
  PUBLIC
    horizon_source
    ocp
    rclcpp::rclcpp
    ${control_msgs_TARGETS}
    ${crane_msgs_TARGETS}
    ${diagnostic_msgs_TARGETS}
    ${nav_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${trajectory_msgs_TARGETS}
  PRIVATE
    crane_mpc_parameters)

# Beside the manager and not inside it (ROS 2 Interfaces §2), so it is an
# executable a launch profile composes rather than a plugin a spawner loads.
add_executable(crane_mpc_node src/mpc_main.cpp)
target_link_libraries(crane_mpc_node PRIVATE mpc_node)

install(
  DIRECTORY include/
  DESTINATION include)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(
  TARGETS horizon_source mpc_node ocp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(
  TARGETS crane_mpc_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
ament_python_install_package(${PROJECT_NAME})
# `crane_mpc_node.py` beside `crane_mpc_node` while both exist: they are the
# same node in two languages and the parity run needs to start either. The
# suffix goes when the C++ does.
install(
  PROGRAMS scripts/crane_mpc_node.py
  DESTINATION lib/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(horizon_source mpc_node ocp)
ament_export_dependencies(
  control_msgs
  crane_model
  crane_msgs
  crane_ocp
  diagnostic_msgs
  Eigen3
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  trajectory_msgs)

# The checked-in solver is still what `scripts/export_ocp.py` writes from
# `crane_model/scripts/crane_symbolic.py`. Build time and not a test, for the
# reason `crane_planning` gives: a stale `generated/` tree ships a binary that
# solves a different problem than the source says, and with the parity tests
# retired this is the only thing that ties the two together. The check
# regenerates into a scratch tree and diffs.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(crane_mpc_generated_is_current ALL
  COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_ocp.py" --check
  COMMENT "checking generated/ against scripts/export_ocp.py"
  VERBATIM)
add_dependencies(ocp crane_mpc_generated_is_current)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The node's own cycle, the solver wrapper and the resample. The solver tests
  # compile a solver on their first run and reuse the cache after.
  # Run from the build tree, not from the source: with the source directory as
  # the working directory `import crane_mpc` finds the source package, which has
  # no generated parameter module in it and is not what a deployment runs.
  ament_add_pytest_test(crane_mpc_python ${CMAKE_CURRENT_SOURCE_DIR}/test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 900)
endif()

ament_package()
