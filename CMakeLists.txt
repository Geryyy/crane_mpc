cmake_minimum_required(VERSION 3.8)
project(crane_mpc)

find_package(ament_cmake REQUIRED)
# Everything that runs here is Python: issue 132 deleted the pre-port C++ node
# `src/` and `include/` carried. What is left of CMake is the Python install,
# the parameter module and the staleness check at the bottom -- an honest
# intermediate state, not a resting place. Issue 133 turns the package into
# `ament_python` and takes this file with it.
#
# **No `find_package(acados)` any more, and none is needed.** The node builds
# its OCP through `acados_template` at startup and acados compiles it into
# `CRANE_MPC_OCP_CACHE`, so acados is resolved by the Python at run time and not
# by CMake at build time. There is no rosdep key for it either; the image is
# what provides it. If a C++ consumer ever comes back, `crane_planning`'s issue
# 043 note and this package's git history carry the config-mode search reasoning
# (`/usr/local/cmake/acadosConfig.cmake`, found with no hint).
find_package(ament_cmake_python REQUIRED)
find_package(generate_parameter_library REQUIRED)

# The parameter declaration, as a Python module. One yaml, generated rather than
# hand-written, so a parameter cannot be declared in one place and read in
# another. It sat under `src/` while a C++ module was generated from the same
# file; with that gone the file moved up beside the package it belongs to.
generate_parameter_module(parameters
  crane_mpc_parameters.yaml)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
ament_python_install_package(${PROJECT_NAME})
# The `.py` suffix stays. It is in the launch contract and in
# `wiki/implementation/ros2_interfaces.md:453`, and dropping it belongs with the
# `ament_python` conversion (issue 133), where the entry point becomes a
# `console_scripts` one anyway.
install(
  PROGRAMS scripts/crane_mpc_node.py
  DESTINATION lib/${PROJECT_NAME})

# The checked-in solver is still what `scripts/export_ocp.py` writes from
# `crane_mpc/problem.py`. **`generated/` is compiled by nobody since the C++
# went** -- the node builds its own solver at startup -- and it is kept anyway,
# as the one reviewable form of the OCP: a diff of that tree is how a change to
# the problem is seen, and it costs a rebuild of nothing. The check is what
# makes the tree worth reviewing, so it is kept too, re-homed from a prerequisite
# of the deleted `ocp` library to a standalone `ALL` target. It regenerates into
# a scratch tree and diffs; it does not compile, and it takes about 2 s.
# `architecture-audit.md` §3 asks for it as a pytest once CMake goes -- that is
# issue 133, and until then a build-time gate runs more often than a test does.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(crane_mpc_generated_is_current ALL
  COMMAND ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/export_ocp.py" --check
  COMMENT "checking generated/ against scripts/export_ocp.py"
  VERBATIM)

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
