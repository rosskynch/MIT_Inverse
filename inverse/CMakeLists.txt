##
#  Cmake script - learning how to link multiple sources.
##

set(TARGET inverse)
set(MY_LIB_SRC "../src")
set(MY_LIB_INC "../include")

#file(GLOB test_headers "include/*.h") # NOT NEEDED

include_directories(${MY_LIB_INC})
# Method 1: use a global file pointing to a directory with
# all source files contained within it:
# Apparently not the correct way.
#file(GLOB test_sources "src/*.cc*")
#set(TARGET_SRC ${test_sources})

# Method 2: List all files manually, with each named individually.
# This is apparently the "correct" way, but new
# files must be added to this manualy.
set(TARGET_SRC
    ${MY_LIB_SRC}/all_data.cc
    ${MY_LIB_SRC}/curlfunction.cc
    ${MY_LIB_SRC}/backgroundfield.cc
    ${MY_LIB_SRC}/inputtools.cc
    ${MY_LIB_SRC}/inversesolver_voltages.cc
    ${MY_LIB_SRC}/mydofrenumbering.cc
    ${MY_LIB_SRC}/mypreconditioner.cc
    ${MY_LIB_SRC}/mypolynomials.cc
    ${MY_LIB_SRC}/myfe_nedelec.cc
    ${MY_LIB_SRC}/myvectortools.cc
    ${MY_LIB_SRC}/new_voltages.cc
    ${MY_LIB_SRC}/outputtools.cc
    ${MY_LIB_SRC}/forwardsolver.cc
    src/${TARGET}
    )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

TARGET_LINK_LIBRARIES(${TARGET} "-lgfortran -lcomplex_bessel")
