/*
 * StoRM - Build-in Options
 *
 * This file is parsed by CMake during makefile generation
 * It contains information such as the base path to the test/example data
 */

#ifndef STORM_GENERATED_STORMCONFIG_H_
#define STORM_GENERATED_STORMCONFIG_H_

// The directory of the sources from which Storm was built.
#define STORM_SOURCE_DIR "/home/ss96869/storage/NSVS/vendors/storm-stable"

// The directory of the test resources used in the tests (model files, ...).
#define STORM_TEST_RESOURCES_DIR "/home/ss96869/storage/NSVS/vendors/storm-stable/resources/examples/testfiles"

// The directory in which Storm was built.
#define STORM_BUILD_DIR "/home/ss96869/storage/NSVS/vendors/storm-stable/build"

// Boost include directory used during compilation.
#define STORM_BOOST_INCLUDE_DIR "/usr/include"

// Carl include directory used during compilation.
#define STORM_CARL_INCLUDE_DIR "/home/ss96869/storage/NSVS/vendors/carl-storm/src"

// Whether Gurobi is available and to be used (define/undef)
/* #undef STORM_HAVE_GUROBI */

// Whether GLPK is available and to be used (define/undef)
#define STORM_HAVE_GLPK

// Whether Z3 is available and to be used (define/undef)
/* #undef STORM_HAVE_Z3 */

// Whether the optimization feature of Z3 is available and to be used (define/undef)
/* #undef STORM_HAVE_Z3_OPTIMIZE */

// Version of Z3 used by Storm.
#define STORM_Z3_VERSION_MAJOR 
#define STORM_Z3_VERSION_MINOR 
#define STORM_Z3_VERSION_PATCH 
#define STORM_Z3_VERSION 
/* #undef STORM_Z3_API_USES_STANDARD_INTEGERS */

// Whether MathSAT is available and to be used (define/undef)
/* #undef STORM_HAVE_MSAT */

// Whether SoPlex is available and to be used
/* #undef STORM_HAVE_SOPLEX */

// Whether benchmarks from QVBS can be used as input
/* #undef STORM_HAVE_QVBS */

// The root directory of QVBS
/* #undef STORM_QVBS_ROOT */

// Whether Intel Threading Building Blocks are available and to be used (define/undef)
/* #undef STORM_HAVE_INTELTBB */

// Whether support for parametric systems should be enabled
/* #undef PARAMETRIC_SYSTEMS */

// Whether CLN is available and to be used (define/undef)
#define STORM_HAVE_CLN

// Include directory for CLN headers
#define CLN_INCLUDE_DIR "/usr/include"

// Whether GMP is available  (it is always available nowadays)
#define STORM_HAVE_GMP

// Include directory for GMP headers
#define GMP_INCLUDE_DIR "/usr/include/x86_64-linux-gnu"
#define GMPXX_INCLUDE_DIR "/usr/include"

// Whether carl is available and to be used.
#define STORM_HAVE_CARL
// Whether carl has headers for forward declarations
#define STORM_CARL_SUPPORTS_FWD_DECL
// Version of CARL used by Storm.
#define STORM_CARL_VERSION_MAJOR 14
#define STORM_CARL_VERSION_MINOR 32
#define STORM_CARL_VERSION 14.32
/* #undef STORM_Z3_API_USES_STANDARD_INTEGERS */

/* #undef STORM_USE_CLN_EA */

#define STORM_USE_CLN_RF

/* #undef STORM_HAVE_XERCES */

// Whether Spot is available and to be used
#define STORM_HAVE_SPOT

// Whether LTL model checking shall be enabled
#ifdef STORM_HAVE_SPOT
        #define STORM_HAVE_LTL_MODELCHECKING_SUPPORT
#endif // STORM_HAVE_SPOT

// Whether smtrat is available and to be used.
/* #undef STORM_HAVE_SMTRAT */

/* #undef STORM_LOGGING_FRAMEWORK */

#define STORM_LOG_DISABLE_DEBUG

#endif // STORM_GENERATED_STORMCONFIG_H_
