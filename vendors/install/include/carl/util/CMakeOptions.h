/**
 * @file CMakeOptions.h
 * @author Gereon Kremer <gereon.kremer@cs.rwth-aachen.de>
 */

#pragma once

#include <iostream>

namespace carl {

void printCMakeOptions(std::ostream& os);

namespace cmakeoptions {

	static constexpr auto _ALLOW_SHIPPED_CLN = "OFF";
	static constexpr auto _ALLOW_SHIPPED_GINAC = "OFF";
	static constexpr auto _ALLWARNINGS = "OFF";
	static constexpr auto _BUILD_DOXYGEN = "OFF";
	static constexpr auto _BUILD_GMOCK = "ON";
	static constexpr auto _BUILD_STATIC = "OFF";
	static constexpr auto _Boost_DIR = "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0";
	static constexpr auto _CARL_BIN_INSTALL_DIR = "/home/ss96869/storage/NSVS/vendors/install/bin";
	static constexpr auto _CARL_CMAKE_INSTALL_DIR = "/home/ss96869/storage/NSVS/vendors/install/lib/cmake";
	static constexpr auto _CARL_COMPILE_BENCHMARKS = "OFF";
	static constexpr auto _CARL_EXPORT_TO_CMAKE = "ON";
	static constexpr auto _CARL_INCLUDE_INSTALL_DIR = "/home/ss96869/storage/NSVS/vendors/install/include";
	static constexpr auto _CARL_LIB_INSTALL_DIR = "/home/ss96869/storage/NSVS/vendors/install/lib";
	static constexpr auto _CARL_LOGGING = "OFF";
	static constexpr auto _CARL_TARGETS = "lib_carl;eigen3carl";
	static constexpr auto _CARL_WARNING_AS_ERROR = "OFF";
	static constexpr auto _CLANG_SANITIZER = "none";
	static constexpr auto _CMAKE_BUILD_TYPE = "Release";
	static constexpr auto _CMAKE_INSTALL_DIR = "";
	static constexpr auto _CMAKE_INSTALL_PREFIX = "/home/ss96869/storage/NSVS/vendors/install";
	static constexpr auto _DEVELOPER = "OFF";
	static constexpr auto _ENABLE_PACKAGING = "OFF";
	static constexpr auto _EXCLUDE_TESTS_FROM_ALL = "OFF";
	static constexpr auto _EXECUTABLE_OUTPUT_PATH = "/home/ss96869/storage/NSVS/vendors/carl-storm/build/bin";
	static constexpr auto _FETCHCONTENT_BASE_DIR = "/home/ss96869/storage/NSVS/vendors/carl-storm/build/_deps";
	static constexpr auto _FETCHCONTENT_FULLY_DISCONNECTED = "OFF";
	static constexpr auto _FETCHCONTENT_QUIET = "ON";
	static constexpr auto _FETCHCONTENT_SOURCE_DIR_GOOGLETEST = "";
	static constexpr auto _FETCHCONTENT_UPDATES_DISCONNECTED = "OFF";
	static constexpr auto _FETCHCONTENT_UPDATES_DISCONNECTED_GOOGLETEST = "OFF";
	static constexpr auto _FORCE_SHIPPED_GMP = "OFF";
	static constexpr auto _FORCE_SHIPPED_RESOURCES = "OFF";
	static constexpr auto _GTEST_HAS_ABSL = "OFF";
	static constexpr auto _INSTALL_GTEST = "ON";
	static constexpr auto _LOGGING_DISABLE_INEFFICIENT = "OFF";
	static constexpr auto _PORTABLE = "OFF";
	static constexpr auto _PRUNE_MONOMIAL_POOL = "ON";
	static constexpr auto _THREAD_SAFE = "ON";
	static constexpr auto _TIMING = "OFF";
	static constexpr auto _USE_BLISS = "OFF";
	static constexpr auto _USE_CLN_NUMBERS = "TRUE";
	static constexpr auto _USE_COCOA = "OFF";
	static constexpr auto _USE_GINAC = "TRUE";
	static constexpr auto _USE_MPFR_FLOAT = "OFF";
	static constexpr auto _USE_Z3_NUMBERS = "OFF";
}

}
