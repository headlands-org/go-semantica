# Makefile for C++ benchmark using llama.cpp
# Task 1.2: Configure llama.cpp library linking with auto-detection

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
LDFLAGS := -pthread

# Optimization flags (default: release)
RELEASE_FLAGS := -O3 -march=native -DNDEBUG
DEBUG_FLAGS := -g -O0 -DDEBUG

# Default to release build
CXXFLAGS += $(RELEASE_FLAGS)

# ============================================================================
# llama.cpp Detection and Configuration
# ============================================================================

# Skip llama.cpp checks if building test target only
ifeq ($(MAKECMDGOALS),test)
    SKIP_LLAMA := 1
endif
ifeq ($(MAKECMDGOALS),test_metrics)
    SKIP_LLAMA := 1
endif
ifeq ($(MAKECMDGOALS),clean)
    SKIP_LLAMA := 1
endif

ifndef SKIP_LLAMA

# 1. Check LLAMA_CPP_PATH environment variable first
ifdef LLAMA_CPP_PATH
    LLAMA_DIR := $(LLAMA_CPP_PATH)
else
    # 2. Fallback to ../llama.cpp/ relative path
    LLAMA_DIR := ../../llama.cpp
endif

# Verify llama.cpp directory exists
ifeq ($(wildcard $(LLAMA_DIR)),)
    $(error llama.cpp not found at $(LLAMA_DIR). Set LLAMA_CPP_PATH environment variable or clone llama.cpp to ../llama.cpp/)
endif

# Detect header include path
LLAMA_INCLUDE := $(LLAMA_DIR)/include
ifeq ($(wildcard $(LLAMA_INCLUDE)/llama.h),)
    # Try alternate location (some builds put headers in root)
    LLAMA_INCLUDE := $(LLAMA_DIR)
    ifeq ($(wildcard $(LLAMA_INCLUDE)/llama.h),)
        $(error llama.h not found in $(LLAMA_DIR)/include or $(LLAMA_DIR). Check your llama.cpp installation.)
    endif
endif

# Auto-detect library: prefer shared library, fallback to static
LLAMA_SHARED := $(wildcard $(LLAMA_DIR)/build/bin/libllama.so $(LLAMA_DIR)/libllama.so $(LLAMA_DIR)/build/libllama.so $(LLAMA_DIR)/lib/libllama.so)
LLAMA_STATIC := $(wildcard $(LLAMA_DIR)/build/bin/libllama.a $(LLAMA_DIR)/libllama.a $(LLAMA_DIR)/build/libllama.a $(LLAMA_DIR)/lib/libllama.a)

ifdef LLAMA_SHARED
    # Use shared library with rpath for runtime resolution
    LLAMA_LIB := $(firstword $(LLAMA_SHARED))
    LLAMA_LIB_DIR := $(dir $(LLAMA_LIB))
    LDFLAGS += -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(abspath $(LLAMA_LIB_DIR)) -lllama
    LLAMA_LIB_TYPE := shared
else ifdef LLAMA_STATIC
    # Use static library
    LLAMA_LIB := $(firstword $(LLAMA_STATIC))
    LDFLAGS += $(LLAMA_LIB)
    LLAMA_LIB_TYPE := static
else
    $(error No llama.cpp library found (libllama.so or libllama.a) in $(LLAMA_DIR). Build llama.cpp first: cd $(LLAMA_DIR) && cmake -B build && cmake --build build)
endif

endif # ifndef SKIP_LLAMA

# ============================================================================
# Project Structure
# ============================================================================

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
TARGET := $(BUILD_DIR)/benchmark_cpp
TEST_TARGET := $(BUILD_DIR)/test_data_test
TEST_METRICS_TARGET := $(BUILD_DIR)/test_metrics

# Source files (auto-detected, exclude test files)
SOURCES := $(filter-out $(SRC_DIR)/test_data_test.cpp $(SRC_DIR)/test_metrics.cpp,$(wildcard $(SRC_DIR)/*.cpp))
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Test sources (only test_data.cpp and test_data_test.cpp)
TEST_SOURCES := $(SRC_DIR)/test_data.cpp $(SRC_DIR)/test_data_test.cpp
TEST_OBJECTS := $(TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Metrics test sources (metrics.cpp and test_metrics.cpp)
METRICS_TEST_SOURCES := $(SRC_DIR)/metrics.cpp $(SRC_DIR)/test_metrics.cpp
METRICS_TEST_OBJECTS := $(METRICS_TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Include paths
ifndef SKIP_LLAMA
    INCLUDES := -I$(INCLUDE_DIR) -I$(LLAMA_INCLUDE)
else
    INCLUDES := -I$(INCLUDE_DIR)
endif
CXXFLAGS += $(INCLUDES)

# ============================================================================
# Build Targets
# ============================================================================

.PHONY: all clean debug release info help test test_metrics install

# Default target
all: release

# Test target (standalone test executable, no llama.cpp required)
test: $(TEST_TARGET)

# Metrics test target (standalone test executable, no llama.cpp required)
test_metrics: $(TEST_METRICS_TARGET)

# Release build (optimized)
release: CXXFLAGS := $(filter-out $(DEBUG_FLAGS),$(CXXFLAGS))
release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

# Debug build (with debug symbols, no optimization)
debug: CXXFLAGS := $(filter-out $(RELEASE_FLAGS),$(CXXFLAGS))
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@
	@echo "Build complete: $@"
	@echo "Library used: $(LLAMA_LIB) ($(LLAMA_LIB_TYPE))"

# Link test executable (no llama.cpp required)
$(TEST_TARGET): $(BUILD_DIR)/test_data.o $(BUILD_DIR)/test_data_test.o | $(BUILD_DIR)
	@echo "Linking test executable $@..."
	$(CXX) $(BUILD_DIR)/test_data.o $(BUILD_DIR)/test_data_test.o -pthread -o $@
	@echo "Test build complete: $@"
	@echo "Run with: $@"

# Link metrics test executable (no llama.cpp required)
$(TEST_METRICS_TARGET): $(METRICS_TEST_OBJECTS) | $(BUILD_DIR)
	@echo "Linking metrics test executable $@..."
	$(CXX) $(METRICS_TEST_OBJECTS) -pthread -o $@
	@echo "Metrics test build complete: $@"
	@echo "Run with: $@"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)/*.o $(TARGET) $(TEST_TARGET) $(TEST_METRICS_TARGET)
	@echo "Clean complete"

# Install target (optional)
install: release
	@echo "Installing benchmark_cpp..."
	@if [ -z "$(PREFIX)" ]; then \
		echo "Error: PREFIX not set. Use: make install PREFIX=/usr/local"; \
		exit 1; \
	fi
	@mkdir -p $(PREFIX)/bin
	@install -m 755 $(TARGET) $(PREFIX)/bin/benchmark_cpp
	@echo "Installed to $(PREFIX)/bin/benchmark_cpp"

# Display configuration
info:
	@echo "=== Build Configuration ==="
	@echo "Target:          $(TARGET)"
	@echo "Compiler:        $(CXX)"
	@echo "C++ Standard:    C++17"
	@echo "Build Type:      $(if $(filter $(DEBUG_FLAGS),$(CXXFLAGS)),Debug,Release)"
	@echo ""
	@echo "=== llama.cpp Configuration ==="
	@echo "llama.cpp Path:  $(LLAMA_DIR)"
	@echo "Include Path:    $(LLAMA_INCLUDE)"
	@echo "Library:         $(LLAMA_LIB)"
	@echo "Library Type:    $(LLAMA_LIB_TYPE)"
	@echo ""
	@echo "=== Compiler Flags ==="
	@echo "CXXFLAGS:        $(CXXFLAGS)"
	@echo "LDFLAGS:         $(LDFLAGS)"
	@echo ""
	@echo "=== Source Files ==="
	@echo "Sources:         $(SOURCES)"
	@echo "Objects:         $(OBJECTS)"

# Help target
help:
	@echo "llama.cpp C++ Benchmark Build System"
	@echo "====================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Build release binary"
	@echo "  release       - Build optimized binary (-O3 -march=native)"
	@echo "  debug         - Build debug binary (-g -O0)"
	@echo "  test          - Build and run test data validation"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install binary (requires PREFIX variable)"
	@echo "  info          - Display build configuration"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  LLAMA_CPP_PATH - Path to llama.cpp installation"
	@echo "                   Default: ../llama.cpp"
	@echo "  PREFIX         - Installation prefix for 'make install'"
	@echo "                   Example: /usr/local or ~/.local"
	@echo ""
	@echo "Examples:"
	@echo "  make release                              # Build optimized binary"
	@echo "  make debug                                # Build debug binary"
	@echo "  LLAMA_CPP_PATH=/opt/llama.cpp make       # Use custom llama.cpp path"
	@echo "  make install PREFIX=~/.local              # Install to ~/.local/bin"
	@echo "  make info                                 # Show current configuration"
	@echo "  make clean && make                        # Clean rebuild"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - llama.cpp must be built first"
	@echo "  - Either libllama.so or libllama.a must exist"
	@echo "  - llama.h header must be accessible"
	@echo ""
	@echo "Building llama.cpp:"
	@echo "  cd ../llama.cpp"
	@echo "  cmake -B build"
	@echo "  cmake --build build -j"
