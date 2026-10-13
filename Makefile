# Makefile for the image search assignment
#
#   make            build the main program (needs src/main.cpp)
#   make test       build and run every tests/test_*.cpp with sanitizers
#   make debug      build the main program with -g and sanitizers
#   make clean      remove all build output
#
# Layout: include/ (headers), src/ (sources), tests/ (test_*.cpp, each with its own main)

CXX      ?= g++
TARGET   ?= search
BUILD    ?= build
MODE     ?= -O2

# OpenCV is only needed by the feature extraction code; these are empty if it is not installed.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null)

CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic $(MODE) -Iinclude $(OPENCV_CFLAGS) -MMD -MP
LDLIBS   = $(OPENCV_LIBS)

SRCS      := $(wildcard src/*.cpp)
OBJS      := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRCS))
LIB_OBJS  := $(filter-out $(BUILD)/main.o,$(OBJS))
TESTS     := $(wildcard tests/test_*.cpp)
TEST_BINS := $(patsubst tests/%.cpp,$(BUILD)/tests/%,$(TESTS))

SANITIZE  := -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer

.PHONY: all test run-tests debug clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Each test links against every source file except main.cpp.
$(BUILD)/tests/%: tests/%.cpp $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJS) -o $@ $(LDLIBS)

# Tests and the debug build use separate build directories, so they never mix
# with the optimized objects.
test:
	$(MAKE) BUILD=build/test MODE="$(SANITIZE)" run-tests

run-tests: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t || exit 1; done
	@echo "All tests passed."

debug:
	$(MAKE) BUILD=build/debug TARGET=search_debug MODE="$(SANITIZE)" search_debug

clean:
	rm -rf build $(TARGET) search_debug

-include $(OBJS:.o=.d)