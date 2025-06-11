# Makefile

CXX = g++
CXXFLAGS = -O3 -std=c++20 -Iinclude -Wall -Wextra -Wpedantic -march=native -flto -fno-exceptions -fno-rtti -DNDEBUG
SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
TARGET = $(BIN_DIR)/blossom

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

all: $(TARGET)

$(TARGET): $(OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
