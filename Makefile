# Makefile for the CUDA PIC Simulation

NVCC    := nvcc
TARGET  := pic
SRC     := main.cu
CFLAGS  := -O3

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
