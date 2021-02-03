CC      = gcc
CXX     = g++
LINK    = g++
CFLAGS  = -g -W -std=c99 -O0
OBJECT  = gemm.o strassen.o loop_divide.o
all:gemm strassen loop_divide

gemm:gemm.o
	$(CC) $(CFLAGS) $^ -o $@

strassen:strassen.o
	$(CC) $(CFLAGS) $^ -o $@

loop_divide:loop_divide.o
	$(CC) $(CFLAGS) $^ -o $@


%*.o:%*.c
	$(CC) $(CFLAGS) -c -$^ -o $@
.PHONY:
	clean
clean:
	rm -rf $(ALL) $(OBJECT)
