CC = gcc
CFLAGS =

all:	assign2

exponent.o:	exponent.s
	$(CC) $(CFLAGS) -c exponent.s

assign2:	main.o functions.o exponent.o
	$(CC) $(CFLAGS) -o assign2 main.o functions.o exponent.o

