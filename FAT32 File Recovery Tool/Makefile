CC=gcc
CFLAGS=-g -pedantic -std=gnu11 -Wall -Wextra
LDLIBS = -lcrypto

.PHONY: all
all: nyufile

nyufile: nyufile.o

nyufile.o: nyufile.c

.PHONY: clean
clean:
	rm -f *.o nyufile