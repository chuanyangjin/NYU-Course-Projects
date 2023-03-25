CC=gcc
CFLAGS=-g -pedantic -std=gnu11 -Wall -Wextra

.PHONY: all
all: nyush

nyush: nyush.o

nyush.o: nyush.c global.h suspended.h suspended.c prompt.h prompt.c parse.h parse.c execute.h execute.c commands.h commands.c

.PHONY: clean
clean:
	rm -f *.o nyush