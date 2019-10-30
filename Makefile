# # MAKEFILE # #

DB = -Ofast
FLAGS = -ansi -std=c99 -Wall $(DB)
CC = gcc $(FLAGS)
CO = $(CC) -c

nettest: net.c net.h matmath.o
	$(CC) -DUNITTEST net.c matmath.o -o nettest

matmath.o: matmath.c matmath.h
	$(CO) matmath.c -o matmath.o

mattest: matmath.c matmath.h
	$(CC) -DUNITTEST matmath.c -o mattest

clean:
	rm -rf mattest*
	rm -rf nettest*
	rm -f  *.o
