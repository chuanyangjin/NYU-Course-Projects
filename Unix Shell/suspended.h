#ifndef _SUSPENDED_H_
#define _SUSPENDED_H_

#include "global.h"

struct Node {
    int pid;
	char* command;
    struct Node* next;
};

struct Node* insert_tail(struct Node* head, int pid, char* command);

#endif