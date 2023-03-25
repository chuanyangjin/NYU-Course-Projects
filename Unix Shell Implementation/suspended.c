#include "global.h"

// The head of the linked list that records the suspended processes
struct Node* processes = NULL;

struct Node* insert_tail(struct Node* head, int pid, char* command) {
    struct Node* new = (struct Node*)malloc(sizeof(struct Node));
    new->pid = pid;
    new->command = command;

    if (head == NULL) {
        return new;
    }

    struct Node* cur = head;
    while (cur->next != NULL) {
        cur = cur->next;
    }
    cur->next = new;
    return head;
}