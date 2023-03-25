#include "global.h"

extern struct Node* processes;

void jobs(char** args) {
	if (args[1] != NULL) {
		fprintf(stderr, "Error: invalid command\n");
		return;
	}

    struct Node *cur = processes;
	int i = 1;
    while (cur != NULL) {
		printf("[%d] %s\n", i, cur->command);
    	cur = cur->next;
		i++;
    }
}

void fg(char** args, char* command) {
	if ((args[1] == NULL) || (args[2] != NULL)) {
		fprintf(stderr, "Error: invalid command\n");
		return;
	}

	int index = atoi(args[1]);
	if (index <= 0) {
		fprintf(stderr, "Error: invalid job\n");
		return;
	}
	
	struct Node *cur = processes;
	if (cur == NULL) {
		fprintf(stderr, "Error: invalid job\n");
		return;
	}

	int i = 1;
	struct Node *prev;
    while (i != index) {
		i++;
		prev = cur;
    	cur = cur->next;
		if (cur == NULL) {
			fprintf(stderr, "Error: invalid job\n");
			return;
		}
    }
	kill(cur->pid, SIGCONT);
	int status;
	waitpid(cur->pid, &status, WUNTRACED);

	if (index == 1){
		processes = processes -> next;
	}
	else {
		if (cur->next == NULL) {
			prev->next = NULL;
		}
		else {
			cur->next = cur->next->next;
		}
	}

	// If the job is suspended again, it would be inserted to the end of the job list
	if (WIFSTOPPED(status)) {
		processes = insert_tail(processes, cur->pid, command);
	}
}

void my_exit(char** args) {
	if (args[1] != NULL) {
		fprintf(stderr, "Error: invalid command\n");
		return;
	}

	if (processes != NULL) {
		fprintf(stderr, "Error: there are suspended jobs\n");
		return;
	}

	exit(1);
}

void cd(char** args) {
	if ((args[1] == NULL) || (args[2] != NULL)) {
		fprintf(stderr, "Error: invalid command\n");
		return;
	}

	char* path = args[1];
	if (chdir(path) < 0) {
		fprintf(stderr, "Error: invalid directory\n");
		return;
	}
}