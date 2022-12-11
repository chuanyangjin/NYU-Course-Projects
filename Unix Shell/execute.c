#include "global.h"

extern struct Node* processes;

void execute_by_execvp(char** args, int in_redir_fd, int out_redir_fd, int in_redir, int out_redir, char* command) {
	pid_t pid = fork();
	if (pid == 0) {
		// Signal handling: set the following signals in child processes to default
		signal(SIGINT, SIG_DFL);
		signal(SIGQUIT, SIG_DFL);
		signal(SIGTSTP, SIG_DFL);

		if (in_redir != 0) {
			dup2(in_redir_fd, 0);
			close(in_redir_fd);
		}
		if (out_redir != 0) {
			dup2(out_redir_fd, 1);
			close(out_redir_fd);
		}

		execvp(args[0], args);
		exit(-1);

		if (in_redir != 0) {
			close(in_redir_fd);
		}
		if (out_redir != 0) {
			close(out_redir_fd);
		}
	}

	if (in_redir != 0) {
		close(in_redir_fd);
	}
	if (out_redir != 0) {
		close(out_redir_fd);
	}

	int status;
	waitpid(pid, &status, WUNTRACED);

	if (WIFSTOPPED(status)) {
    	processes = insert_tail(processes, pid, command);
	}
}

void execute(char** args, int in_redir_fd, int out_redir_fd, int in_redir, int out_redir, char* command) {
	if (!strcmp(args[0], "cd")) {
		if ((in_redir != 0) || (out_redir != 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
		cd(args);
	}
	else if (!strcmp(args[0], "jobs")) {
		if ((in_redir != 0) || (out_redir != 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
        jobs(args);
	}
	else if (!strcmp(args[0], "fg")) {
		if ((in_redir != 0) || (out_redir != 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
        fg(args, command);
	}
	else if (!strcmp(args[0], "exit")) {
		if ((in_redir != 0) || (out_redir != 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
        my_exit(args);
	}
	else {
		execute_by_execvp(args, in_redir_fd, out_redir_fd, in_redir, out_redir, command);
	}
}