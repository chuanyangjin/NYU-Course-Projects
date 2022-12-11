#include "global.h"

char* get_line() {
	char *line = NULL;
  	size_t len = 0;
  	getline(&line, &len, stdin);
	int n = strlen(line);
	line[n-1] = '\0';
	return line;
}

void parse_line(char* line) {
	char** commands = malloc(LINELENGTH * sizeof(char*));
	char* command = malloc(LINELENGTH * sizeof(char*));
	int command_n = 0;
	
	// I/O redirect statuses
	// in_redir = 1  -- "<"
	// in_redir = 2  -- input from pipe
	// out_redir = 1 -- ">"
	// out_redir = 2 -- ">>"
	char* in_redir = malloc(LINELENGTH * sizeof(int));
	char* out_redir = malloc(LINELENGTH * sizeof(int));
	in_redir[0] = 0;
	out_redir[0] = 0;

	// I/O redirect files
	char** in_redir_file = malloc(LINELENGTH * sizeof(char*));
	char** out_redir_file = malloc(LINELENGTH * sizeof(char*));

	// I/O redirect file descriptors
	char* in_redir_fd = malloc(LINELENGTH * sizeof(int));
	char* out_redir_fd = malloc(LINELENGTH * sizeof(int));

	int flag = 0;
	char* token;
	token = strtok(line, " ");
	if (token == NULL) {
		return;
	}

	while (token != NULL) {
		// Prevent the case of cat < a.txt b.txt
		if (flag && (strcmp(token, "|") != 0) && (strcmp(token, "<") != 0) && (strcmp(token, ">") != 0) && (strcmp(token, ">>") != 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
		flag = 0;

		if ((strcmp(token, "|") == 0)) {
			if (strlen(command) == 0){
				fprintf(stderr, "Error: invalid command\n");
				return;
			}
			commands[command_n] = malloc(strlen(command) * sizeof(char));
			strcpy(commands[command_n], command);
			memset(command,0,strlen(command));

			command_n ++;
			in_redir[command_n] = 0;
			out_redir[command_n] = 0;
		}
		else if ((strcmp(token, "<") == 0) || (strcmp(token, ">") == 0) || (strcmp(token, ">>") == 0)) {
			if (strcmp(token, "<") == 0) {
				if (in_redir[command_n]){
					fprintf(stderr, "Error: invalid command\n");
					return;
				}
				else{
					in_redir[command_n] = 1;
					token = strtok(NULL, " \0");
					if (token == NULL){
						fprintf(stderr, "Error: invalid command\n");
						return;
					}
					in_redir_file[command_n] = malloc(strlen(token) * sizeof(char));
					memcpy(in_redir_file[command_n], token, strlen(token));
				}
			}	
			if (strcmp(token, ">") == 0) {
				if (out_redir[command_n]){
					fprintf(stderr, "Error: invalid command\n");
					return;
				}
				else{
					out_redir[command_n] = 1;
					token = strtok(NULL, " \0");
					if (token == NULL){
						fprintf(stderr, "Error: invalid command\n");
						return;
					}
					out_redir_file[command_n] = malloc(strlen(token) * sizeof(char));
					memcpy(out_redir_file[command_n], token, strlen(token));
				}
			}
			if ((strcmp(token, ">>") == 0)) {
				if (out_redir[command_n]){
					fprintf(stderr, "Error: invalid command\n");
					return;
				}
				else{
					out_redir[command_n] = 2;
					token = strtok(NULL, " \0");
					if (token == NULL){
						fprintf(stderr, "Error: invalid command\n");
						return;
					}
					out_redir_file[command_n] = malloc(strlen(token) * sizeof(char));
					memcpy(out_redir_file[command_n], token, strlen(token));
				}
			}

			// Prevent the case of cat < a.txt b.txt
			flag = 1;
		}
		else if ((strcmp(token, "<<") == 0)) {
			fprintf(stderr, "Error: invalid command\n");
			return;
		}
		else {
			if (command[0] != '\0') {
				strcat(command, " ");
			}
			strcat(command, token);
		}

		token = strtok(NULL, " \0");
	}

	// Store the last command
	if (strlen(command) == 0){
		fprintf(stderr, "Error: invalid command\n");
		return;
	}
	commands[command_n] = malloc(strlen(command) * sizeof(char));
	strcpy(commands[command_n], command);
	memset(command, 0, strlen(command));

	// For each command, parse_command() and execute()
	int p_fd[2];
	for (int i = 0; i <= command_n; i++) {
		// Get the fd
		if (in_redir[i] == 1) {
			in_redir_fd[i] = open(in_redir_file[i], O_RDONLY);

			// If the input file does not exist
			if (in_redir_fd[i] == -1) {
				fprintf(stderr, "Error: invalid file\n");
				return;
			}
		}
		if (out_redir[i] == 1) {
			out_redir_fd[i] = open(out_redir_file[i], O_WRONLY | O_CREAT | O_TRUNC, 0644);
		}
		else if (out_redir[i] == 2) {
			out_redir_fd[i] = open(out_redir_file[i], O_WRONLY | O_CREAT | O_APPEND, 0644);
		}

		// Pipe
		if (i != command_n) {
			if ((in_redir[i+1] != 0) || (out_redir[i] != 0)) {
				fprintf(stderr, "Error: invalid command\n");
				return;
			}

			pipe(p_fd);
			in_redir_fd[i+1] = p_fd[0];
			out_redir_fd[i] = p_fd[1];
			in_redir[i+1] = 2;
			out_redir[i] = 1;
		}

		// Print the command information
		// printf("%s\n", commands[i]);
		// printf("in_redir: %d\n", in_redir[i]);
		// printf("out_redir: %d\n", out_redir[i]);
		// printf("in_redir_file: %s\n", in_redir_file[i]);
		// printf("out_redir_file: %s\n", out_redir_file[i]);
		// printf("in_redir_fd: %d\n", in_redir_fd[i]);
		// printf("out_redir_fd: %d\n", out_redir_fd[i]);
		
		char** args = malloc(LINELENGTH * sizeof(char*));
		args = parse_command(commands[i]);
		execute(args, in_redir_fd[i], out_redir_fd[i], in_redir[i], out_redir[i], commands[i]);
	}
}

char** parse_command(char* command) {
	char** args = malloc(LINELENGTH * sizeof(char*));
	int i = 0;
  	char* token;
	
	char* command_copy = malloc(strlen(command) * sizeof(char));
	strcpy(command_copy, command);
  	token = strtok(command_copy, " ");
  	while (token != NULL)
  	{
		int n = strlen(token);
		args[i] = malloc(n * sizeof(char));
		for (int j = 0; j < n; j++) {
            args[i][j] = token[j];
	    }
		// Split by both " " in the middle and "\0" in the end
    	token = strtok(NULL, " \0");    
		i++;
  	}
	args[i] = '\0';
	return args;
}