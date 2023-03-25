#ifndef _EXECUTE_H_
#define _EXECUTE_H_

#include "global.h"

void execute_by_execvp(char** args, int in_redir_fd, int out_redir_fd, int in_redir, int out_redir, char* command);
void execute(char** args, int in_redir_fd, int out_redir_fd, int in_redir, int out_redir, char* command);

#endif