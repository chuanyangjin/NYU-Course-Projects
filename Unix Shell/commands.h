#ifndef _COMMANDS_H_
#define _COMMANDS_H_

#include "global.h"

void cd(char** args);
void jobs(char** args);
void fg(char** args, char* command);
void my_exit(char** args);

#endif