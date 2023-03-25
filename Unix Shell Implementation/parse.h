#ifndef _PARSE_H_
#define _PARSE_H_

#include "global.h"

char* get_line();
void parse_line(char* line);
char** parse_command(char* command);

#endif