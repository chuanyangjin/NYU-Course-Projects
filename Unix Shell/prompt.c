#include "global.h"

void print_prompt() {
	char* path;
	char* buf = malloc(LINELENGTH*sizeof(char));
	path = getcwd(buf, LINELENGTH);
	char* base_name;
	base_name = basename(path);
	printf("[nyush %s]$ ", base_name);
	fflush(stdout);
}