// New Yet Usable SHell
// The shell is the main command-line interface between a user and the operating system, and it is an essential part 
// of the daily lives of computer scientists, software engineers, system administrators, and such.
// In this lab, I build a simplified version of the Unix shell called the New Yet Usable SHell, or nyush for short.
// It includes the following:
//    Prompt the user for input.
//    Process creation and termination.
//    I/O redirection and pipe.
//    Handling suspended jobs (jobs and fg).
//    Other built-in commands (cd and exit) and error handling.

#include "global.h"

int main() {
	// Signal handling: ignore the following signals in shell
	signal(SIGINT, SIG_IGN);
	signal(SIGQUIT, SIG_IGN);
	signal(SIGTSTP, SIG_IGN);

	while (1) {
		// Print the prompt for the next command
		print_prompt();

		// Read in the command line
		char* line = malloc(LINELENGTH * sizeof(char));
		line = get_line();
		if (feof(stdin)) {
			break;
		}

		// Parse and execute the command
		parse_line(line);
	}
}