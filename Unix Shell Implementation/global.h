#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>

#define LINELENGTH 1000

#include "suspended.h"
#include "prompt.h"
#include "parse.h"
#include "execute.h"
#include "commands.h"

#endif