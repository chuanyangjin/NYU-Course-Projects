# Unix-Shell-in-C
The shell is the main command-line interface between a user and the operating system, and it is an essential part of the daily lives of computer scientists, software engineers, system administrators, and such. In this lab, I build a simplified version of the Unix shell called the **New Yet Usable SHell**, or **NYUSH** for short. It includes the following:
1. Prompt the user for input.   
2. Process creation and termination.
3. I/O redirection and pipe.
4. Handling suspended jobs (jobs and fg).
5. Other built-in commands (cd and exit) and error handling.

## Usage
#### Some examples of valid commands
[nyush lab2]$ cat < input.txt  

[nyush lab2]$ ls -l > output.txt  

[nyush lab2]$ ls -l >> output.txt  

[nyush lab2]$ cat shell.c | wc -l  

[nyush lab2]$ cat shell.c | grep main | less

#### cd <dir>  
[nyush lab2]$ cd /usr/local  

[nyush local]$ cd bin  

[nyush bin]$ ¨€

#### jobs  
[nyush lab2]$ jobs  
[1] ./hello  
[2] /usr/bin/top -c  
[3] cat > output.txt  

[nyush lab2]$ ¨€

#### fg <index>  
[nyush lab2]$ jobs  
[1] ./hello  
[2] /usr/bin/top -c  
[3] cat > output.txt  

[nyush lab2]$ fg 2  

[nyush lab2]$ jobs  
[1] ./hello  
[2] cat > output.txt  
[3] /usr/bin/top -c  

[nyush lab2]$ ¨€

#### exit
[nyush lab2]$ exit