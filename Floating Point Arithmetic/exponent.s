	.text
	.globl	exponent
	.def	exponent;	.scl	2;	.type	32;	.endef


	# This function takes three parameters:
	#   -- a 32-bit integer x in the %ecx register
	#   -- a 32-bit integer y in the %edx register
	#   -- a 32-bit integer n in the %r8d register		
	# It returns x^n + y^n, that is, x to the nth power plus y to the nth power.	
	# The return value, being 32 bits, must be placed in the %eax register.
	
exponent:	

	# Note: You can overwrite the 64-bit registers %rax, %rcx, %rdx, %r8,
	# %r9, %r10, %r11 (as well as their 32-bit halves, %eax, %ecx,
	# %edx, %r8d, %r9d, %r10d, %r11d) as you like. These are
	# "caller-saved" registers.

	pushq	%rbp	     # LEAVE THIS ALONE
	movq	%rsp, %rbp   # LEAVE THIS ALONE

	mov		$1, %eax	 # use the 32-bit register %eax to hold x^n, initially 1
	                     # 
	
    mov		$1, %ebx   # use a 32-bit register to hold y^n, initially 1
	mov		$0, %r9d   # use a 32-bit register to hold i, initially 0
LOOP_TOP:
        cmp		%r8d, %r9d  # compare i to n (remember, the comparison appears reversed)
        jge		DONE   # and if i is greater or equal to n, jump to DONE

        imul	%ecx, %eax   # multiply register holding x^n by x
        imul	%edx, %ebx   # multiply register holding y^n by y 
        inc		%r9d   # i++
        jmp		LOOP_TOP   # jump to LOOP_TOP
DONE:
        add		%ebx, %eax   # add y^n to x^n
        		     # Since the result is already in %eax, just leave it there

	popq	%rbp         # LEAVE THIS ALONE
	retq		     # LEAVE THIS ALONE
	

