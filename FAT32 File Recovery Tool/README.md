# FAT32 File Recovery Tool

Have you ever accidentally deleted a file? Do you know that it could be recovered? In this lab, I build a FAT32 file recovery tool called **Need You to Undelete my FILE**, or **nyufile** for short.

## Usage

<pre>

Usage: ./nyufile disk [options]
  -i                     Print the file system information.
  -l                     List the root directory.
  -r filename [-s sha1]  Recover a contiguous file.
  -R filename -s sha1    Recover a possibly non-contiguous file.
</pre>
