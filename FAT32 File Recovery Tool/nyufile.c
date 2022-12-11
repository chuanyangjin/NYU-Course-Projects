#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <openssl/sha.h>

#define SHA_DIGEST_LENGTH 20

// Boot sector
#pragma pack(push,1)
typedef struct BootEntry {
    unsigned char  BS_jmpBoot[3];     // Assembly instruction to jump to boot code
    unsigned char  BS_OEMName[8];     // OEM Name in ASCII
    unsigned short BPB_BytsPerSec;    // Bytes per sector. Allowed values include 512, 1024, 2048, and 4096
    unsigned char  BPB_SecPerClus;    // Sectors per cluster (data unit). Allowed values are powers of 2, but the cluster size must be 32KB or smaller
    unsigned short BPB_RsvdSecCnt;    // Size in sectors of the reserved area
    unsigned char  BPB_NumFATs;       // Number of FATs
    unsigned short BPB_RootEntCnt;    // Maximum number of files in the root directory for FAT12 and FAT16. This is 0 for FAT32
    unsigned short BPB_TotSec16;      // 16-bit value of number of sectors in file system
    unsigned char  BPB_Media;         // Media type
    unsigned short BPB_FATSz16;       // 16-bit size in sectors of each FAT for FAT12 and FAT16. For FAT32, this field is 0
    unsigned short BPB_SecPerTrk;     // Sectors per track of storage device
    unsigned short BPB_NumHeads;      // Number of heads in storage device
    unsigned int   BPB_HiddSec;       // Number of sectors before the start of partition
    unsigned int   BPB_TotSec32;      // 32-bit value of number of sectors in file system. Either this value or the 16-bit value above must be 0
    unsigned int   BPB_FATSz32;       // 32-bit size in sectors of one FAT
    unsigned short BPB_ExtFlags;      // A flag for FAT
    unsigned short BPB_FSVer;         // The major and minor version number
    unsigned int   BPB_RootClus;      // Cluster where the root directory can be found
    unsigned short BPB_FSInfo;        // Sector where FSINFO structure can be found
    unsigned short BPB_BkBootSec;     // Sector where backup copy of boot sector is located
    unsigned char  BPB_Reserved[12];  // Reserved
    unsigned char  BS_DrvNum;         // BIOS INT13h drive number
    unsigned char  BS_Reserved1;      // Not used
    unsigned char  BS_BootSig;        // Extended boot signature to identify if the next three values are valid
    unsigned int   BS_VolID;          // Volume serial number
    unsigned char  BS_VolLab[11];     // Volume label in ASCII. User defines when creating the file system
    unsigned char  BS_FilSysType[8];  // File system type label in ASCII
} BootEntry;
#pragma pack(pop)

// Directory entry
#pragma pack(push,1)
typedef struct DirEntry {
    unsigned char  DIR_Name[11];      // File name
    unsigned char  DIR_Attr;          // File attributes
    unsigned char  DIR_NTRes;         // Reserved
    unsigned char  DIR_CrtTimeTenth;  // Created time (tenths of second)
    unsigned short DIR_CrtTime;       // Created time (hours, minutes, seconds)
    unsigned short DIR_CrtDate;       // Created day
    unsigned short DIR_LstAccDate;    // Accessed day
    unsigned short DIR_FstClusHI;     // High 2 bytes of the first cluster address
    unsigned short DIR_WrtTime;       // Written time (hours, minutes, seconds
    unsigned short DIR_WrtDate;       // Written day
    unsigned short DIR_FstClusLO;     // Low 2 bytes of the first cluster address
    unsigned int   DIR_FileSize;      // File size in bytes. (0 for directories)
} DirEntry;
#pragma pack(pop)

char* get_dir_name(unsigned char DIR_Name[11]) {
    static unsigned char dir_name[13];
    int j = 0;

    // File name
    for (int i = 0; i < 8; i++) {
        if (DIR_Name[i] != ' ') {
            dir_name[j] = DIR_Name[i];
            j++;
        }
    }

    // Dot seperator
    if (DIR_Name[8] != ' ') {
        dir_name[j] = '.';
        j++;
    }

    // File extension
    for (int i = 8; i < 11; i++) {
        if (DIR_Name[i] != ' ') {
            dir_name[j] = DIR_Name[i];
            j++;
        }
    }
    dir_name[j] = '\0';

    return dir_name;
}

void print_file_system_information(BootEntry* boot) {
    printf("Number of FATs = %d\n", boot->BPB_NumFATs);
    printf("Number of bytes per sector = %d\n", boot->BPB_BytsPerSec);
    printf("Number of sectors per cluster = %d\n", boot->BPB_SecPerClus);
    printf("Number of reserved sectors = %d\n", boot->BPB_RsvdSecCnt);
}

void list_root_directory(BootEntry* boot, unsigned char* mapped_address) {
    int cluster = boot->BPB_RootClus;
    int total_entry_per_cluster = boot->BPB_BytsPerSec * boot->BPB_SecPerClus / 32;
    int total_entry = 0;

    // Loop through clusters of the root directory
    while (cluster < 0x0ffffff8) {
        // Reserved area + FAT area + data area before root directory (starting with cluster 2)
        int root = boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + boot->BPB_NumFATs * boot->BPB_FATSz32 * boot->BPB_BytsPerSec + (cluster - 2) * boot->BPB_SecPerClus * boot->BPB_BytsPerSec;

        // Loop through entries of the cluster
        for (int i = 0; i < total_entry_per_cluster; i++) {
            DirEntry* entry = (DirEntry*)(mapped_address + root + 32 * i);

            if ((entry->DIR_Name[0] != 0x00) & (entry->DIR_Name[0] != 0xe5)) {
                // Not empty entry or deleted entry
                unsigned char* dir_name = get_dir_name(entry->DIR_Name);
                printf("%s", dir_name);

                // If the entry is a directory, append a / indicator
                if (((entry->DIR_Attr >> 4) & 0x1) == 1) {
                    printf("/");
                }

                printf(" (size = %d, starting cluster = %d)\n", entry->DIR_FileSize, (entry->DIR_FstClusHI << 16) + entry->DIR_FstClusLO);

                total_entry++;
            }
        }

        cluster = *((int*)(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster));
    }

    printf("Total number of entries = %d\n", total_entry);
}

bool check_sha(BootEntry* boot, DirEntry* entry, unsigned char* mapped_address, unsigned char* sha1_provided) {
    // Locate the file content of the deleted file
    int cluster_to_recover = (entry->DIR_FstClusHI << 16) + entry->DIR_FstClusLO;
    unsigned char* file_content = (unsigned char*)(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + boot->BPB_NumFATs * boot->BPB_FATSz32 * boot->BPB_BytsPerSec + (cluster_to_recover - 2) * boot->BPB_SecPerClus * boot->BPB_BytsPerSec);

    // Compute the SHA-1 hash of file content and stores the result in sha1_20
    unsigned char sha1_20[SHA_DIGEST_LENGTH];
    SHA1(file_content, entry->DIR_FileSize, sha1_20);

    // Store the right format of sha1_20 in sha1_40
    unsigned char sha1_40[2 * SHA_DIGEST_LENGTH];
    for (int j = 0; j < SHA_DIGEST_LENGTH; j++) {
        sprintf(sha1_40 + 2 * j, "%02x", sha1_20[j]);
    }

    return (strcmp(sha1_provided, sha1_40) == 0);
}

void recover_FAT(BootEntry* boot, DirEntry* entry, unsigned char* mapped_address) {
    // Locate the related cluster entries in FAT
    int cluster_to_recover = (entry->DIR_FstClusHI << 16) + entry->DIR_FstClusLO;

    if (entry->DIR_FileSize != 0) {
        // Determine the number of cluster entries to recover
        unsigned int num_cluster_to_recover;
        if ((entry->DIR_FileSize) % (boot->BPB_SecPerClus * boot->BPB_BytsPerSec)) {
            num_cluster_to_recover = entry->DIR_FileSize / (boot->BPB_SecPerClus * boot->BPB_BytsPerSec) + 1;
        }
        else {
            num_cluster_to_recover = entry->DIR_FileSize / (boot->BPB_SecPerClus * boot->BPB_BytsPerSec);
        }

        // Restore the zero into the next cluster addresses
        int next_cluster;
        for (int i = 0; i < num_cluster_to_recover; i++) {
            if (i == num_cluster_to_recover - 1) {
                next_cluster = 0x0ffffff8;
            }
            else {
                next_cluster = cluster_to_recover + 1;
            }

            *(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster_to_recover) = next_cluster;
            *(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster_to_recover + 1) = next_cluster >> 8;
            *(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster_to_recover + 2) = next_cluster >> 16;
            *(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster_to_recover + 3) = next_cluster >> 24;
            cluster_to_recover++;
        }
    }
}

void recover_contiguous_file(int sflag, BootEntry* boot, unsigned char* mapped_address, unsigned char* filename, unsigned char* sha1_provided) {
    int cluster = boot->BPB_RootClus;
    int total_entry_per_cluster = boot->BPB_BytsPerSec * boot->BPB_SecPerClus / 32;
    int match = 0;
    unsigned char* entry_address_to_recover;

    // Loop through clusters of the root directory
    while (cluster < 0x0ffffff8) {
        // Reserved area + FAT area + data area before root directory (starting with cluster 2)
        int root = boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + boot->BPB_NumFATs * boot->BPB_FATSz32 * boot->BPB_BytsPerSec + (cluster - 2) * boot->BPB_SecPerClus * boot->BPB_BytsPerSec;

        // Loop through entries of the cluster
        for (int i = 0; i < total_entry_per_cluster; i++) {
            DirEntry* entry = (DirEntry*)(mapped_address + root + 32 * i);

            if (entry->DIR_Name[0] == 0xe5) {
                // Deleted entry
                unsigned char* dir_name = get_dir_name(entry->DIR_Name);

                // Compare the filenames
                if (strcmp(filename + 1, dir_name + 1) == 0) {
                    if (sflag == 0) {
                        // Recover without SHA-1 hash
                        match++;
                        entry_address_to_recover = mapped_address + root + 32 * i;
                    }
                    else {
                        // Recover with SHA-1 hash
                        if (check_sha(boot, entry, mapped_address, sha1_provided)) {
                            match++;
                            entry_address_to_recover = mapped_address + root + 32 * i;
                        }
                    }
                }
            }
        }

        cluster = *((int*)(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster));
    }

    if (match == 0) {
        printf("%s: file not found\n", filename);
    }
    else if (match > 1) {
        printf("%s: multiple candidates found\n", filename);
    }
    else {
        // Recover is successful when there is a unique match
        if (sflag == 0) {
            printf("%s: successfully recovered\n", filename);
        }
        else {
            printf("%s: successfully recovered with SHA-1\n", filename);
        }

        // Obtain the entry to recover
        DirEntry* entry = (DirEntry*)(entry_address_to_recover);

        // Recover the filename in root directory
        entry->DIR_Name[0] = filename[0];

        // Recover the related FAT entries
        recover_FAT(boot, entry, mapped_address);
    }
}

void recover_noncontiguous_file(BootEntry* boot, unsigned char* mapped_address, unsigned char* filename, unsigned char* sha1_provided) {
    int cluster = boot->BPB_RootClus;
    int total_entry_per_cluster = boot->BPB_BytsPerSec * boot->BPB_SecPerClus / 32;
    int match = 0;
    unsigned char* entry_address_to_recover;

    // Loop through clusters of the root directory
    while (cluster < 0x0ffffff8) {
        // Reserved area + FAT area + data area before root directory (starting with cluster 2)
        int root = boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + boot->BPB_NumFATs * boot->BPB_FATSz32 * boot->BPB_BytsPerSec + (cluster - 2) * boot->BPB_SecPerClus * boot->BPB_BytsPerSec;

        // Loop through entries of the cluster
        for (int i = 0; i < total_entry_per_cluster; i++) {
            DirEntry* entry = (DirEntry*)(mapped_address + root + 32 * i);

            if (entry->DIR_Name[0] == 0xe5) {
                // Deleted entry
                unsigned char* dir_name = get_dir_name(entry->DIR_Name);

                // Compare the filenames
                if (strcmp(filename + 1, dir_name + 1) == 0) {
                    if (check_sha(boot, entry, mapped_address, sha1_provided)) {
                        match++;
                        entry_address_to_recover = mapped_address + root + 32 * i;
                    }
                }
            }
        }

        cluster = *((int*)(mapped_address + boot->BPB_RsvdSecCnt * boot->BPB_BytsPerSec + 4 * cluster));
    }

    if (match == 0) {
        printf("%s: file not found\n", filename);
    }
    else if (match > 1) {
        printf("%s: multiple candidates found\n", filename);
    }
    else {
        // Recover is successful when there is a unique match
        if (sflag == 0) {
            printf("%s: successfully recovered\n", filename);
        }
        else {
            printf("%s: successfully recovered with SHA-1\n", filename);
        }

        // Obtain the entry to recover
        DirEntry* entry = (DirEntry*)(entry_address_to_recover);

        // Recover the filename in root directory
        entry->DIR_Name[0] = filename[0];

        // Recover the related FAT entries
        recover_FAT(boot, entry, mapped_address);
    }
}

int main(int argc, char* argv[]) {
    // Validate usage and parse arguments
    char usage_info[300] = "Usage: ./nyufile disk <options>\n\
  -i                     Print the file system information.\n\
  -l                     List the root directory.\n\
  -r filename [-s sha1]  Recover a contiguous file.\n\
  -R filename -s sha1    Recover a possibly non-contiguous file.\n";
    if (argc == 1) {
        printf("%s", usage_info);
        return 0;
    }
    else if (*(argv[1]) == '-') {
        printf("%s", usage_info);
        return 0;
    }
    char* disk = argv[1];

    int ch;
    int iflag = 0;
    int lflag = 0;
    int rflag = 0;
    int Rflag = 0;
    int sflag = 0;
    unsigned char* filename;
    unsigned char* sha1_provided;
    while ((ch = getopt(argc, argv, ":ilr:R:s:")) != -1) {
        switch (ch) {
        case 'i': 
            iflag++;  
            break;
        case 'l': 
            lflag++; 
            break;
        case 'r': 
            rflag++; 
            filename = (unsigned char*)optarg; 
            break;
        case 'R': 
            Rflag++; 
            filename = (unsigned char*)optarg;
            break;
        case 's': 
            if ((rflag == 0) && (Rflag == 0)) {
                printf("%s", usage_info);
                return 0;
            }
            sflag++; 
            sha1_provided = (unsigned char*)optarg;
            break;
        case ':':
            printf("%s", usage_info);
            return 0;
        case '?':
            printf("%s", usage_info);
            return 0;
        }
    }
    if (!(((iflag == 1) && (lflag == 0) && (rflag == 0) && (Rflag == 0) && (sflag == 0)) 
        || ((iflag == 0) && (lflag == 1) && (rflag == 0) && (Rflag == 0) && (sflag == 0))
        || ((iflag == 0) && (lflag == 0) && (rflag == 1) && (Rflag == 0))
        || ((iflag == 0) && (lflag == 0) && (rflag == 0) && (Rflag == 1) && (sflag == 1)))) {
        printf("%s", usage_info);
        return 0;
    }

    // Open the disk image and map its address
    struct stat sb;
    stat(disk, &sb);
    int fd = open(disk, O_RDWR);
    unsigned char* mapped_address = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    BootEntry* boot = (BootEntry*)(mapped_address);

    // Print the file system information
    if (iflag == 1) {
        print_file_system_information(boot);
    }

    // List the root directory
    if (lflag == 1) {
        list_root_directory(boot, mapped_address);
    }

    // Recover a contiguous file
    if (rflag == 1) {
        recover_contiguous_file(sflag, boot, mapped_address, filename, sha1_provided);
    }

    // Recover a possibly non-contiguous file
    if (Rflag == 1) {
        recover_noncontiguous_file(boot, mapped_address, filename, sha1_provided);
    }

    return 0;
}