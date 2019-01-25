#include <stdio.h>
#include "real.h"

void vectorReader( real *v, const int *GPU, const int *n, const char *vectorFile)
{
    const int &gpu = *GPU;

    int acumulate=0;
    for (int i=0; i<gpu; ++i) {
        acumulate+=n[i];
    } // end for //
    const size_t offset = (acumulate )* sizeof(real) ;

    // opening vector file to read values
    FILE *filePtr;
    filePtr = fopen(vectorFile, "rb");
    // reading cols vector (n) values //
    fseek(filePtr, offset, SEEK_SET);
    if ( !fread(v, sizeof(real), (size_t) n[gpu], filePtr)) exit(0);
    fclose(filePtr);
    // end of opening vector file to read values
} // end of vectoReader //
