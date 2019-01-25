#include <stdio.h>
#include <stdlib.h>
#include "real.h"

void startComunication(real **x_ptr,
                       real **x_off_ptr,
                       int **recvSendCount,
                       int ***sendColumns,
                       const int *Ngpus
                        )

{
    const int &ngpus = *Ngpus;
    int *indexes = (int *) calloc(ngpus, sizeof(int));
    
    for (int gpuE=0; gpuE<ngpus; ++gpuE ) {
        for (int gpuI=0; gpuI<ngpus; ++gpuI ) {
            for (int i=0;  i<recvSendCount[gpuI][gpuE]; ++i) {
                x_off_ptr[gpuI][ indexes[gpuI]++ ] = x_ptr[gpuE][ sendColumns[gpuE][gpuI][i]  ];
            } // end for //
        } // end for //
    } // end for //
    free(indexes);
} // end of startComunication() //
