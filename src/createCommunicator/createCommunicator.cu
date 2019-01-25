#include <stdio.h>
#include <stdlib.h>
#include "real.h"

#include "parallelSpmv.h"

int createColIdxMap(int **b,  int *a, const int *n);

void createCommunicator(int       *nColsOff,
                        int       ***sendColumns,
                        int       **recvSendCount,
                        int       **col_idx_off,
                        const int *off_node_nnz,                     
                        const int *nRows,
                        const int *Ngpus
                        )                        


{
 
    const int &ngpus = *Ngpus;

    int *firstColumn;
    firstColumn     = (int *) malloc( (ngpus+1) * sizeof(int)); 
    firstColumn[0]=0;


    // creating the firstColumn array
    for (int i=1; i<=ngpus; ++i) {
        firstColumn[i] = firstColumn[i-1] + nRows[i-1];
    } // enf for //
    // end of creating the firstColumn array

    int **off_proc_column_map= (int **) malloc(ngpus * sizeof(int *));
    
    for (int gpu=0; gpu<ngpus; ++gpu) {
        recvSendCount[gpu] = (int *) calloc(ngpus,sizeof(int)); 
        if (*off_node_nnz)  {
            nColsOff[gpu]= createColIdxMap(&off_proc_column_map[gpu],  col_idx_off[gpu], &off_node_nnz[gpu] ); 
        } // end if //
        

        // finding gpu holding each off_proc column
        // and establishing the receive count arrays
        // device: is the gpu
        // recvSendCount[process]: how many to receive from that gpu 
        for (int i=0; i<nColsOff[gpu]; ++i) {
            int device=0;
            while( off_proc_column_map[gpu][i] < firstColumn[device] ||  
                   off_proc_column_map[gpu][i] >= firstColumn[device+1]  ) {
                ++device;
            }// end while //
            ++recvSendCount[gpu][device];
        } // end for
    } // end for //

    int ***reciveColumns = (int ***) malloc(ngpus*sizeof(int **)); 

    for (int gpu=0; gpu<ngpus; ++gpu) {
        // Crerating a 2d-array capable to hold rows of
        // independent size to store the lists of columns 
        // this rank need to receive.
        reciveColumns[gpu] = (int **) malloc(ngpus*sizeof(int *));
        for (int process=0; process<ngpus; ++process){
            reciveColumns[gpu][process] = (int *) malloc( recvSendCount[gpu][process]*sizeof(int ));
        } // end for //
        
        // filling the reciveColumns arrays
        for (int process=0, k=0; process<ngpus; ++process){
            for (int i=0; i < recvSendCount[gpu][process]; ++i, ++k){
                reciveColumns[gpu][process][i] = off_proc_column_map[gpu][k];
            } // end for //
        } // end for //

        // Crerating a 2d-array capable to hold rows of
        // independent size to store the lists of columns 
        // this rank need to send.
        sendColumns[gpu]   = (int **) malloc(ngpus*sizeof(int *));
        
        for (int process=0; process<ngpus; ++process) {
            //sendColumns[gpu][process] = (int *) calloc( sendCount[gpu][process], sizeof(int));
            sendColumns[gpu][process] = (int *) calloc( recvSendCount[process][gpu], sizeof(int));
        } // end for //

        // end of establishing the send/receive count arrays

    } // end for //

    for (int gpu=0; gpu<ngpus; ++gpu) {    
        for (int process=0; process<ngpus; ++process){
            for (int i=0; i < recvSendCount[gpu][process]; ++i) {
                sendColumns[process][gpu][i] = reciveColumns[gpu][process][i] - firstColumn[process];
            } // end for //
        } // end for //    
    } // end for //

    for (int proc=0; proc<ngpus; ++proc){
        for (int gpu=0; gpu<ngpus; ++gpu){
            free(reciveColumns[proc][gpu]);
        } // end for /
        free(reciveColumns[proc]);
    } // end for /
    free(reciveColumns);

    for (int gpu=0; gpu<ngpus; ++gpu){
        free(off_proc_column_map[gpu]);
    } // end for /
    free(off_proc_column_map);
    free(firstColumn);
} // end of createCommunicator() //
