#include <stdio.h>
#include <stdlib.h>
#include "dataDef.h"

#include "real.h"

void getRowsNnzPerProc(int *rowsPerGpu,int *nnzPP,  const int *global_n, const int *global_nnz, const int *row_Ptr, const int ngpus);

void reader( int **rowsPerGpu,
             int *on_proc_nnz, 
             int *off_proc_nnz, 
             int ***rowPtr, int ***colIdx, real ***val,
             int ***rowPtrO, int ***colIdxO, real ***valO,
             const char *matrixFile,
             const int ngpus)
{
    int n_global, nnz_global;
    FILE *filePtr;
    
    int *nnzPP = (int *) malloc(ngpus * sizeof(int)); 
    
    filePtr = fopen(matrixFile, "rb");
    
    // reading global nun rows //
    if ( !fread(&n_global, sizeof(int), 1, filePtr) ) exit(0); 

    // reading global nnz //
    if ( !fread(&nnz_global, sizeof(int), (size_t) 1, filePtr)) exit(0);

    int *rows_Ptr = (int *) malloc((n_global+1)*sizeof(int));
    // reading rows vector (n+1) values //
    if ( !fread(rows_Ptr, sizeof(int), (size_t) (n_global+1), filePtr)) exit(0);
    fclose(filePtr);
    
    getRowsNnzPerProc((*rowsPerGpu), nnzPP,&n_global,&nnz_global, rows_Ptr,ngpus);

    int *firstColumn = (int *) malloc( (ngpus+1) * sizeof(int));
    int *fColumn     = (int *) malloc( (ngpus)   * sizeof(int)); 
    int *lColumn     = (int *) malloc( (ngpus)   * sizeof(int)); 
    firstColumn[0] = 0;
    
    // forming on-proc columns per proc
    for (int gpu=0; gpu<ngpus; ++gpu) {
        firstColumn[gpu+1] = firstColumn[gpu] + (*rowsPerGpu)[gpu];
        fColumn[gpu] = firstColumn[gpu];
        lColumn[gpu] = firstColumn[gpu+1]-1;
    } // end for //

    free(rows_Ptr);
    free(firstColumn);

    int *offsetR = (int *) malloc(ngpus * sizeof(int));  
    int *offsetC = (int *) malloc(ngpus * sizeof(int)); 
    offsetR[0]=offsetC[0]=0;
    
    
    for (int gpu=1; gpu<ngpus ; ++gpu) {
        offsetR[gpu] = offsetR[gpu-1] +  (*rowsPerGpu)[gpu-1];   
        offsetC[gpu] = offsetC[gpu-1] +   nnzPP[gpu-1];   
    } // end for //


    size_t offset;
    for (int gpu=0; gpu<ngpus; ++gpu) {
        offset=(3 + n_global + offsetC[gpu])*sizeof(int);

        // each gpu read the columns associated with their non-zeros
        int *cols_Ptr = (int *) malloc(nnzPP[gpu]*sizeof(int));
        
        // opening file to read column information for this process
        filePtr = fopen(matrixFile, "rb");
        // reading cols vector (nnz) values //
        fseek(filePtr, offset, SEEK_SET);
        if ( !fread(cols_Ptr, sizeof(int), (size_t) nnzPP[gpu], filePtr)) exit(0);
        // end of opening file to read column information for this process

        // determining on_proc_nnz and of_proc_nnz for this process
        //int on_proc_nnz=0;
        off_proc_nnz[gpu]=0;
        for (int i=0; i<nnzPP[gpu]; ++i) {
            if (cols_Ptr[i] >= fColumn[gpu]  &&  cols_Ptr[i] <= lColumn[gpu]  ) {
                ++on_proc_nnz[gpu];
            } else {
                ++off_proc_nnz[gpu];
            } // end if 
        } // end for //
        // end of determining on_proc_nnz and of_proc_nnz for each GPU

        // allocating for on-proc solution //
        (*rowPtr)[gpu] = (int *)  malloc( ((*rowsPerGpu)[gpu]+1) * sizeof(int));
        (*rowPtr)[gpu][0] = 0;
        (*colIdx)[gpu] = (int *)  malloc( on_proc_nnz[gpu] * sizeof(int)); 
        (*val)[gpu]    = (real *) malloc( on_proc_nnz[gpu] * sizeof(real)); 

        // allocating for off-proc solution if needed //
        if ( off_proc_nnz[gpu] ) {
            (*rowPtrO)[gpu] = (int *)  malloc(((*rowsPerGpu)[gpu]+1) * sizeof(int)); 
            (*rowPtrO)[gpu][0] = 0;
            (*colIdxO)[gpu] = (int *)  malloc(off_proc_nnz[gpu] * sizeof(int)); 
            (*valO)[gpu]    = (real *) malloc(off_proc_nnz[gpu] * sizeof(real)); 
        } // end if //


        // each process read the rows pointers
        offset=(2 + offsetR[gpu])*sizeof(int);
        
        int nowsP1=(*rowsPerGpu)[gpu]+1;
         
        int *rows_Ptr = (int *) malloc(nowsP1*sizeof(int));

        fseek(filePtr, offset, SEEK_SET);
        if ( !fread(rows_Ptr, sizeof(int), (size_t) nowsP1, filePtr)) exit(0);
        // read the rows pointers fot each gpu 

        // read the vals one by one for each gpu
        offset=(3 + n_global + nnz_global  ) * sizeof(int) + offsetC[gpu] * sizeof(double);
        fseek(filePtr, offset, SEEK_SET);
    
        for (int i=1,k=0,on=0,off=0; i<= (*rowsPerGpu)[gpu]; ++i) {
            int nnzPR = rows_Ptr[i] - rows_Ptr[i-1];
            int rowCounterOn=0;
            int rowCounterOff=0;
            double temp;
            for (int j=0; j<nnzPR; ++j, ++k ) {
                if ( !fread(&temp, sizeof(double), (size_t) (1), filePtr)) exit(0);
                if (cols_Ptr[k] >=  fColumn[gpu]  &&  cols_Ptr[k] <=  lColumn[gpu]  ) {
                    // on process data goes here
                    ++rowCounterOn;
                    (*colIdx)[gpu][on] = cols_Ptr[k] - fColumn[gpu];
                    (*val)[gpu][on] = (real) temp;
                    ++on;
                } else {
                    // off process data goes here
                    ++rowCounterOff;
                    (*colIdxO)[gpu][off] = cols_Ptr[k];
                    (*valO)[gpu][off] = (real) temp;
                    ++off;
                } // end if 
            } // end for //
            (*rowPtr)[gpu][i]  = (*rowPtr)[gpu][i-1]  + rowCounterOn;
            if (off_proc_nnz[gpu]) (*rowPtrO)[gpu][i] = (*rowPtrO)[gpu][i-1] + rowCounterOff;
        } // end for //
        free(rows_Ptr);
        free(cols_Ptr);
    } // end for //

    free(offsetC);
    free(offsetR);
    free(nnzPP);    
    free(lColumn);  
    free(fColumn);  
    fclose(filePtr);
    //printf("file: %s, line: %d,  off_proc_nnz: %d\n", __FILE__, __LINE__, (*off_proc_nnz)[0] ); exit(0);
} // end of reader //
