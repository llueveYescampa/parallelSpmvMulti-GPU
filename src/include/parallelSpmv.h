void reader(int **n,  int *on_proc_nnz, int *off_proc_nnz, 
            int ***rPtr,int ***cIdx,real ***v,int ***rPtrO,int ***cIdxO,real ***vO,
            const char *matrixFile, const int ngpus);

void vectorReader(real *v, const int *ngpus, const int *n, const char *vectorFile);


void createCommunicator( int *nColsOff,
                         int ***sendColumns,
                         int **recvSendCount,
                         int **col_idx_off,
                         const int *off_node_nnz,
                         const int *n,
                         const int *ngpus
                         );                        

void startComunication(real **x_ptr,
                       real **x_off_ptr,
                       int **recvSendCount,
                       int ***sendColumns,
                       const int *ngpus
                        );

__global__ 
void spmv(real *__restrict__ y, 
         //real *__restrict__ x, 
         //real *__restrict__ val,  
         int  *__restrict__ row_ptr, 
         int  *__restrict__ col_idx, 
         const int nRows
         );

//void spmv(real *b, real *__restrict__ val, real *x, int *row_ptr, int *col_idx, int nRows);

