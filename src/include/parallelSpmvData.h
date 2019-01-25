    int  ngpus=0;
    int *n=NULL;
    int *on_proc_nnz=NULL;
    int *off_proc_nnz=NULL;

    // data for the on_proc solution: host an device
    int  **row_ptr=NULL;
    int  **col_idx=NULL;
    real **val=NULL;
    
    int  **row_ptr_d=NULL;
    int  **col_idx_d=NULL;
    real **val_d=NULL;
    // end of data for the on_proc solution
    
    // data for the off_proc solution: host an device
    int  **row_ptr_off=NULL;   // partially allocated inside reader.cu 
    int  **col_idx_off=NULL;   // partially allocated inside reader.cu 
    real **val_off=NULL;       // partially allocated inside reader.cu 
    
    int  **row_ptr_off_d=NULL;
    int  **col_idx_off_d=NULL;
    real **val_off_d=NULL;
    // end of data for the off_proc solution
    
    real **w  =NULL;
    real **w_d=NULL;
    
    real **v  =NULL;
    real **v_d=NULL;

    real **v_off  =NULL;
    real **v_off_d=NULL;
    

    // creatinng communicator data//
    int *nColsOff=NULL;
    int **recvSendCount=NULL;  // partially allocated inside createCommunicator.cu 
    int ***sendColumns=NULL;   // partially allocated inside createCommunicator.cu 
    // end of creatinng communicator data//

    const int basicSize = 32;
    const real parameter2Adjust = 0.5;
    size_t *sharedMemorySize0=NULL;
    size_t *sharedMemorySize1=NULL;
    
    real *meanNnzPerRow0=NULL;
    real *meanNnzPerRow1=NULL;
    real *sd0=NULL;
    real *sd1=NULL;
    
    dim3 *block0=NULL;
    dim3 *block1=NULL;
    dim3 *grid0=NULL;
    dim3 *grid1=NULL;

    cudaStream_t *stream0=NULL;
    cudaStream_t *stream1=NULL;
