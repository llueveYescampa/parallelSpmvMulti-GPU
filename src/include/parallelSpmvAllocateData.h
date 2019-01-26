    // allocating external arrays depending on number of gpus
    n            = (int *) malloc(ngpus * sizeof(int)); 
    on_proc_nnz  = (int *) calloc(ngpus, sizeof(int)); 
    off_proc_nnz = (int *) calloc(ngpus, sizeof(int)); 

    nColsOff     = (int *) calloc(ngpus, sizeof(int)); 


    // data for the on_proc solution: host an device
    row_ptr =    (int **)  malloc(ngpus * sizeof(int *)); 
    col_idx =    (int **)  malloc(ngpus * sizeof(int *)); 
    val     =    (real **) malloc(ngpus * sizeof(real *)); 

    row_ptr_d = (int **)  malloc(ngpus * sizeof(int *)); 
    col_idx_d = (int **)  malloc(ngpus * sizeof(int *)); 
    val_d     = (real **) malloc(ngpus * sizeof(real *)); 
    // end of data for the on_proc solution: host an device

    recvSendCount = (int **) malloc(ngpus * sizeof(int *)); 
    sendColumns   = (int ***) malloc(ngpus * sizeof(int **)); 
    
    
    w     = (real **) malloc(ngpus * sizeof(real *));
    v     = (real **) malloc(ngpus * sizeof(real *));
    
    w_d     = (real **) malloc(ngpus * sizeof(real *));
    v_d     = (real **) malloc(ngpus * sizeof(real *));
    
    if (ngpus > 1) {
        row_ptr_off   = (int **)  malloc(ngpus * sizeof(int *)); 
        col_idx_off   = (int **)  malloc(ngpus * sizeof(int *)); 
        val_off       = (real **) malloc(ngpus * sizeof(real *)); 

        row_ptr_off_d = (int **)  malloc(ngpus * sizeof(int *)); 
        col_idx_off_d = (int **)  malloc(ngpus * sizeof(int *)); 
        val_off_d     = (real **) malloc(ngpus * sizeof(real *)); 


        v_off   = (real **) malloc(ngpus * sizeof(real *));
        v_off_d = (real **) malloc(ngpus * sizeof(real *));
    } // end if //
    
    meanNnzPerRow0 = (real*) malloc(ngpus*sizeof(real));
    meanNnzPerRow1 = (real*) malloc(ngpus*sizeof(real));
    
    sd0            = (real*) malloc(ngpus*sizeof(real ));
    sd1            = (real*) malloc(ngpus*sizeof(real ));
    
    block0 = (dim3 *) malloc(ngpus*sizeof(dim3 )); 
    block1 = (dim3 *) malloc(ngpus*sizeof(dim3 )); 
    grid0  = (dim3 *) malloc(ngpus*sizeof(dim3 )); 
    grid1  = (dim3 *) malloc(ngpus*sizeof(dim3 )); 
    sharedMemorySize0 = (size_t *) calloc(ngpus, sizeof(size_t)); 
    sharedMemorySize1 = (size_t *) calloc(ngpus, sizeof(size_t)); 

    for (int gpu=0; gpu<ngpus; ++gpu) {
        
        block0[gpu].x = basicSize;
        block0[gpu].y = 1;
        block0[gpu].z = 1;
        block1[gpu].x = basicSize;
        block1[gpu].y = 1;
        block1[gpu].z = 1;
        
        grid0[gpu].x = 1;
        grid0[gpu].y = 1;
        grid0[gpu].z = 1;
        grid1[gpu].x = 1;
        grid1[gpu].y = 1;
        grid1[gpu].z = 1;
    } // end for //
    
    stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * ngpus);
    // end of allocating external arrays depending on number of gpus
