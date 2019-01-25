    free(n);
    free(on_proc_nnz);
    
    free(nColsOff);


    for (int gpu=0; gpu<ngpus; ++gpu) {
        cudaSetDevice(gpu);
        free(row_ptr[gpu]);
        free(col_idx[gpu]);
        free(val[gpu]);

        cudaFreeHost(w);
        cudaFreeHost(v);
        //free(v[gpu]);
        //free(w[gpu]);

        cudaFree(row_ptr_d[gpu]);
        cudaFree(col_idx_d[gpu]);
        cudaFree(val_d[gpu]);

        cudaFree(v_d[gpu]);
        cudaFree(w_d[gpu]);

//        if ( off_proc_nnz[gpu] ) {
        if ( ngpus > 1) {
            free(row_ptr_off[gpu]);
            free(col_idx_off[gpu]);
            free(val_off[gpu]);
            free(v_off[gpu]);
            
            cudaFree(row_ptr_off_d[gpu]);
            cudaFree(col_idx_off_d[gpu]);
            cudaFree(val_off_d[gpu]);
            cudaFree(v_off_d[gpu]);
            
            free(recvSendCount[gpu]);            
        } // end if //
        cudaStreamDestroy(stream0[gpu]);
        cudaStreamDestroy(stream1[gpu]);
    } // end for /
    free(recvSendCount);
    free(stream0);
    free(stream1);
    

    free(v);
    free(w);
    free(v_d);
    free(w_d);

    free(v_off);
    free(v_off_d);
    
    free(off_proc_nnz);

    free(row_ptr);
    free(col_idx);
    free(val);
    
    free(row_ptr_d);
    free(col_idx_d);
    free(val_d);


    free(meanNnzPerRow0);
    free(meanNnzPerRow1);
    free(sd0);
    free(sd1);
    free(sharedMemorySize0);
    free(sharedMemorySize1);
    
    free(block0);
    free(block1);
    free(grid0);
    free(grid1);

    if (ngpus > 1) {
        free(row_ptr_off);
        free(col_idx_off);
        free(val_off);

        free(row_ptr_off_d);
        free(col_idx_off_d);
        free(val_off_d);


        
        for (int proc=0; proc<ngpus; ++proc){
            for (int gpu=0; gpu<ngpus; ++gpu){
                free(sendColumns[proc][gpu]);
            } // end for /
            free(sendColumns[proc]);
        } // end for /
        
        
    } // end if //
    
    free(sendColumns);

