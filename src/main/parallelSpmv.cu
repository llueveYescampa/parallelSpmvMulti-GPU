#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "real.h"
#include "dataDef.h"

#include "parallelSpmv.h"

#define MAXTHREADS 128
#define REP 1000

#ifdef DOUBLE
    texture<int2>  xTex;
    texture<int2>  valTex;
#else
    texture<float> xTex;
    texture<float> valTex;
#endif

void meanAndSd(real *mean, real *sd,real *data, int n)
{
    real sum = (real) 0.0; 
    real standardDeviation = (real) 0.0;

    for(int i=0; i<n; ++i) {
        sum += data[i];
    } // end for //

    *mean = sum/n;

    for(int i=0; i<n; ++i) {
        standardDeviation += pow(data[i] - *mean, 2);
    } // end for //
    *sd=sqrt(standardDeviation/n);
} // end of calculateSD //


int main(int argc, char *argv[]) 
{
    #include "parallelSpmvData.h"

    cudaError_t cuda_ret;
    
    cuda_ret = cudaGetDeviceCount(&ngpus);
    if(cuda_ret != cudaSuccess) FATAL("Unable to deternine number of GPUs");
    //ngpus=4;

    // verifing number of input parameters //
   char exists='t';
   char checkSol='f';
   
    if (argc < 3 ) {
        printf("Use: %s  Matrix_filename InputVector_filename  [SolutionVector_filename]  \n", argv[0]);     
        exists='f';
    } // endif //
    
    FILE *fh=NULL;
    // testing if matrix file exists
    if((fh = fopen(argv[1], "rb")  )   == NULL) {
        printf("No matrix file found.\n");
        exists='f';
    } // end if //
    
    // testing if input file exists
    if((fh = fopen(argv[2], "rb")  )   == NULL) {
        printf("No input vector file found.\n");
        exists='f';
    } // end if //

    // testing if output file exists
    if (argc  >3 ) {
        if((fh = fopen(argv[3], "rb")  )   == NULL) {
            printf("No output vector file found.\n");
            exists='f';
        } else {
            checkSol='t';
        } // end if //
    } // end if //
    if (fh) fclose(fh);
        
    if (exists == 'f') {
        printf("Quitting.....\n");
        exit(0);
    } // end if //

    printf("%s Precision. Solving using %d %s\n", (sizeof(real) == sizeof(double)) ? "Double": "Single", ngpus, (ngpus > 1) ? "GPUs": "GPU"  );

    #include "parallelSpmvAllocateData.h" 
    
    reader(&n, 
           on_proc_nnz,
           off_proc_nnz,
           &row_ptr,&col_idx,&val,
           &row_ptr_off,&col_idx_off,&val_off,
           argv[1], ngpus);
           
           
    if (ngpus>1) {
        createCommunicator(nColsOff, sendColumns, recvSendCount , col_idx_off, off_proc_nnz, n,&ngpus);
    } // end if //    

    
    // ready to start //    
    
    int totalNNZ=0;
    for (int gpu=0; gpu<ngpus; ++gpu) {
        totalNNZ+=on_proc_nnz[gpu];
        totalNNZ+=off_proc_nnz[gpu];
        
        cuda_ret = cudaSetDevice(gpu);
        if(cuda_ret != cudaSuccess) FATAL("Unable to set gpu");
    
        //cuda_ret = cudaStreamCreateWithFlags(&stream0[gpu], cudaStreamDefault);
        cuda_ret = cudaStreamCreateWithFlags(&stream[gpu], cudaStreamNonBlocking ) ;
        if(cuda_ret != cudaSuccess) FATAL("Unable to create stream0 ");
        
        v[gpu] = (real *) malloc((n[gpu])*sizeof(real));
        w[gpu] = (real *) malloc((n[gpu])*sizeof(real));
        vectorReader(v[gpu], &gpu, n, argv[2]);
        if (ngpus > 1) cudaHostAlloc((real **)&v_off[gpu]  , nColsOff[gpu]*sizeof(real),cudaHostAllocDefault);


        /////////////////////////////////////////////////////
        // determining the standard deviation of the nnz per row
        real *temp=(real *) malloc((n[gpu])*sizeof(real));
        for (int row=0; row<n[gpu]; ++row) {
            temp[row] = row_ptr[gpu][row+1] - row_ptr[gpu][row];
        } // end for //
        meanAndSd(&meanNnzPerRow0[gpu],&sd0[gpu],temp,n[gpu]);
//printf("file: %s, line: %d, gpu on-prcoc:   %d, mean: %7.3f, sd: %7.3f using: %s \n", __FILE__, __LINE__, gpu , meanNnzPerRow0[gpu], sd0[gpu], (meanNnzPerRow0[gpu] + 0.5*sd0[gpu] < 32) ? "spmv0": "spmv1");
        if (nColsOff[gpu]) {
            for (int row=0; row<n[gpu]; ++row) {
                temp[row] = row_ptr_off[gpu][row+1] - row_ptr_off[gpu][row];
            } // end for //
            meanAndSd(&meanNnzPerRow1[gpu],&sd1[gpu],temp,n[gpu]);
//printf("file: %s, line: %d, gpu off-prcoc:  %d, mean: %7.3f, sd: %7.3f using: %s \n", __FILE__, __LINE__, gpu , meanNnzPerRow1[gpu], sd1[gpu], (meanNnzPerRow1[gpu] + 0.5*sd1[gpu] < 32) ? "spmv0": "spmv1");
        } // end if //        
        free(temp);
        /////////////////////////////////////////////////////
    
        cudaSetDevice(gpu);
        //printf("file: %s, line: %d, setting gpu: %d\n", __FILE__, __LINE__,gpu);        

       
        // Allocating device memory for on_proc input matrices 

        cuda_ret = cudaMalloc((void **) &row_ptr_d[gpu],  (n[gpu]+1)      * sizeof(int) );
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
        
        cuda_ret = cudaMalloc((void **) &col_idx_d[gpu], on_proc_nnz[gpu] * sizeof(int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");

        cuda_ret = cudaMalloc((void **) &val_d[gpu],     on_proc_nnz[gpu] * sizeof(real));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

        // Copy the input on_proc  matrices from the host memory to the device memory
        
        cuda_ret = cudaMemcpy(row_ptr_d[gpu], row_ptr[gpu], (n[gpu]+1)*sizeof(int),cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix row_ptr_d");

        cuda_ret = cudaMemcpy(col_idx_d[gpu], col_idx[gpu], on_proc_nnz[gpu]*sizeof(int),cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix col_idx_d");

        cuda_ret = cudaMemcpy(val_d[gpu], val[gpu],         on_proc_nnz[gpu]*sizeof(real),cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix val_d");

        // Allocating device memory for inpit and output vectors

        cuda_ret = cudaMalloc((void **) &(w_d[gpu]),  n[gpu]*sizeof(real));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for w_d");
        
        cuda_ret = cudaMalloc((void **) &(v_d[gpu]),  n[gpu]*sizeof(real));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for v_d");

        // Copy the input vector to device

        cuda_ret = cudaMemcpy(v_d[gpu], v[gpu], n[gpu]*sizeof(real),cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix v_d");



        if (ngpus > 1) {
            // Allocating device memory for off_proc input matrices 
            cuda_ret = cudaMalloc((void **) &row_ptr_off_d[gpu],  (n[gpu]+1)*sizeof(int));
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");

            cuda_ret = cudaMalloc((void **) &col_idx_off_d[gpu], off_proc_nnz[gpu] * sizeof(int));
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");

            cuda_ret = cudaMalloc((void **) &val_off_d[gpu],  off_proc_nnz[gpu] *sizeof(real));
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");


            // Copy the input off_proc  matrices from the host memory to the device memory

            cuda_ret = cudaMemcpy(col_idx_off_d[gpu], col_idx_off[gpu], off_proc_nnz[gpu]*sizeof(int),cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix col_idx_d");

            cuda_ret = cudaMemcpy(val_off_d[gpu]   , val_off[gpu],      off_proc_nnz[gpu]*sizeof(real),cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix val_d");

            cuda_ret = cudaMemcpy(row_ptr_off_d[gpu], row_ptr_off[gpu], (n[gpu]+1)*sizeof(int),cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix row_ptr_d");

            // Allocating device memory for inpit and output vectors
            cuda_ret = cudaMalloc((void **) &v_off_d[gpu],  nColsOff[gpu] *sizeof(real));
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for v_off_d");

        } // end if //


/////////////////////////////////////////////////////////////////////////

        printf("In GPU: %d\n",gpu);
        if (meanNnzPerRow0[gpu] < warpSize && parameter2Adjust*sd0[gpu] < warpSize) {
        	// these mean use scalar spmv
            if (meanNnzPerRow0[gpu] < (real) 4.5) {
                block0[gpu].x=128;
            } else if (meanNnzPerRow0[gpu]< (real) 14.4) {
                block0[gpu].x=64;
            } else {
                block0[gpu].x=32;
            } // end if //

            grid0[gpu].x = ( (n[gpu] + block0[gpu].x -1) /block0[gpu].x );
            printf("using scalar spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",block0[gpu].x,block0[gpu].y, meanNnzPerRow0[gpu], sd0[gpu]) ;
        } else {
            // these mean use vector spmv
            if (meanNnzPerRow0[gpu] > 10.0*warpSize) {
                block0[gpu].x=2*warpSize;
            }  else {
                block0[gpu].x=warpSize;
            } // end if //
            block0[gpu].y=MAXTHREADS/block0[gpu].x;
            grid0[gpu].x = ( (n[gpu] + block0[gpu].y - 1) / block0[gpu].y ) ;
        	sharedMemorySize0[gpu]=block0[gpu].x*block0[gpu].y*sizeof(real);
            printf("using vector spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",block0[gpu].x,block0[gpu].y, meanNnzPerRow0[gpu], sd0[gpu]) ;
        } // end if // 

        if (ngpus > 1) {
            if (meanNnzPerRow1[gpu] < warpSize  && parameter2Adjust*sd1[gpu] < warpSize) {
            	// these mean use scalar spmv
                if (meanNnzPerRow1[gpu] < (real) 4.5) {
                    block1[gpu].x=128;
                } else if (meanNnzPerRow1[gpu] < (real) 14.4) {
                    block1[gpu].x=64;
                } else {
                    block1[gpu].x=32;
                } // end if //
                grid1[gpu].x = ( (n[gpu] + block1[gpu].x -1) /block1[gpu].x );
                printf("using scalar spmv for off matrix, blockSize: [%d, %d] %f, %f\n",block1[gpu].x,block1[gpu].y, meanNnzPerRow1[gpu], sd1[gpu]) ;
            } else {
                // these mean use vector spmv
                if (meanNnzPerRow1[gpu] > 10.0*warpSize) {
                    block1[gpu].x=2*warpSize;
                }  else {
                    block1[gpu].x=warpSize;
                } // end if //

                block1[gpu].y=MAXTHREADS/block1[gpu].x;
                grid1[gpu].x = ( (n[gpu] + block1[gpu].y - 1) / block1[gpu].y ) ;
            	sharedMemorySize1[gpu]=block1[gpu].x*block1[gpu].y*sizeof(real);
                printf("using vector spmv for off matrix, blockSize: [%d, %d] %f, %f\n",block1[gpu].x,block1[gpu].y, meanNnzPerRow1[gpu], sd1[gpu]) ;
            } // end if // 
        }
    } // end for //

    // Timing should begin here//
    struct timeval tp;                                   // timer
    double elapsed_time;
    
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    
    for (int t=0; t<REP; ++t) {
        // send the first spmv
        for (int gpu=0; gpu<ngpus; ++gpu) {
            cudaSetDevice(gpu);
        
            cuda_ret = cudaMemset(w_d[gpu], 0, sizeof(real)*n[gpu] );
            if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d[gpu]");

            cuda_ret = cudaBindTexture(NULL, xTex,   v_d[gpu],   n[gpu]           * sizeof(real));
            cuda_ret = cudaBindTexture(NULL, valTex, val_d[gpu], on_proc_nnz[gpu] * sizeof(real));
            spmv<<<grid0[gpu], block0[gpu], sharedMemorySize0[gpu]>>>(w_d[gpu],  row_ptr_d[gpu], col_idx_d[gpu], n[gpu]);
            cuda_ret = cudaUnbindTexture(xTex);
            cuda_ret = cudaUnbindTexture(valTex);
            
        } // end for //
        
        if (ngpus > 1 ) {
            startComunication(v,v_off,recvSendCount, sendColumns, &ngpus);
            // send the second spmv
            for (int gpu=0; gpu<ngpus; ++gpu) {
                cudaSetDevice(gpu);
            
                cuda_ret = cudaMemcpyAsync(v_off_d[gpu], v_off[gpu], nColsOff[gpu]*sizeof(real),cudaMemcpyHostToDevice,stream[gpu] ) ;
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device array v_off_d");
            
                cuda_ret = cudaBindTexture(NULL, xTex,   v_off_d[gpu],   nColsOff[gpu]     * sizeof(real));
                cuda_ret = cudaBindTexture(NULL, valTex, val_off_d[gpu], off_proc_nnz[gpu] * sizeof(real));
                spmv<<<grid1[gpu], block1[gpu], sharedMemorySize1[gpu],stream[gpu] >>>(w_d[gpu],  row_ptr_off_d[gpu], col_idx_off_d[gpu], n[gpu]);
                cuda_ret = cudaUnbindTexture(xTex);
                cuda_ret = cudaUnbindTexture(valTex);
                
            } // end for //
        } // end if //

        for (int gpu=0; gpu<ngpus; ++gpu) {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(NULL);
            cudaStreamSynchronize(stream[gpu]);
        } // end for //
    } // end for //
    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds, GFLOPS: %f\n", elapsed_time*1.0e-6,  2.0*totalNNZ*REP*1.0e-3/elapsed_time  );
    
    for (int gpu=0; gpu<ngpus; ++gpu) {
        cudaSetDevice(gpu);
        cuda_ret = cudaMemcpy(w[gpu], w_d[gpu], n[gpu]*sizeof(real),cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");
    } // end for //




    if (checkSol=='t') {

        for (int gpu=0; gpu<ngpus; ++gpu) {
            real *sol= (real *) malloc( n[gpu] * sizeof(real));  
            cudaSetDevice(gpu);
            // reading input vector
            vectorReader(sol, &gpu, n, argv[3]);
            
            int row=0;
            real tolerance = 1.0e-08;
            if (sizeof(real) != sizeof(double) ) {
                tolerance = 1.0e-02;
            } // end if //
            real error;

            do {
                error =  fabs(sol[row] - w[gpu][row]) /fabs(sol[row]);
                if ( error > tolerance ) break;
                ++row;
            } while (row < n[gpu]); // end do-while //
            
            if (row == n[gpu]) {
                printf("Solution match in gpu %d\n",gpu);
            } else {    
                printf("For Matrix %s, solution does not match at element %d in gpu %d   %20.13e   -->  %20.13e  error -> %20.13e, tolerance: %20.13e \n", 
                argv[1], (row+1),gpu, sol[row], w[gpu][row], error , tolerance  );
            } // end if //
            free(sol);    
        } // end for //

        
    } // end if //

    #include "parallelSpmvCleanData.h" 
    return 0;    
//    printf("file: %s, line: %d, so far so good\n", __FILE__, __LINE__ ); exit(0);
} // end main() //
