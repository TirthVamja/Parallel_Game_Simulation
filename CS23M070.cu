#include <thrust/device_vector.h>
#include<thrust/copy.h>

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#define MAX_DIST 1e18
using namespace std;

//*******************************************

// Write down the kernels here
__device__ long long min_val;

__device__ bool comp(long long x1, long long y1, long long x2, long long y2, long long x3, long long y3){
    bool flgx=false, flgy=false;
    if((x1>=x2 && x1>=x3) || (x1<=x2 && x1<=x3)) flgx=true;
    if((y1>=y2 && y1>=y3) || (y1<=y2 && y1<=y3)) flgy=true;
    return (flgx && flgy);
}
                                                                          
__global__ void reduce_health(long long *SCORE, int *HEALTH, long long *HIT, int *sz, int T){
    long long attacker = threadIdx.x;
    long long victim = HIT[attacker];
    if(victim!=T){
        SCORE[attacker]+=1;
        long long old_health = atomicAdd(&HEALTH[victim],-1);
        if(old_health==1) atomicAdd(sz,-1);
    }
}

__global__ void copy_health_and_score(int *HEALTH, long long *SNAP_HEALTH, long long *SCORE, long long *SNAP_SCORE, int T){
    long long id = threadIdx.x;
    SNAP_HEALTH[id] = HEALTH[id];
    SNAP_SCORE[id] = SCORE[id];
}
__global__ void play_fast(int *HEALTH, long long *SNAP_HEALTH, long long *SCORE, long long *SNAP_SCORE, int T){
    long long id = threadIdx.x;
    if(id==0) min_val = MAX_DIST;
    __syncthreads();
    if(HEALTH[id]>0){
        long long current_health = HEALTH[id];
        long long health_change = SNAP_HEALTH[id] - current_health;
        long long rounds_before_death = ceil(current_health/(float)health_change) - 1;
        atomicMin(&min_val, rounds_before_death);
    }
    __syncthreads();
    if(HEALTH[id]>0){
        long long current_health = HEALTH[id];
        long long health_change = SNAP_HEALTH[id] - current_health;
        long long current_score = SCORE[id];
        long long score_change = current_score - SNAP_SCORE[id];
        SCORE[id]+=(score_change * min_val);
        HEALTH[id]-=(health_change * min_val);
    }
    //if(id==0) printf("Fast Forward --> %d\n",min_val);
}

__global__ void play_game(long long *XCOORD, long long *YCOORD, long long *SCORE, int *HEALTH, long long *HIT, long long *DIST, int gap, int T){
    long long attacker = blockIdx.x, new_victim = threadIdx.x;
    long long org_victim=(attacker+gap)%T;
    long long xa = XCOORD[attacker], xov = XCOORD[org_victim], xnv = XCOORD[new_victim];
    long long ya = YCOORD[attacker], yov = YCOORD[org_victim], ynv = YCOORD[new_victim];
    
    if(new_victim==0){
        DIST[attacker] = MAX_DIST;
        HIT[attacker] = T;
    }
    __syncthreads();
    long long non_collinear = xa*(yov-ynv) + xov*(ynv-ya) + xnv*(ya-yov);
    long long comp_dist = MAX_DIST;
    if(new_victim!=attacker && HEALTH[attacker]>0 && HEALTH[new_victim]>0 && non_collinear==0 && comp(xa,ya,xov,yov,xnv,ynv)){    
        long long x=(xa-xnv),y=(ya-ynv);
        comp_dist = x*x+y*y;
        atomicMin(&DIST[attacker],comp_dist);
    }
    __syncthreads();
    if(comp_dist!=MAX_DIST && comp_dist==DIST[attacker]) HIT[attacker]=new_victim;  
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    thrust::device_vector<long long> XCOORD(T), YCOORD(T), SCORE(T,0), HIT(T,T), SNAP_SCORE(T,0), SNAP_HEALTH(T,H),DIST(T);
    thrust::device_vector<int> HEALTH(T,H);
    thrust::copy(xcoord, xcoord+T, XCOORD.begin());
    thrust::copy(ycoord, ycoord+T, YCOORD.begin());

    int *sz;
    cudaHostAlloc(&sz, sizeof(int), 0);
    *sz=T;

    for(int gap=0, prev_sz=T;*sz>1;gap=(gap+1)%T){
        if(gap==0){
            copy_health_and_score<<<1,T>>>(thrust::raw_pointer_cast(HEALTH.data()),thrust::raw_pointer_cast(SNAP_HEALTH.data()),thrust::raw_pointer_cast(SCORE.data()),thrust::raw_pointer_cast(SNAP_SCORE.data()),T);
            prev_sz = *sz;
            continue;
        }
        play_game<<<T,T>>>(thrust::raw_pointer_cast(XCOORD.data()), thrust::raw_pointer_cast(YCOORD.data()), thrust::raw_pointer_cast(SCORE.data()), thrust::raw_pointer_cast(HEALTH.data()), thrust::raw_pointer_cast(HIT.data()), thrust::raw_pointer_cast(DIST.data()), gap, T);
        reduce_health<<<1,T>>>(thrust::raw_pointer_cast(SCORE.data()),thrust::raw_pointer_cast(HEALTH.data()),thrust::raw_pointer_cast(HIT.data()),sz,T);
        cudaDeviceSynchronize();
        if(gap==T-1 && prev_sz==*sz) play_fast<<<1,T>>>(thrust::raw_pointer_cast(HEALTH.data()), thrust::raw_pointer_cast(SNAP_HEALTH.data()),thrust::raw_pointer_cast(SCORE.data()), thrust::raw_pointer_cast(SNAP_SCORE.data()), T);
    }
    thrust::copy(SCORE.begin(), SCORE.end(), score);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}