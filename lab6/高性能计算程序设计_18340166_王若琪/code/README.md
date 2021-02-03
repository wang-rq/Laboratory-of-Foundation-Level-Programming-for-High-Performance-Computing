# README

在 code 文件夹下有三个源文件：

- 任务一对应源文件 1.cu 

  编译运行指令：

  ```
  nvcc 1.cu -o 1   
  ./1 <M> <N> <K> <CUDA_BLOCK_SIZE>
  ```

- 任务二对应源文件 2.cu 

  编译运行指令：

  ```
  nvcc -Xcompiler -fopenmp -arch=sm_70 2.cu -o 2     
  ./2 <M> <N> <K> <CUDA_BLOCK_SIZE> <OpenMp_THREAD_NUM>
  ```

- 任务三对应源文件 3.cu 

  编译运行指令：

  ```
  nvcc 3.cu -o 3 -lcublas  
  ./3 <M> <N> <K>
  ```

  