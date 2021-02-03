# README

在 code 文件夹下有以下目录或文件：



###　gemm_1.c 对应第1题，编译运行指令如下：

```
gcc -o gemm_1 gemm_1.c -fopenmp
./gemm_1 <number of threads>
```



### gemm_schedules_compare.c 对应第2题，编译运行指令如下：

```
gcc -o gemm_schedules_compare gemm_schedules_compare.c -fopenmp
./gemm_schedules_compare <number of threads>
```



### 文件夹 4.3 对应第3题，其中各个文件的功能及编译运行方法请见实验报告。