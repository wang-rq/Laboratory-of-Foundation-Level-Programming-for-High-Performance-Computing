#ifndef func_h
#define func_h
void parallel_for(int start, int end, int increment, void *(*functor)(void*), void *arg, int num_thread);
#endif 