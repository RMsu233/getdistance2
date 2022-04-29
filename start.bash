rm test1
g++ getdistance2.cpp -o test1  `pkg-config --libs --cflags opencv4`
./test1 