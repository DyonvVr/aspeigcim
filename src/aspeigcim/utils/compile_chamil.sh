g++ -c -fPIC ../classes/chamil.cpp -o ../obj/chamil.o
g++ -shared -Wl,-soname,libchamil.so -o ../lib/libchamil.so  ../obj/chamil.o
