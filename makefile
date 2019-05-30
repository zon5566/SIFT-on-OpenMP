ICC = icc
OPENCV = `pkg-config --cflags --libs opencv`
CXXFLAGS = $(OPENCV) --std=c++11 -qopenmp

sift: sift.o util.o
	$(ICC) sift.o util.o -o sift $(CXXFLAGS) -fstack-protector-all

sift.o: sift.cpp
	$(ICC) -c sift.cpp $(CXXFLAGS)

util.o: util.cpp
	$(ICC) -c util.cpp

clean:
	rm -f sift.o sift  util.o
