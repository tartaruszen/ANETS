
#Make sure you set a correct matlab path
CC = g++
#CFLAGS = -I /usr/include/c++/4.6/
CFLAGS = 
SOURCES=src/CoarseNet.cpp src/DisjointSets.cpp src/Edge.cpp src/graph.cpp
EXECUTABLE=src/CoarseNet

EXAMPLE = ../data/disney/
percent = 70
THREAD1 = 1
THREAD2 = 1
 


$(EXECUTABLE): $(SOURCES)
	$(CC) $(SOURCES)  -std=c++0x $(CFLAGS) -o $(EXECUTABLE)

demo:
	python Anets.py $(EXAMPLE) $(percent) $(THREAD1) $(THREAD2)

clean:
	rm -rf $(EXECUTABLE) 
