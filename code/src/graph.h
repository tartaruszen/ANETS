#pragma once

#include <string>
#include <list>
#include "Edge.h"
#include <unordered_map>
#include <unordered_set>
#include <iostream>

using namespace std;

class Graph
{
public:
	Graph(string fileName);
	void addEdge(int v1,int v2,double w);
	double getEdgeWeight(int v1,int v2);
	void getNeighborVertices(int vertexID,list<int> *t);
	void getAllNeighborVerticesAsSet(int vertexID,unordered_set<int> *t);
	void UpdateWeight(int v1,int v2,double w);
	void deleteEdge(int v1,int v2);
	unordered_set<int> getVertices();
	void writeToFile(ofstream &fout);
	void writeToFileInf(ostream &fout, ostream &mapout);
	void deleteVertex(int v1);
	int getNumOfVertices();
	int getNumOfEdges();
private:
	void initialize(string fileName);	
	void buildGraph(int v1, int v2, double w);
	unordered_map <int, unordered_map<int,double> > edge_map;
	unordered_map <int, unordered_map<int,double> > :: const_iterator neighbor_iter;
	unordered_map <int, double> :: const_iterator edge_iter;
	unordered_set <int> vertices;
	int numOfEdges;
};

