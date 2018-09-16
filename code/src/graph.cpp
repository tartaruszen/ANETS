#include "graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace std;

/*
  Uses sparse adjacency matrix representation for graph
  Always assume directed graph
 */

/**
   Assume that fileName contains graph in weighted edge list format. Directed graph by default, thus, in case of undirected graphs, each edge should be given twice in appropriate vertex order.
 */
Graph::Graph(string fileName)
{
  numOfEdges = 0;
  initialize(fileName);
}

void Graph::initialize(string fileName){
  string line;

  ifstream graphfile (fileName.c_str());
  int v1, v2;
  double weight;
  if (graphfile.is_open()) {
    //getline(graphfile, line); //To ignore the first line which has number of nodes / edges
    while (getline (graphfile,line)) {
      string from, to, last1, last2;
      istringstream lines( line );
      getline( lines, from, '\t' );
      getline( lines, to, '\t' );
      getline( lines, last1, '\t' );
      getline( lines, last2, '\t' );
      v1 = atoi( from.c_str() );      
      v2 = atoi( to.c_str() );      
      weight = atof(last1.c_str() );
      buildGraph(v1,v2,weight);
      weight = atof(last2.c_str() );
      buildGraph(v2,v1,weight);
      //cout<<"Adding "<<v1<<","<<v2<<","<<weight<<endl;
    }
    graphfile.close();
  }
  else 
    cout << "Unable to open file "<< fileName; 
}

/**
   Adds edge v1,v2 with weight w
   If v2,v1 does not exist, add 0 weight edge
 */
void Graph::buildGraph(int v1,int v2,double w){
  vertices.insert(v1);
  vertices.insert(v2);
  /*neighbor_iter = edge_map.find(v1);
  if(neighbor_iter == edge_map.end()) {
    numOfEdges++;
  }
  else if((neighbor_iter->second).find(v2) == (neighbor_iter->second).end()){
    numOfEdges++;
    }*/
  if(edge_map[v1][v2]==0) {
    numOfEdges++;
  }
  UpdateWeight(v1,v2,w);
  edge_map[v2][v1];
}

/**
   Assumes that the edge exists and updates the weight. Doesn't bother about complementary (v2, v1) edge.
 */
void Graph::UpdateWeight(int v1,int v2,double w){
  edge_map[v1][v2]=w;
}

/**
   Deletes edge (v1,v2).
   Decrements numOfEdges
   If (v2,v1) is not zero, then (v1,v2) will remain as a zero weight edge.
   If (v2,v1) is zero, then (v1,v2) and (v2,v1) will both be erased.
 */
void Graph::deleteEdge(int v1,int v2){
  if(getEdgeWeight(v1,v2)==0) {
    return;
  }
    
  if(getEdgeWeight(v2,v1) != 0)
    edge_map[v1][v2] = 0;
  else {
    edge_map[v1].erase(v2);
    if(edge_map[v1].size()==0) {
      edge_map.erase(v1);
    }

    edge_map[v2].erase(v1);
    if(edge_map[v2].size() ==0) {
      edge_map.erase(v2);
    }
  }
  numOfEdges--;
}

/**
   Returns the edge weight. If return value is 0, the edge does not exist.
   This may imply that both (v1, v2) and (v2, v1) don't exist or that only (v1,v2) doesn't exist.
 */
double Graph::getEdgeWeight(int v1,int v2){
  neighbor_iter=edge_map.find(v1);
  if(neighbor_iter!=edge_map.end()){
    edge_iter=(neighbor_iter->second).find(v2);
    if(edge_iter!=(neighbor_iter->second).end()){
      return edge_iter->second;
    }
    else return 0;
  }
  else 
    return 0;
}

/**
Adds all out neighbors of vertexID into list t. Zero weight edges are ignored.
 */
void Graph::getNeighborVertices(int vertexID,list<int> *t){
  neighbor_iter=edge_map.find(vertexID);
  for (edge_iter = (neighbor_iter->second).begin();
       edge_iter != (neighbor_iter->second).end(); edge_iter++){
    if(edge_iter->second != 0) {
      t->push_back(edge_iter->first);
    }
  }
}

void Graph::getAllNeighborVerticesAsSet(int vertexID,unordered_set<int> *t){
  neighbor_iter=edge_map.find(vertexID);
  for (edge_iter = (neighbor_iter->second).begin();
       edge_iter != (neighbor_iter->second).end(); edge_iter++){
    t->insert(edge_iter->first);
  }
}


/**
   Add edge (v1,v2) with weight w
   When adding (v1,v2), also add (v2,v1) as a 0 wt edge if it doesn't exist already
 */
void Graph::addEdge(int v1,int v2,double w){
  if(getEdgeWeight(v1,v2) == 0) {// && getEdgeWeight(v2,v1)==0) {
    ////cout<<"("<<v1<<","<<v2<<") needs to be added."<<endl;
    ////cout<<"Increasing numOfEdges from"<<numOfEdges;
    numOfEdges++;
    ////cout<<" to"<<numOfEdges<<endl;
  }
  UpdateWeight(v1,v2,w);
  edge_map[v2][v1];
}



unordered_set<int> Graph::getVertices(){
  return vertices;
}

int Graph::getNumOfVertices() {
  return vertices.size();
}

int Graph::getNumOfEdges() {
  return numOfEdges;
}

void Graph::writeToFile(ofstream &fout) {
  fout<<"source\ttarget\tweight"<<endl;
  for(neighbor_iter = edge_map.begin() ; neighbor_iter != edge_map.end() ; neighbor_iter++ ) {
    for(edge_iter = (neighbor_iter->second).begin(); edge_iter != (neighbor_iter->second).end(); edge_iter++) {
      if(edge_iter->second != 0)
	fout<<neighbor_iter->first<<"\t"<<edge_iter->first<<"\t"<<edge_iter->second<<endl;
    }
  }  
}

void Graph::writeToFileInf(ostream &fout, ostream &mapout ) {
  //fout<<getNumOfVertices()<<" "<<getNumOfEdges()<<endl;
  int u,v;
  int i=1;
  unordered_map<int, int> map_our_to_chen;
  unordered_map<int, int> map_chen_to_our;
  //map_our_to_chen.reserve(getNumOfVertices());
  //map_chen_to_our.reserve(getNumOfVertices());
  for(unordered_set<int>::const_iterator x = vertices.begin(); x!=vertices.end();++x) {
    map_chen_to_our[i] = *x;
    map_our_to_chen[*x] = i;
    i++;
  }
  for(neighbor_iter = edge_map.begin(); neighbor_iter != edge_map.end(); neighbor_iter++) {
    for(edge_iter = (neighbor_iter->second).begin(); edge_iter != (neighbor_iter->second).end(); edge_iter++) {
      u = neighbor_iter->first;
      v = edge_iter->first;
      if(edge_iter->second !=0)
	fout<<map_our_to_chen[u]<<"\t"<<map_our_to_chen[v]<<"\t"<<edge_iter->second<<"\t"<<edge_map[v][u]<<endl;
      //if(edge_map[v][u] != 0)
      //fout<<map_our_to_chen[v]<<"\t"<<map_our_to_chen[u]<<"\t"<<edge_map[v][u]<<"\t"<<edge_iter->second<<endl;
    }
  }
  //write the map
  //for(auto& x : map_chen_to_our) {
  for(unordered_map<int, int>::const_iterator x = map_chen_to_our.begin(); x != map_chen_to_our.end(); x++) {
    mapout<<(*x).first<<" : "<<(*x).second<<endl;
  }
}

void Graph::deleteVertex(int v) {
  //cout <<"Deleting "<<v<<endl;
  neighbor_iter = edge_map.find(v);
  //cout<<"Got iter"<<endl;
  if(neighbor_iter == edge_map.end()) { //v is an isolated vertex
    //cout <<"Erasing 1 "<<v<<endl;
    vertices.erase(v);
    //cout <<"Erased 1 "<<v<<endl;
    return;
  } 
  int count = 0, neighbor, weight;
  //cout<<"Num of neighbors : "<<edge_map[v].size()<<endl;
  //cout<<"Num of neighbors : "<<(neighbor_iter->second).size()<<endl;
  
  for(edge_iter = (neighbor_iter->second).begin(); edge_iter != (neighbor_iter->second).end(); ++edge_iter) {
    
    //cout<<"Checking neighbor ";
    //cout<<flush;
    neighbor = edge_iter->first;
    //cout<<neighbor<<endl;
    weight = edge_iter->second;
    //cout<<"Weight of edge : "<<weight<<endl;
    if(weight != 0) {
      //cout<<"Edge from "<<v<<" to "<<neighbor<<endl;
      count++;
    }
    ////cout<<"Weight of reverse edge : "<<getEdgeWeight(neighbor, v)<<endl;
    if(edge_map[neighbor][v] != 0) {
      //cout<<"Edge from "<<neighbor<<" to "<<v<<endl;
      count++;
    }
    //cout<<"Erasing "<<v<<" from neighbor ";
    edge_map[neighbor].erase(v);
    //cout<<neighbor<<endl;
    if(edge_map[neighbor].size()==0)
      edge_map.erase(neighbor);
    //cout<<"Done with neighbor "<<neighbor<<endl;
  }
  //cout<<"Am I here?"<<endl;
  edge_map.erase(v);
  numOfEdges = numOfEdges - count;
  //cout <<"Erasing 2 "<<v<<endl;
  vertices.erase(v);
  //cout <<"Erased 2 "<<v<<endl;
  //cout<<"Deleted "<<v<<endl;
}

