// Authors: Manish Purohit, Chanhyun Kang and Yao Zhang.
#include "graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <ctime>
#include "DisjointSets.h"
using namespace std;

list<Edge> readMergingList(string fileName);

void printElementSets(const DisjointSets & s)
{
  for (int i = 0; i < s.NumElements(); ++i)
    cout << i << " : "<<s.FindSet(i) << ",  ";
  cout << endl;
}

int main(int argc, char* argv[])
{
  string usage = "Usage: executable_file graph.inf to_merge_list percent threshold coarse_graph_file final_map_file time_file";
  
  if (argc != 8) {
    cout<<usage<<endl;
    return 1;
  }
  
  string graphFile=argv[1]; //Read arguments
  string toMergeFile=argv[2];
  double percent=atof(argv[3]);
  //double threshold = 1.0/atof(argv[4]); TODO!!! This is only temporary
  double threshold = atof(argv[4]);
  //double epsilon = 1e-7;
  //cout<<"epsilon: "<<epsilon<<endl;
  //double threshold = 0;
  string coarseFile = argv[5];
  string finalmapFile = argv[6];
  string timeFile = argv[7];
  
  Graph graph (graphFile); //Create graph
  unordered_set<int> verticesALL=graph.getVertices();
  
  list<Edge> mList=readMergingList(toMergeFile);
  
  //int limit =(int)((mList.size()*percent)/100);
  int limit = (int)((graph.getNumOfVertices()*percent)/100);
  //cout<<"Total edges to merge = "<<mList.size()<<endl;
  //cout<<"limit: "<<limit<<endl;
  list<int>::iterator i;
  list<Edge>::iterator it;
  
  DisjointSets intermediate_map(verticesALL.size()+1);
  //  printElementSets(intermediate_map);
  unordered_map <int, list<int> > FinalMapping;
  
  int edge_num = 0;
  clock_t Start_time =clock();
  for(it=mList.begin(); it != mList.end(); ++it) {
    Edge mergingEdge= *it;
    if(edge_num > limit) {
      break;
    }
    
    int v1=mergingEdge.getFirstVertexID();
    int v2=mergingEdge.getSecondVertexID();
    
    //cout<<"contracting "<<v1<<" "<<v2<<endl;

    int a = intermediate_map.FindSet(v1);
    int b = intermediate_map.FindSet(v2);
    
    if(a == b) {
      //cout<<"a == b"<<endl;
      continue;
    }
    double b1 = graph.getEdgeWeight(a, b);
    double b2 = graph.getEdgeWeight(b, a);
    if(b1 == 0.0) {
      //cout<<"edge has 0 weight"<<endl;
      continue;
    }

    edge_num++;
    //cout<<"finding neighbors"<<endl;
    unordered_set<int> NeighborOfFromV;
    unordered_set<int> NeighborOfToV;
    
    int toV = intermediate_map.Union(a, b);
    int fromV = a==toV?b:a;

    //Checki if union occurred
    //cout<<"FindSet("<<fromV<<") = "<<intermediate_map.FindSet(fromV);
    //cout<<"FindSet("<<toV<<") = "<<intermediate_map.FindSet(toV);

    //printElementSets(intermediate_map);
    graph.getAllNeighborVerticesAsSet(fromV,&NeighborOfFromV);
    graph.getAllNeighborVerticesAsSet(toV,&NeighborOfToV);
    
    unordered_set<int> allNeighbors;
    allNeighbors.insert(NeighborOfFromV.begin(), NeighborOfFromV.end());
    allNeighbors.insert(NeighborOfToV.begin(), NeighborOfToV.end());
    
    //cout<<"Found neighbors"<<endl;

    for(unordered_set<int>::const_iterator i=allNeighbors.begin(); i!=allNeighbors.end(); ++i) {
      int nV = *i;
      //cout<<"neighbor:"<<nV<<", toV:"<<toV<<", fromV:"<<fromV<<endl;
      if(nV==fromV || nV==toV)
	continue;
      double aout = graph.getEdgeWeight(a, nV);
      double ain = graph.getEdgeWeight(nV, a);
      double bout = graph.getEdgeWeight(b, nV);
      double bin = graph.getEdgeWeight(nV, b);
      graph.deleteEdge(fromV, nV);
      graph.deleteEdge(nV, fromV);
      graph.deleteEdge(toV, nV);
      graph.deleteEdge(nV, toV);
      double newin = ((1+b1)*ain + (1+b2)*bin) / 2.0;
      double newout = ((1+b2)*aout + (1+b1)*bout)/2.0;
      if (ain > 0.0 && bin > 0.0)
	{	newin /= 2.0;}
      if (aout > 0.0 && bout > 0.0)
	{newout /= 2.0;}
      if(newin > threshold)
  graph.addEdge(nV, toV, newin);
      if(newout > threshold)
  graph.addEdge(toV, nV, newout);
//      if(newin < epsilon)
//          newin = epsilon;
//  graph.addEdge(nV, toV, newin);
//      if(newout < epsilon)
//          newout = epsilon;
//  graph.addEdge(toV, nV, newout);

    }
    //cout<<"Merged.. now only deleting left"<<endl;
    graph.deleteVertex(fromV);
    //cout<<"Edge_num :"<<edge_num<<", Vertices left: "<<graph.getNumOfVertices()<<endl;
    //cout<<"Done!"<<endl;
  }
  
 
  //cout<<"Maintain final mapping"<<endl;
  for(unordered_set<int>::const_iterator x = verticesALL.begin(); x!= verticesALL.end(); ++x){
    int u = intermediate_map.FindSet(*x);
    FinalMapping[u].push_back(*x);
    //cout<<"Added"<<*x<<" to FinalMapping["<<u<<"]"<<endl;
  }

  clock_t End_time =clock();
  
  ofstream myfile;
  //  string coarsefile = coarseFile;
  myfile.open (coarseFile.c_str());
  graph.writeToFile(myfile);
  myfile.close();
  
  unordered_set<int> verticesAfterCoarsening=graph.getVertices();
  
  ofstream myfile_map;
  //  string coarsefile_map = graphFile+"_coarse_map";
  myfile_map.open (finalmapFile.c_str());
 
  for(unordered_set<int>::const_iterator x = verticesAfterCoarsening.begin(); x!= verticesAfterCoarsening.end(); x++){
    //FinalMapping[*i]->second;
    list<int>::iterator iter;
    myfile_map<<*x<<" : ";	
    for(iter = (FinalMapping[*x]).begin(); iter != (FinalMapping[*x]).end(); ++iter) {
      myfile_map<<*iter<<",";
      
    }
    myfile_map<<endl;
  }
  myfile_map.close();
  

 ofstream myfile_time;
 //string coarsefile_time = graphFile+"_coarse_time";
  myfile_time.open (timeFile.c_str());
  myfile_time<<(double)((double)(End_time-Start_time)/(double)CLOCKS_PER_SEC)<<endl;
  myfile_time.close();
  

  return 0;
}


list<Edge> readMergingList(string fileName){
  
  string line;
  list<Edge> L;
  ifstream graphfile (fileName.c_str());
  int v1;
  int v2;
  double weight;
  if (graphfile.is_open())
    {
      while (getline(graphfile,line))
	{
	  // cout<<line<<endl; 
	  string to, from, last;
	  istringstream liness( line );
	  getline( liness, from, '\t' );
	  getline( liness, to,  '\t' );
	  getline( liness, last,   '\t');
	  // cout << line << endl;
	  v1 = atoi( from.c_str() );
	  v2 = atoi( to.c_str() );
 	  weight = atof(last.c_str() );
	  //	  cout<<"Read : "<<v1<<" "<<v2<<" "<<weight <<endl;
	  Edge e (v1,v2,weight,true);
	  L.push_back(e);
	}
      graphfile.close();
    }
  else cout << "Unable to open file"; 
  return L;
}

// int getMappedVertex(int v,unordered_map <int, int> *map){
  
//   int startV=v;
//   list<int> tempList;
  
//   unordered_map <int, int> :: const_iterator map_Iter;
//   while(true){
//     map_Iter=map->find(startV);
//     if(map_Iter==map->end()){
//       break;
//     }else{			
//       tempList.push_back(startV);
//       startV=map_Iter->second;			
//     }
//   }
//   list<int>::iterator i;
  
//   for(i=tempList.begin(); i != tempList.end(); ++i) {
//     (*map)[*i]=startV;
//   }
  
//   return startV; 
// }
