#pragma once

class Edge
{
public:
	Edge(int v1,int v2, double w);
	Edge(int v1,int v2, double w, bool directedEdge);
	int getFirstVertexID();
	int getSecondVertexID();
	double getEdgeWeight();
	bool equals(const Edge& e) ;
	void setFirstVertexID(int v1);
	void setSecondVertexID(int v2);
	void setEdgeWeight(double w);
	bool operator==(const Edge& e) ;
	bool operator!=(const Edge& e) ;

private:
	bool isDirectedEdge;
	int V1_ID;
	int V2_ID;
	double weight;
};


