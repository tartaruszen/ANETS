#include "Edge.h"


Edge::Edge(int v1,int v2, double w)
{
	V1_ID=v1;
	V2_ID=v2;
	weight=w;
	isDirectedEdge=false;
}

Edge::Edge(int v1,int v2, double w,bool directedEdge)
{
	V1_ID=v1;
	V2_ID=v2;
	weight=w;
	isDirectedEdge=directedEdge;
}

int Edge::getFirstVertexID(){
	return V1_ID;
}

int Edge::getSecondVertexID(){
	return V2_ID;
}

double Edge::getEdgeWeight(){
	return weight;
}

bool Edge::equals(const Edge& e){
	
	if(!isDirectedEdge &&(
		(e.V1_ID==V1_ID && e.V2_ID==V2_ID)||(e.V1_ID==V2_ID && e.V2_ID==V1_ID))){
			return true;
	}

	if(isDirectedEdge && (e.V1_ID==V1_ID && e.V2_ID==V2_ID)){
		return true;
	}

	return false;
}

void Edge::setFirstVertexID(int v1){
	V1_ID=v1;
}

void Edge::setSecondVertexID(int v2){
	V2_ID=v2;
}

void Edge::setEdgeWeight(double w){
	weight=w;
}

bool Edge::operator==(const Edge& e){
	return equals(e);
}

bool Edge::operator!=(const Edge& e){
	return !equals(e);
}
