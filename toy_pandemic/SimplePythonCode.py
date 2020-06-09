#Importing the libraries we will use

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv
from collections import defaultdict
from tqdm import trange

#Defining parameters for our simulation

np.random.seed(seed=17)

n=100 
p_inf=0.2 #Probability of infection
epochs=15 
infected_history=[1] #We start with just one infected person.
new_infections=[]

#NetworkX provides lots of built-in graphs. 
#To understand how to create one from scratch, our simulation begins with a randomly generated
#graph. This can also be done using the graph generators from the nx library. 
#We create a graph with approx. n edges.

def GenerateRandomGraph():
    '''
    Generate a graph with n nodes and approximately n edges
    
    returns:
    :G: a graph with n nodes and n edges
    '''
    
    G=nx.Graph()
    nodes=[*range(n)]
    edges=[*zip(np.random.randint(0,n,n),np.random.randint(0,n,n))]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return(G)
    
#Each node in the graph can have metadata (a dictionary of info.). In our case,
#this metadata will be if the node is Infected or not. The color of the node will indicate this. 
#Blue = healthy, red = infected.
#We insert this in the graph using the following function.

def Populate(G):
    '''
    Fill in the metadata for the nodes of the graph G
    
    params:
    :G: a graph
    '''
    
    for n in G.nodes():
        G.nodes[n]['I']='blue'
        
#Let us plot our graph to see how it turned out.

G=GenerateRandomGraph()
Populate(G)

nx.draw_networkx(G,node_color='blue',node_size=20,with_labels=False)
plt.title('Our population and its connections.')
plt.show()

infection_seed=np.random.randint(len(G.nodes()))
G.nodes[infection_seed]['I']='red'

#To plot the network we passed a color argument to nx.draw(). Since now we will have nodes
#with different colors, let us create a function to return a list of colors.

def Color(G):
    '''
    Generates a list of colors from the edges' colors.
    
    params:
    :G: a graph
    
    returns:
    :color: a list of blue/red colors, symbolizing healthy and infected nodes.
    '''
    
    color=[n[1]['I'] for n in [*G.nodes(data=True)]]
    return(color)

#Plotting again

nx.draw_networkx(G,node_color=Color(G),node_size=20,with_labels=False)
plt.title('Our population and its connections with one infection.')
plt.show()

infection_seed=np.random.randint(len(G.nodes()))
G.nodes[infection_seed]['I']='red'

#To plot the network we passed a color argument to nx.draw(). Since now we will have nodes
#with different colors, let us create a function to return a list of colors.

def Color(G):
    '''
    Generates a list of colors from the edges' colors.
    
    params:
    :G: a graph
    
    returns:
    :color: a list of blue/red colors, symbolizing healthy and infected nodes.
    '''
    
    color=[n[1]['I'] for n in [*G.nodes(data=True)]]
    return(color)

#Plotting again

nx.draw_networkx(G,node_color=Color(G),node_size=20,with_labels=False)
plt.title('Our population and its connections with one infection.')
plt.show()

def Update(G):
    '''
    Update the graph by:
    - Scanning the nodes;
    - Seeing if a node's neighbor is infected;
    - Seeing if this neighbor infects the node;
    - The newly infected nodes are collected in a list and are updated at 
    the end to avoid themselves being infectious at this iteration.
    
    params:
    :G: a graph to be updated
    
    returns:
    :n_new: the number of new infected nodes during one iteration
    '''
    
    new_infected_nodes=[]
    s_nodes = [n for n,d in G.nodes(data=True) if d['I']=='blue']
    i_nodes = [n for n,d in G.nodes(data=True) if d['I']=='red']
    for n in s_nodes:
        for m in G.neighbors(n):
            if m in i_nodes and np.random.choice([False,True],p=[1-p_inf,p_inf]):
                new_infected_nodes.append(n)
    for n in new_infected_nodes:
        G.nodes[n]['I']='red'
    n_new=len(set(new_infected_nodes))
    return(n_new)
        
def CountInfected(G):
    '''
    Counts the number of infected people in the graph.
    
    params:
    :G: a graph
    
    returns:
    the number of infected people in the graph
    '''
    
    return(len([n for n,d in G.nodes(data=True) if d['I']=='red']))
    
def FixPosition(G):
    '''
    Creates a quasi-rectangular grid to fix the position of the nodes
    of the graph when using nx.draw_networkx.
    
    params:
    :G: a graph
    
    returns:
    :pos: a lis
    t of (x,y) coordinates in a quasi-rectangular grid
    '''
    size=int(math.sqrt(len(G.nodes())))
    markers=[]
    for i in range(0,size+1):
        for j in range(0,size):
            markers.append([i+np.random.normal(0,0.1),j+np.random.normal(0,0.1)])
    pos=defaultdict(list)
    for n in [*G.nodes()]:
        try:
            pos[n]=markers[n]
        except:
            pass
    return(pos)

pos=FixPosition(G)

nx.draw_networkx(G,node_color=Color(G),node_size=20,with_labels=False,pos=pos)
plt.title('Our population and its connections with one infection.')
plt.show()

def Simulate(G,plot_all=False,plot_final=True):
    '''
    Runs the simulation outlined in the notebook, iterating epoch times.
    
    params:
    :G: a graph
    :plot_all: a boolean indicating whether or not to draw all steps in the evolution of G
    :plot_final: a boolean indicating wheter or not to draw the final result of the simulation.
    
    '''
    
    if plot_all:
        fig,ax=plt.subplots(int(epochs/5),5,figsize=(20,10))
        i,j=0,0
    
    t0=len(infected_history)-1
    for t in trange(epochs):
        new_infections.append(Update(G))
        infected_history.append(CountInfected(G))
        if plot_all:
            nx.draw_networkx(G,node_color=Color(G),ax=ax[i][j],node_size=20,with_labels=False,pos=pos)
            ax[i][j].set_title(f"t={t0}+{t}")
            j+=1
            if (j)%5==0:
                i+=1
                j=0

    infected_percentage=infected_history[-1]*100/(len(G.nodes()))
    if plot_final:
        k=3
    else:
        k=2
    fig,ax=plt.subplots(1,k,figsize=(15,5))
    ax[0].plot(infected_history)
    pd.Series(new_infections).plot(kind='bar',ax=ax[1],rot=45)
    if plot_final:
        nx.draw_networkx(G,node_color=Color(G),
                         node_size=50,
                         with_labels=False,
                         ax=ax[k-1]) #We do not fix the position here to get a more presentable graph.
        ax[k-1].set_title("% infected at the end: {0:.2f}%".format(infected_percentage))
    plt.show()
    
Simulate(G,plot_all=True,plot_final=True)
Simulate(G,plot_all=True,plot_final=True)	

G1=nx.relaxed_caveman_graph(10,6,0.3,seed=17)

Populate(G1)
infection_seed=np.random.randint(len(G1.nodes()))
G1.nodes[infection_seed]['I']='red'

nx.draw_networkx(G1,node_color=Color(G1),with_labels=False,node_size=30)

G2=nx.windmill_graph(4, 5)

Populate(G2)
infection_seed=np.random.randint(len(G2.nodes()))
G2.nodes[infection_seed]['I']='red'

nx.draw_networkx(G2,node_color=Color(G2),with_labels=False,node_size=30)

new_infections=[]
infected_history=[1]

Simulate(G1,plot_all=True)

new_infections=[]
infected_history=[1]

Simulate(G2,plot_all=True)

data=pd.read_csv('./facebook_combined.txt',header=None,sep=' ')
edges=[]

for row in data.itertuples(index=False):
    edges.append((row[0],row[1]))
    
FbG=nx.Graph()
FbG.add_edges_from(edges)

Populate(FbG)
infection_seed=np.random.randint(len(FbG.nodes()))
FbG.nodes[infection_seed]['I']='red'
new_infections=[]
infected_history=[1]

Simulate(FbG,plot_all=False,plot_final=False)

#Saving the previous data

infected_history_no_dist=infected_history
new_infections_no_dist=new_infections

#Each time we add an edge now we give it a 60% chance of being deleted.

edges=[]

for row in dataa.itertuples(index=False):
    edges.append((row[0],row[1]))
    if np.random.choice([True,False],p=[0.6,0.4]):
        del edges[-1]
 
FbG=nx.Graph()
FbG.add_edges_from(edges)

Populate(FbG)
FbG.nodes[infection_seed]['I']='red'
new_infections=[]
infected_history=[1]

Simulate(FbG,plot_all=False,plot_final=False)

plt.plot(infected_history_no_dist,label='no_dist')
plt.plot(infected_history,label='with_dist')
plt.legend()
plt.title('Comparison between total # of infected people with and without social distancing.')
plt.show()

fig,ax=plt.subplots(1,2,figsize=(20,5),sharey=True)
pd.Series(new_infections_no_dist).plot(kind='bar',label='no_dist',color='red',ax=ax[0])
ax[0].set_title('# of new infected people without social distancing.')
pd.Series(new_infections).plot(kind='bar',label='with_dist',color='blue',alpha=0.4,ax=ax[1])
ax[1].set_title('# of new infected people with social distancing.')

plt.show()
