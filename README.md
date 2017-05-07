# Mamut_xml_processing

## Overview

This repository contains code developed to analyze the results of cell tracking with [MaMuT](http://imagej.net/MaMuT). The data originally used for this project of ~17 hours of *Drosophila* embryonic development was collected in the [Keller Lab](https://www.janelia.org/lab/keller-lab). For an introduction to this code and examples of its visualizations, please see [this introductory Jupyter notebook](Track Processing and Analysis Example.ipynb).

## *mamut* class

Initializing the *mamut* class begins the process of parsing and formatting MaMuT xml data. Each track is defined as all the position data generated by selecting a node with no successor (an end node) and tracking all the way backwards to a node with no predecessor (a start node). Once the node ids of all nodes in a track have been compiled (*trackFromEnd*), the corresponding position data is compiled generating a dictionary of all tracks with track key containing a list of positions [0] and a list of ids [1] (*createArrayTracks*). Tracks that are incomplete will have nans replacing missing position data and zeros replacing missing ids.

Next, all complete tracks will be identified and compiled into a dictionary by *createCompleteTracks*. This track data can be smoothed by *smoothAllTracks* with a specified window for smoothing. Local movement of cells can be extracted by subtracting the average trajectory of all cells from the movement of each cell (*createLocalTracks*). 

Finally, lineages are identified and compiled into a dictionary which lists all tracks within a lineage for each lineages (*findLineages*, *nodesInLineages*). A lineage is defined as a single start cell with all its daughter cells or in this case tracks.

## Data Visualization

Cell movement can be visualized using three different tools. gt.graph3dTracks plots each track in the dataset in 3D with the random color assigned by gt.randomColor.

Similarly, gt.threeD_gradient plots tracks in the dataset in 3D with a color gradient along each line corresponding to time. To visualize the time gradient in 2D, gt.twoD_gradient plots three perspectives (Lateral, DV, and AP) on images at a specified time point.

## Velocity

## Position Fatemap

The positionFatemap class has functions associated with it to plot the position of cell tracks at specified time points according to a specified color scheme. In the case of this example, the color scheme is an anterior to poster gradient calculated based on the position of the lineage at the final time point.

In the example here, the position of cells is plotted on top of images in three different perspectives: Lateral (XY), Dorsal-Ventral (XZ), and Anterior-Posterior (YZ). These images are maximum intensity projections created from 3D stacks and cropped to both focus on the region of interest and limit unnecessary signal in the MIP. For the XY MIP, the volume was cropped so that only the brain lobe of interest contributed to the MIP. For the XZ MIP, the volume was cropped to exclude the VNC. Finally for the YZ MIP, the volume was cropped to include only the anterior half of the embyro.

All images were produced manually and are saved in three folders corresponding to the perspective within the Image_Data folder. The path to this directory is specified under 'Script Set Up'.

As a result of this cropping, the coordinate system of the cell tracks does not directly align with the coordinate system of the images. This shift can be corrected by adding or subtracting the amount of cropping from the position values for each cell. Within the module 'image_finder', there are three dictionaries contining a list of information with the index of the position data needed for graphing: eg. XY = (0,1). Additionally, this dictionary contains information about the shift required to correct the coordinate system. The function findImage returns the path to the image at the correct perspective and time as well as returning the list of information described above.

## Movement Based Clustering 

The distAnalysis class attempts to characterize types of cellular movement by considering the distances between tracks. The euclidean distance between each pair of tracks at every time point is calculated by the *calculateDist* function. If the distance between the tracks is consistent, the tracks are considered to be moving in a similar way. If the distance varies, tracks are considered to be moving differently. To reduce this calculation to a single metric, the average derivative is calculated for the distance between each pair of tracks by the function *distVarianceSlope*. 

The clusterVarslope function takes the average derivative value computed for each pair of tracks and uses the Ward method of hierarchical clustering to generate the linkage needed for the figure shown above. Individual clusters can be better identified by applying a color threhold at the y value of interest as can be seen below.

The track ids of each cluster can be extracting using the extractClusters function. The input to this function determines the number of output clusters, which can be determined based on the colored cluster figure generated above. The average movement of each cluster can be visualized using the *graphClusterAvg* function with an input of an assigned color for each cluster.

## Neighborhood Analysis 

The *neighbors* class initiates by calculating for each cell the distance to every other cell in the track input at each time point. Next, the nearest neighbors for each cell and time point are selected based on the input to the class at the second position. A representation of the nearest neighbors of a cell can be visualized by creating a scatterplot of the neighbors at each time point over time using the *graphNearestNBs*. 

This representation of nearest neighbors is interesting for looking at individual cells, but it does not give a sense of neighborhood dynamics on the scale of the whole brain. The windowConsistency function seeks to represent neighborhood dynamics over time to give a global representation on the scale of the whole dataset.

The windowConsistency function take an input that determines the size of the window for this function. In this case it determines the number of time points on either side of the window: e.g. an input of 20 produces a window of 41. Within the window, the neighborhood of every timepoint is compared against every other timepoint. For each comparison, the fraction of common neighbors is computed and the average common neighbor is calculated for all comparisons within the window. This average is the 'Neighborhood Consistency Score'.
