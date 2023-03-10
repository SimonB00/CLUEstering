import pandas as pd
import numpy as np
import random as rnd
import time
import matplotlib.pyplot as plt
import CLUEsteringCPP as Algo 
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from math import sqrt
from tqdm import tqdm

def sign():
    sign = rnd.random()
    if sign > 0.5:
        return 1
    else:
        return -1

def findCentroid(points):
    """
    Function that finds the centroids
    """
    centroid = []
    for i in range(len(points[0])):
        centroid_i = 0
        for coord in points:
            centroid_i += coord[i]
        centroid.append(centroid_i/len(coord))
    return centroid    
        
def makeBlobs(nSamples, Ndim, nBlobs=4, mean=0, sigma=0.5, x_max=15, y_max=15):
    """
    Returns a test dataframe containing randomly generated 2-dimensional or 3-dimensional blobs. 

    Parameters:
    nSamples (int): The number
    Ndim (int): The number of dimensions.
    nBlobs (int): The number of blobs that should be produced. By default it is set to 4.
    mean (float): The mean of the gaussian distribution of the z values.
    sigma (float): The standard deviation of the gaussian distribution of the z values.
    x_max (flaot): Limit of the space where the blobs are created in the x direction.
    y_max (flaot): Limit of the space where the blobs are created in the y direction.
    """

    try:
        if x_max < 0. or y_max < 0.:
            raise ValueError('Error: wrong parameter value\nx_max and y_max must be positive')
        if nBlobs < 0:
            raise ValueError('Error: wrong parameter value\nThe number of blobs must be positive')
        if mean < 0. or sigma < 0.:
            raise ValueError('Error: wrong parameter value\nThe mean and sigma of the blobs must non negative')

        sqrtSamples = sqrt(nSamples)
        centers = []
        if Ndim == 2:
            data = {'x0': [], 'x1': [], 'weight': []}
            for i in range(nBlobs):
                centers.append([sign()*x_max*rnd.random(),sign()*y_max*rnd.random()])
            blob_data, blob_labels = make_blobs(n_samples=nSamples,centers=np.array(centers))
            for i in range(nSamples):
                data['x0'] += [blob_data[i][0]]
                data['x1'] += [blob_data[i][1]]
                data['weight'] += [1]

            return pd.DataFrame(data)
        elif Ndim == 3:
            data = {'x0': [], 'x1': [], 'x2': [], 'weight': []}
            z = np.random.normal(mean,sigma,sqrtSamples)
            for i in range(nBlobs):
                centers.append([sign()*x_max*rnd.random(),sign()*x_max*rnd.random()]) # the centers are 2D because we create them for each layer
            for value in z: # for every z value, a layer is generated.
                blob_data, blob_labels = make_blobs(n_samples=sqrtSamples,centers=np.array(centers))
                for i in range(nSamples):
                    data['x0'] += [blob_data[i][0]]
                    data['x1'] += [blob_data[i][1]]
                    data['x2'] += [value]
                    data['weight'] += [1]

            return pd.DataFrame(data)
        else:
            raise ValueError('Error: wrong number of dimensions\nBlobs can only be generated in 2 or 3 dimensions')
    except ValueError as ve:
        print(ve)
        exit()

class clusterer:
    def __init__(self, dc, rhoc, outlier, pPBin=10): 
        try:
            if float(dc) != dc:
                raise ValueError('Error: wrong parameter type\nThe dc parameter must be a float')
            self.dc = dc
            if float(rhoc) != rhoc:
                raise ValueError('Error: wrong parameter type\nThe rhoc parameter must be a float')
            self.rhoc = rhoc
            if float(outlier) != outlier:
                raise ValueError('Error: wrong parameter type\nThe outlier parameter must be a float')
            self.outlier = outlier
            if type(pPBin) != int:
                raise ValueError('Error: wrong parameter type\nThe pPBin parameter must be a int')
            self.pPBin = pPBin
        except ValueError as ve:
            print(ve)
            exit()
        self.kernel = Algo.flatKernel(0.5)
            
    def readData(self, inputData):
        """
        Reads the data in input and fills the class members containing the coordinates of the points, the energy weight, the number of dimensions and the number of points.

        Parameters:
        inputData (pandas dataframe): The dataframe should contain one column for every coordinate, each one called 'x*', and one column for the weight.
        inputData (string): The string should contain the full path to a csv file containing the data.
        inputData (list or numpy array): The list or numpy array should contain a list of lists for the coordinates and a list for the weight.
        """

        #print('Start reading points')
        
        # numpy array
        if type(inputData) == np.array:
            try:
                if len(inputData) < 2:
                    raise ValueError('Error: inadequate data\nThe data must contain at least one coordinate and the energy.')
                self.coords = inputData[:-1]
                self.weight = inputData[-1]
                if len(inputData[:-1]) > 10:
                    raise ValueError('Error: inadequate data\nThe maximum number of dimensions supported is 10')
                self.Ndim = len(self.coords)
                self.Npoints = self.weight.size
            except ValueError as ve:
                print(ve)
                exit()

        # lists
        if type(inputData) == list:
            try:
                if len(inputData) < 2:
                    raise ValueError('Error: inadequate data\nThe data must contain at least one coordinate and the energy.')
                self.coords = np.array(inputData[:-1])
                self.weight = np.array(inputData[-1])
                if len(inputData[:-1]) > 10:
                    raise ValueError('Error: inadequate data\nThe maximum number of dimensions supported is 10')
                self.Ndim = len(self.coords)
                self.Npoints = self.weight.size
            except ValueError as ve:
                print(ve)
                exit()

        # path to .csv file or pandas dataframe
        if type(inputData) == str or type(inputData) == pd.DataFrame:
            if type(inputData) == str:
                try:
                    if inputData[-3:] != 'csv':
                        raise ValueError('Error: wrong type of file\nThe file is not a csv file.')
                    df = pd.read_csv(inputData)
                except ValueError as ve:
                    print(ve)
                    exit()
            if type(inputData) == pd.DataFrame:
                try:
                    if len(inputData.columns) < 2:
                        raise ValueError('Error: inadequate data\nThe data must contain at least one coordinate and the energy.')
                    df = inputData
                except ValueError as ve:
                    print(ve)
                    exit()

            try:
                if not 'weight' in df.columns:
                    raise ValueError('Error: inadequate data\nThe input dataframe must contain a weight column.')
                    
                coordinate_columns = [col for col in df.columns if col[0] == 'x']
                if len(coordinate_columns) > 10:    
                    raise ValueError('Error: inadequate data\nThe maximum number of dimensions supported is 10')
                self.Ndim = len(coordinate_columns)
                self.Npoints = len(df.index)
                self.coords = np.zeros(shape=(self.Ndim, self.Npoints))
                for dim in range(self.Ndim):
                    self.coords[dim] = np.array(df.iloc[:,dim])
                self.weight = df['weight']
            except ValueError as ve:
                print(ve)
                exit()

        #print('Finished reading points')

    def chooseKernel(self, choice, parameters=[], function = lambda : 0):
        """
        Changes the kernel used in the calculation of local density. The default kernel is a flat kernel with parameter 0.5

        Parameters:
        choice (string): The type of kernel that you want to choose (flat, exp, gaus or custom).
        parameters (list or np.array): List of the parameters needed by the kernels. The flat kernel requires one, 
        the exponential requires two (amplutude and mean), the gaussian requires three (amplitude, mean and standard deviation)
        and the custom doesn't require any, so an empty list should be passed.
        function (function object): Function that should be used as kernel when the custom kernel is chosen.
        """

        try:
            if choice == "flat":
                if len(parameters) != 1:
                    raise ValueError('Error: wrong number of parameters\nThe flat kernel requires 1 parameter')
                self.kernel = Algo.flatKernel(parameters[0])
            elif choice == "exp":
                if len(parameters) != 2:
                    raise ValueError('Error: wrong number of parameters\nThe exponential kernel requires 2 parameters')
                self.kernel = Algo.exponentialKernel(parameters[0], parameters[1])
            elif choice == "gaus":
                if len(parameters) != 3:
                    raise ValueError('Error: wrong number of parameters\nThe gaussian kernel requires 3 parameters')
                self.kernel = Algo.gaussianKernel(parameters[0], parameters[1], parameters[2])
            elif choice == "custom":
                if len(parameters) != 0:
                    raise ValueError('Error: wrong number of parameters\nCustom kernels requires 0 parameters')
                self.kernel = Algo.customKernel(function)
            else: 
                raise ValueError('Error: invalid kernel\nThe allowed choices for the kernels are: flat, exp, gaus and custom')
        except ValueError as ve:
            print(ve)
            exit()
    
    def parameterTuning(self, nRun=2000):
        # Calculate mean and standard deviations in all the coordinates
        means = np.zeros(shape=(self.Ndim, 1))
        covariance_matrix = np.cov(self.coords)
        for dim in range(self.Ndim):
            means[dim] = np.mean(self.coords[dim])
        
        # Normalize all the coordinates as x'_j = (x_j - mu_j) / sigma_j
        for dim in range(self.Ndim):
            self.coords[dim] = (self.coords[dim] - means[dim]) / sqrt(covariance_matrix[dim][dim])

        max_nclusters = 0
        data = np.zeros(shape=(nRun, 4))
        for i in tqdm(range(nRun)):
            self.dc = np.random.uniform(0., 1.5) # This range is general because of the normalization of the coordinates
            self.rhoc = np.random.uniform(0., self.Npoints)
            self.outlier = np.random.uniform(1., 3.)
            self.runCLUE()
            if self.NClusters > max_nclusters:
                max_nclusters = self.NClusters
            data[i] = np.array([self.dc, self.rhoc, self.outlier, self.NClusters])
        print(np.unique(data.T[3]))
        
        # Get the obtained values for the number of clusters and remove isolated values
        nclusters_values = np.unique(data.T[3])
        for value in range(len(nclusters_values) - 1):
            if nclusters_values[value + 1] > nclusters_values[value] + 1:
                data = np.delete(data, [i for i in range(len(data)) if data[i][3] >= value+1], 0)
                break
        max_nclusters = max(data.T[3])
        print(max_nclusters)

        plot_data = go.Scatter3d(x=data.T[0], y=data.T[1], z=data.T[3], mode='markers')
        fig = go.Figure(plot_data)
        fig.update_layout(scene = dict(xaxis=dict(range=[0.,1.5]), yaxis=dict(range=[0.,self.Npoints]), zaxis=dict(range=[0,max_nclusters]), 
                    xaxis_title='dc', yaxis_title='rhoc', zaxis_title='nClusters'))
        fig.update_traces(marker_size = 3)
        fig.show()

        #self.dc = data[max_i][0]
        #self.rhoc = data[max_i][1]
        #self.outlier = data[max_i][2]
        #print('The number of clusters is: ', data[max_i][3])

    def runCLUE(self, verbose=False):
        """
        Executes the CLUE clustering algorithm.

        Parameters:
        verbose (bool): The verbose option prints the execution time of runCLUE and the number of clusters found

        Output:
        self.clusterIds (list): Contains the clusterId corresponding to every point.
        self.isSeed (list): For every point the value is 1 if the point is a seed or an outlier and 0 if it isn't.
        self.NClusters (int): Number of clusters reconstructed.
        self.clusterPoints (list): Contains, for every cluster, the list of points associated to id.
        self.pointsPerCluster (list): Contains the number of points associated to every cluster
        """

        start = time.time_ns()
        clusterIdIsSeed = Algo.mainRun(self.dc,self.rhoc,self.outlier,self.pPBin,self.kernel,self.coords,self.weight,self.Ndim)
        finish = time.time_ns()
        self.clusterIds = np.array(clusterIdIsSeed[0])
        self.isSeed = np.array(clusterIdIsSeed[1])
        self.NClusters = len(np.unique(self.clusterIds))

        clusterPoints = [[] for i in range(self.NClusters)]
        for i in range(self.Npoints):
            clusterPoints[self.clusterIds[i]].append(i)

        self.clusterPoints = clusterPoints
        self.pointsPerCluster = np.array([len(clust) for clust in clusterPoints])

        data = {'clusterIds': self.clusterIds, 'isSeed': self.isSeed}
        self.outputDF = pd.DataFrame(data) 

        self.elapsed_time = (finish - start)/(10**6)
        if verbose:
            print('CLUE run in ' + str(self.elapsed_time) + ' ms')
            print('Number of clusters found: ', self.NClusters)

    def inputPlotter(self, plot_title='', label_size=16, pt_size=1, pt_colour='b'):
        """
        Plots the the points in input.

        Parameters:
        plot_title (string): Title of the plot
        label_size (int): Fontsize of the axis labels
        pt_size (int): Size of the points in the plot
        pt_colour (string): Colour of the points in the plot
        """

        if self.Ndim == 2:
            plt.scatter(self.coords[0],self.coords[1], s=pt_size, color=pt_colour)
            plt.title(plot_title)
            plt.xlabel('x', fontsize=label_size)
            plt.ylabel('y', fontsize=label_size)
            plt.show()
        if self.Ndim >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.coords[0],self.coords[1],self.coords[2], s=pt_size, color=pt_colour)
            ax.set_title(plot_title)
            ax.set_xlabel('x', fontsize=label_size)
            ax.set_ylabel('y', fontsize=label_size)
            ax.set_zlabel('z', fontsize=label_size)
            plt.show()

    def clusterPlotter(self, plot_title='', label_size=16, outl_size=10, pt_size=10, seed_size=25):
        """
        Plots the clusters with a different colour for every cluster. 

        The points assigned to a cluster are prints as points, the seeds as stars and the outliers as little grey crosses. 

        Parameters:
        plot_title (string): Title of the plot
        label_size (int): Fontsize of the axis labels
        outl_size (int): Size of the outliers in the plot
        pt_size (int): Size of the points in the plot
        seed_size (int): Size of the seeds in the plot
        """
        
        if self.Ndim == 2:
            data = {'x0':self.coords[0], 'x1':self.coords[1], 'clusterIds':self.clusterIds, 'isSeed':self.isSeed}
            df = pd.DataFrame(data)

            df_clindex = df["clusterIds"]
            M = max(df_clindex) 
            dfs = df["isSeed"]

            df_out = df[df.clusterIds == -1] # Outliers
            plt.scatter(df_out.x0, df_out.x1, s=outl_size, marker='x', color='0.4')
            for i in range(0,M+1):
                dfi = df[df.clusterIds == i] # ith cluster
                plt.scatter(dfi.x0, dfi.x1, s=pt_size, marker='.')
            df_seed = df[df.isSeed == 1] # Only Seeds
            plt.scatter(df_seed.x0, df_seed.x1, s=seed_size, color='r', marker='*')
            plt.title(plot_title)
            plt.xlabel('x', fontsize=label_size)
            plt.ylabel('y', fontsize=label_size)
            plt.show()
        if self.Ndim == 3:
            data = {'x0':self.coords[0], 'x1':self.coords[1], 'x2':self.coords[2], 'clusterIds':self.clusterIds, 'isSeed':self.isSeed}
            df = pd.DataFrame(data)

            df_clindex = df["clusterIds"]
            M = max(df_clindex) 
            dfs = df["isSeed"]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            df_out = df[df.clusterIds == -1]
            ax.scatter(df_out.x0, df_out.x1, df_out.x2, s=outl_size, color = 'grey', marker = 'x')
            for i in range(0,M+1):
                dfi = df[df.clusterIds == i]
                ax.scatter(dfi.x0, dfi.x1, dfi.x2, s=pt_size, marker = '.')

            df_seed = df[df.isSeed == 1] # Only Seeds
            ax.scatter(df_seed.x0, df_seed.x1, df_seed.x2, s=seed_size, color = 'r', marker = '*')
            ax.set_title(plot_title)
            ax.set_xlabel('x', fontsize=label_size)
            ax.set_ylabel('y', fontsize=label_size)
            ax.set_zlabel('z', fontsize=label_size)
            plt.show()

    def toCSV(self,outputFolder,fileName):
        """
        Creates a file containing the coordinates of all the points, their clusterIds and isSeed.   

        Parameters: 
        outputFolder (string): Full path to the desired ouput folder.
        fileName (string): Name of the file, with the '.csv' suffix.
        """

        outPath = outputFolder + fileName
        data = {}
        for i in range(self.Ndim):
            data['x' + str(i)] = self.coords[i]
        data['clusterIds'] = self.clusterIds
        data['isSeed'] = self.isSeed

        df = pd.DataFrame(data)
        df.to_csv(outPath,index=False)
