import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def haversine_distance(x,y, mode = 'degrees', R = 6371):
    '''
    Calculates great-circle ("haversine") distance between x with coordinates (lat1, lng1) and n points in y with coordinates (lat2, lng2).
    Coordinates are (latitude, longitude) pairs.
    We use the convention that latitudes are in [-90,90] degrees and longitudes are in (-180,180] degrees.
    We consider negative latitudes to be in the south direction and negative longitudes to be in the west direction.
    Note that the haversine distance between two distinct points is always 
    
    ======
    Inputs
    ======
    x : (n,2) np.array-like of float
        The (latitude, longitude) pairs of the first set of n points.
    y : (n,2) np.array-like of float
        The (latitude, longitude) pairs of the second set of n points.
    mode : str in ['radians','degrees']
        The units of the (latitude, longitude) coordinate values.
    R : float
        The radius of the sphere in kilometres (for Earth, this is 6371).
        
    =======
    Outputs
    =======
    d : (n,2) np.array-like of float
        The distances between the points in x and y.
        
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    #pdb.set_trace()
    ## EXCEPTION HANDLING
    if mode not in ['radians','degrees']:
        print('Warning: mode should be radians or degrees. Making it radians...')
        mode = 'radians'
    elif mode == 'degrees':
        # Convert values to radians for haversine distance calculation later
        x = x * np.pi/180
        y = y * np.pi/180

    ## CALCULATE HAVERSINE DISTANCE
    a = np.sin((y[0]-x[0])/2)**2 + np.cos(x[0])*np.cos(y[0])*(np.sin((y[1]-x[1])/2)**2)
    theta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) # Opposite/Adjacent
    d = R * theta # Arc of circle formula
    
    return d

def haversine_distance_vectorised(lat1, lng1, lat2, lng2, mode = 'degrees', R = 6371):
    '''
    Calculates distances between n points with coordinates (lat1, lng1) and n points with coordinates (lat2, lng2).
    Coordinates are (latitude, longitude) pairs.
    We use the convention that latitudes are in [-90,90] degrees and longitudes are in (-180,180] degrees.
    We consider negative latitudes to be in the south direction and negative longitudes to be in the west direction.
    
    ======
    Inputs
    ======
    lat1 : float or (n,) np.array-like of float
        The latitudes of the first set of n points.
    lng1 : float or (n,) np.array-like of float
        The longitudes of the first set of n points.
    lat2 : float or (n,) np.array-like of float
        The latitudes of the second set of n points.
    lng2 : float or (n,) np.array-like of float
        The longitudes of the second set of n points.
    mode : str in ['radians','degrees']
        The units of the (latitude, longitude) coordinate values.
    R : float
        The radius of the sphere in kilometres (for Earth, this is 6371).
        
    =======
    Outputs
    =======
    d : float or (n,) np.array-like of float
        The distances between the n points with coordinates (lat1, lng1) and coordinates (lat2, lng2).
        
    ============
    Dependencies
    ============
    numpy (as np)
    '''
    ## EXCEPTION HANDLING
    if mode not in ['radians','degrees']:
        print('Warning: mode should be radians or degrees. Making it radians...')
        mode = 'radians'
    elif mode == 'degrees':
        lat1 = lat1 * np.pi/180
        lng1 = lng1 * np.pi/180
        lat2 = lat2 * np.pi/180
        lng2 = lng2 * np.pi/180
    
    if not (lat1.shape == lng1.shape and lng1.shape == lat2.shape and lat2.shape == lng2.shape):
        print(f'Coordinate vectors have unequal dimensions! lat1.shape = {lat1.shape}, lng1.shape = {lng1.shape}, lat2.shape = {lat2.shape}, lng2.shape = {lng2.shape}')
    
    ## CALCULATE HAVERSINE DISTANCE
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lng2-lng1)/2)**2)
    theta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) # Opposite/Adjacent
    d = R * theta # Arc of circle formula
    
    return d

def weighted_kmeans(points = np.array([[ 1.0, 1.0],[ 1.0, 2.0],[ 2.0, 1.0],[ 2.0, 2.0],
                                       [-1.0,-1.0],[-1.0,-2.0],[-2.0,-1.0],[-2.0,-2.0],
                                       [-1.0, 1.0],[-1.0, 2.0],[-2.0, 1.0],[-2.0, 2.0],
                                       [ 1.0,-1.0],[ 1.0,-2.0],[ 2.0,-1.0],[ 2.0,-2.0]]),
                    weights = None,
                    k = 4,
                    metric = 'euclidean',
                    seed_val = None,
                    clinit = 'plusplus',
                    max_iter = 10,
                    verbose = True,
                    show_plots = True):
    '''
    Performs a weighted k-means clustering of points with a given distance metric.
    
    ======
    Inputs
    ======
    points : numpy.array of shape (n,p)
        The n points to be clustered, with each point having p dimensions for its coordinates.
    weights : numpy.array of shape (n,1) or None
        The weights (= "importance" or "confidence") given to each of the n points in points.
        If None, every point is given a weight of 1.
    k : int
        The number of clusters to split the points into. Must be a positive integer.
    metric : str or callable
        Any valid argument to the "metric" keyword argument of scipy.spatial.distance.cdist
    seed_val : int or None
        The fixed seed to feed to np.random.seed to ensure replicability of results.
        If None, then no fixed seed is set.
    clinit : str in ['plusplus','uniform']
        If 'plusplus', uses the k-means++ initialisation method for cluster centres (Arthur and Vassilvitskii, 2007).
        If 'unifrom', randomly picks initial cluster centres according to a uniform distribution on the points in points.
    max_iter : int
        The maximum number of iterations to take for the k-means algorithm.
    verbose : bool
        If True, prints status and error messages. If False, prints nothing.
    show_plots : bool
        If True, shows the plots of the points and cluster centres for each iteration of k-means. If False, shows nothing.

    =======
    Outputs
    =======
    centres : numpy.array of shape (k,p)
        The coordinates of the k cluster centres, with each point having p dimensions for its coordinates.
        In other words, centres[i,:] is the cluster centre of cluster i. Here, i is in {0,1,...,k-1}.
    clusters : numpy.array of shape (n,)
        The cluster that each point in points belongs to, with the cluster indices in {0,1,...,k-1}.
        In other words, clusters[i] is the cluster index of the point points[i,:].

    ============
    Dependencies
    ============
    numpy (as np), cdist (from scipy.spatial.distance)
    '''
    ## PREALLOCATION OF OUTPUTS
    n = points.shape[0]
    p = points.shape[1]

    ## EXCEPTION HANDLING
    if weights is None:
        weights = np.ones((n,1))
    if weights.shape[0] != n:
        if verbose: print(f'Error: Number of samples in points ({n}) does not match number of samples in weights ({weights.shape[0]})!')
        return
    if len(weights.shape) == 1: # Means its shape is actually (n,) instead of (n,1)
        if verbose: print(f'Warning: weights was entered as a zero-dimensional array. Expanding its dimension to 1...')
        weights = np.expand_dims(weights, axis = -1)

    ## SETTING OF RANDOM SEED
    if seed_val is not None:
        np.random.seed(seed_val)

    ## INITIALISATION OF CENTRES
    if clinit == 'plusplus':
        centres = np.full((k, p), np.inf) # Preallocate all centres as infinity; we will replace with actual centres later
        dist_from_centres_to_points = np.full((k,n), np.inf) # Preallocate all distances as infinity; we will replace rows with actual distances later
        P = np.ones(n)/n # Initial probability distribution P is uniform
        for i in range(k):
            sample_idx = np.random.choice(n, p = P) # Pick a random index based on the uniform distribution (if i==1) or the probability distribution P (if i > 1)
            centres[i,:] = points[sample_idx,:] # Add the coordinates of the point with that random index to centres
            dist_from_centres_to_points[i,:] = cdist(centres[[i],:], points, metric = metric) # Add the distance of that centre to every point in the distance matrix dist_from_centres_to_points
            D = np.min(dist_from_centres_to_points, axis = 0) # Get distances of points to their nearest centres (based on the centres chosen so far)
            if i < k-1:
                P = D**2 / np.sum(D**2) # Update the probability distribution P. But if we want to somehow choose as many clusters as there are points, prevent a true_divide error by not computing P for the last iteration.
    elif clinit == 'uniform':
        sample_idxs = np.random.permutation(n)[:k]
        centres = points[sample_idxs,:]
        dist_from_centres_to_points = cdist(centres, points, metric = metric)
    else:
        if verbose: print(f'Error: Invalid value entered for clinit!')
        return

    ## INITIALISATION OF CLUSTERS
    clusters = np.argmin(dist_from_centres_to_points, axis = 0) # Now, it's of shape (n,)

    ## EXECUTE WEIGHTED K-MEANS ALGORITHM
    for i in range(max_iter):
        if verbose: print(f'Now on iteration {i}/{max_iter}...', end = '\r')

        if show_plots:
            plt.plot(points[:,0],points[:,1], '.', markersize = 6)
            plt.plot(centres[:,0],centres[:,1], 'rx')
            plt.grid()
            plt.show()

        ## GET NEW CENTRES BASED ON CLUSTERS
        new_centres = np.full((k,p), np.inf) # Preallocate all centres as infinity; we will replace with actual centres later
        for j in range(k): # For each cluster...
            new_centres[j,:] = np.sum(points[clusters == j,:]*weights[clusters == j,:], axis = 0) / np.sum(weights[clusters == j,:]) # Calculate the weighted mean of points in cluster

        ## GET NEW CLUSTERS BASED ON NEW CENTRES
        dist_from_centres_to_points = cdist(new_centres, points, metric = metric)
        clusters = np.argmin(dist_from_centres_to_points, axis = 0)

        ## CHECK IF ALGORITHM HAS CONVERGED
        if np.all(centres == new_centres):
            if verbose: print(f'Algorithm has converged on iteration {i}/{max_iter}')
            break

        ## NEW CENTRES ARE NOW OLD CENTRES
        centres = np.copy(new_centres)
    
    return centres, clusters

def inertia(points,weights,centres,clusters,metric='euclidean'):
    '''
    Returns the inertia (i.e. weighted loss) value of clustered points (e.g. via a k-means algorithm), with respect to given cluster centres.
    
    ======
    Inputs
    ======
    points : numpy.array of shape (n,p)
        The n points to be clustered, with each point having p dimensions for its coordinates.
    weights : numpy.array of shape (n,1) or None
        The weights (= "importance" or "confidence") given to each of the n points in points.
    centres : numpy.array of shape (k,p)
        The coordinates of the k cluster centres, with each point having p dimensions for its coordinates.
        In other words, centres[i,:] is the cluster centre of cluster i. Here, i is in {0,1,...,k-1}.
    clusters : numpy.array of shape (n,)
        The cluster that each point in points belongs to, with the cluster indices in {0,1,...,k-1}.
        In other words, clusters[i] is the cluster index of the point points[i,:].
    metric : str or callable
        Any valid argument to the "metric" keyword argument of scipy.spatial.distance.cdist

    =======
    Outputs
    =======
    inertia_val : float
        The computed inertia value.

    ============
    Dependencies
    ============
    numpy (as np), cdist (from scipy.spatial.distance)
    '''
    k = max(clusters) + 1
    inertia_val = 0
    for i in range(k):
        inertia_val += np.sum(weights[clusters == i,:]*cdist(points[clusters == i,:], centres[[i],:], metric = metric)**2)
    return inertia_val

def dunn(points,clusters,metric='euclidean',verbose=False):
    '''
    Returns the Dunn index of clustered points, given the points and their clusters.
    
    ======
    Inputs
    ======
    points : numpy.array of shape (n,p)
        The n points to be clustered, with each point having p dimensions for its coordinates.
    clusters : numpy.array of shape (n,)
        The cluster that each point in points belongs to, with the cluster indices in {0,1,...,k-1}.
        In other words, clusters[i] is the cluster index of the point points[i,:].
    metric : str or callable
        Any valid argument to the "metric" keyword argument of scipy.spatial.distance.cdist
    verbose : bool
        If True, prints status and error messages. If False, prints nothing.

    =======
    Outputs
    =======
    dunn_index : float
        The computed Dunn index.

    ============
    Dependencies
    ============
    numpy (as np), cdist (from scipy.spatial.distance)
    '''

    dist_from_points_to_points = cdist(points, points, metric = metric) # Precalculate distance matrix for all points

    minimum_intercluster_distance = np.Inf # Initialise with maximum possible value, will decrease later.
    for i in range(max(clusters)):
        for j in range(i+1, max(clusters)+1):
            intercluster_distance = np.min(dist_from_points_to_points[np.ix_(clusters == i, clusters == j)]) # Get distance matrix entries between points in cluster i (rows) and points in cluster j (columns) only.
            if verbose: print(f'Intercluster distance between clusters {i} and {j}: {intercluster_distance} km.')

            if intercluster_distance < minimum_intercluster_distance:
                minimum_intercluster_distance = intercluster_distance

    maximum_intracluster_distance = 0 # Initialise with minimum possible value, will increase later.
    for i in range(max(clusters) + 1):
        intracluster_distance = np.max(dist_from_points_to_points[np.ix_(clusters == i, clusters == i)]) # Get distance matrix entries between points in cluster i (rows and columns) only.
        if verbose: print(f'Diameter of cluster {i}: {intracluster_distance} km.')

        if intracluster_distance > maximum_intracluster_distance:
            maximum_intracluster_distance = intracluster_distance

    dunn_index = minimum_intercluster_distance / maximum_intracluster_distance 
    return dunn_index

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def plot_results(points,centres=None,clusters=None,out_fname='',plot_title='',scale=0.5,show_plot=True,
                 plot_outliers=False,outlier_fmt='ko',plot_centres_first=False,
                 colors=['xkcd:aqua',
                         'xkcd:azure',
                         'xkcd:beige',
                         '#7fff00',
                         'xkcd:coral',
                         'xkcd:crimson',
                         'xkcd:darkgreen',
                         'xkcd:fuchsia',
                         'xkcd:gold',
                         'xkcd:lavender',
                         'xkcd:indigo',
                         'xkcd:magenta',
                         '#9acd32',
                         'xkcd:olive',
                         'xkcd:pink',
                         'xkcd:sienna',
                         'xkcd:tan',
                         'xkcd:violet']*2):
    '''
    Convenience plotting function for points and final centres.
    
    ======
    Inputs
    ======
    points : numpy.array of shape (n,p)
        The n points to be clustered, with each point having p dimensions for its coordinates.
    centres : numpy.array of shape (k,p)
        The coordinates of k cluster centres, with each point having p dimensions for its coordinates.
        In other words, centres[i,:] is the cluster centre of cluster i. Here, i is in {0,1,...,k-1}.
    clusters : numpy.array of shape (n,)
        The cluster that each point in points belongs to, with the cluster indices in {0,1,...,k-1}.
        In other words, clusters[i] is the cluster index of the point points[i,:].
    out_fname : str
        The filename to save the figure to (to plt.savefig)
    plot_title : str
        The title to assign the figure.
    scale : float
        The scale of the figure to save/display.
    show_plot : bool
        Whether to call plt.show() at the end of the plot.
        Use False if further changes are desired to the axes after this function is done.
    plot_outliers : bool
        True to plot points[clusters == -1] (i.e. points with cluster label -1 are considered outliers)
    outlier_fmt : str
        The formatting string to plot outlier points.
    plot_centres_first : bool
        If True, will plot centres first (s.t. the points are on top of the centres).
        Otherwise, will plot centres later (s.t. the centres are on top of the points).
    colors : list of str
        The color palette to use for the individual clusters.

    =======
    Outputs
    =======
    None

    ============
    Dependencies
    ============
    matplotlib.pyplot (as plt)
    
    '''
    plt.figure(figsize = (17*scale,10*scale))
    if centres is not None and plot_centres_first:
        plt.plot(centres[:,1],centres[:,0], 'kx', markersize = 10, markeredgewidth = 3)
    for i in range(max(clusters) + 1):
        plt.plot(points[clusters == i,1],points[clusters == i,0], '.',color=colors[i], markersize = 6)
    if centres is not None and not plot_centres_first:
        plt.plot(centres[:,1],centres[:,0], 'kx', markersize = 10, markeredgewidth = 3)
    for i in range(max(clusters) + 1): # Don't let lone points get covered by the crosses
        if np.sum(clusters == i) == 1:
            plt.plot(points[clusters == i,1],points[clusters == i,0], '.',color=colors[i], markersize = 6)
    if plot_outliers:
        plt.plot(points[clusters == -1,1],points[clusters == -1,0],outlier_fmt,fillstyle='none')
    plt.title(plot_title, fontweight = 'bold', fontsize = 16)
    plt.xlim((103.60,104.05)) # Original: plt.xlim((103.60,104.03))
    plt.ylim((1.22,1.48)) # Original: plt.ylim((1.23,1.49))
    plt.xlabel('Longitude (degrees)', fontweight = 'bold', fontsize = 16)
    plt.ylabel('Latitude (degrees)', fontweight = 'bold', fontsize = 16)
    plt.xticks(np.arange(103.60,104.04,0.05),fontsize=16) # Original: plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    if len(out_fname) > 0:
        plt.savefig(out_fname,bbox_inches='tight',transparent=True)
    if show_plot:
        plt.show()

def autolabel(rects, axs, i, height_multiplier = 1.01):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if type(axs) == np.ndarray:
            axs[i].text(rect.get_x() + rect.get_width()/2., height_multiplier*height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=14)
        else:
            axs.text(rect.get_x() + rect.get_width()/2., height_multiplier*height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=14)