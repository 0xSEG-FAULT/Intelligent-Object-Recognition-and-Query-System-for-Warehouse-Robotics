"""
Object Tracking Module: Tracks objects across multiple frames using Centroid Tracking
Implements: Object ID assignment, tracking, and visualization
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    """
    Tracks objects across frames by matching centroids.
    Uses Euclidean distance to associate detections with existing tracks.
    """
    
    def __init__(self, maxDisappeared=50):
        """
        Initialize the centroid tracker.
        
        Args:
            maxDisappeared: Max frames an object can disappear before being deregistered
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()  # {objectID: (x, y)} - centroid positions
        self.disappeared = OrderedDict()  # {objectID: frames_disappeared_count}
        self.maxDisappeared = maxDisappeared
        self.track_history = OrderedDict()  # {objectID: [(x, y), ...]} - history of positions
        
    def register(self, centroid):
        """
        Register a new object with a new ID.
        
        Args:
            centroid: (x, y) tuple of object center
            
        Returns:
            The newly assigned object ID
        """
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.track_history[self.nextObjectID] = [centroid]
        objectID = self.nextObjectID
        self.nextObjectID += 1
        return objectID
    
    def deregister(self, objectID):
        """
        Deregister an object (remove tracking).
        
        Args:
            objectID: ID of object to remove
        """
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.track_history[objectID]
    
    def update(self, centroids):
        """
        Update tracked objects with new detections.
        Matches new detections to existing tracks based on centroid distance.
        
        Args:
            centroids: List of (x, y) tuples for newly detected objects
            
        Returns:
            Dictionary of {objectID: centroid} for currently tracked objects
        """
        # If no centroids, mark all objects as disappeared
        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        
        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            # Match existing objects to new centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Compute distance between each existing centroid and new centroid
            D = dist.cdist(np.array(objectCentroids), np.array(centroids))
            
            # Find smallest distances (best matches)
            rows = D.min(axis=1).argsort()  # Sort by minimum distance
            cols = D.argmin(axis=1)[rows]   # Get corresponding column indices
            
            # Track which centroids have been used
            usedRows = set()
            usedCols = set()
            
            # Loop through sorted matches
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # Skip if distance is too large (not a good match)
                if D[row, col] > 50:  # threshold: 50 pixels
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = centroids[col]
                self.disappeared[objectID] = 0
                
                # Add to track history
                self.track_history[objectID].append(centroids[col])
                # Keep only last 30 positions for memory efficiency
                if len(self.track_history[objectID]) > 30:
                    self.track_history[objectID].pop(0)
                
                usedRows.add(row)
                usedCols.add(col)
            
            # Register new centroids (not matched to existing objects)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(centroids[col])
            
            # Deregister objects that disappeared
            for row in set(range(0, D.shape[0])).difference(usedRows):
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
        
        return self.objects
    
    def get_track_history(self, objectID):
        """
        Get the movement history of an object.
        
        Args:
            objectID: ID of object
            
        Returns:
            List of (x, y) positions
        """
        return self.track_history.get(objectID, [])
