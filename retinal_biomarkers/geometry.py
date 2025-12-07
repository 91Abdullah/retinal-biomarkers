"""
Geometrical Analysis Module for Retinal Vessels
"""

import numpy as np
from typing import Dict, Tuple, List
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
from collections import deque


class GeometricalAnalysis:
    """
    Custom implementation of geometrical biomarker computation for vessel segmentation.
    Replaces PVBM's GeometricalVBMs class.
    """
    
    def __init__(self):
        # Convolution kernel for neighbor counting
        self.neighbor_kernel = np.array([[1, 1, 1], 
                                          [1, 10, 1], 
                                          [1, 1, 1]])
        
        # Edge detection kernel (Laplacian)
        self.edge_kernel = np.array([[-1, -1, -1], 
                                      [-1, 8, -1], 
                                      [-1, -1, -1]])
    
    def compute_area(self, segmentation: np.ndarray) -> float:
        """
        Compute area as total number of vessel pixels.
        
        Args:
            segmentation: Binary vessel segmentation (H x W)
            
        Returns:
            Area in pixels squared
        """
        return float(np.sum(segmentation))
    
    def compute_particular_points(self, skeleton: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Compute endpoints and intersection points from skeleton.
        
        Endpoints: pixels with exactly 1 neighbor (value 11 after convolution)
        Intersections: pixels with 3+ neighbors (value 13+ after convolution)
        
        Args:
            skeleton: Binary skeleton (H x W)
            
        Returns:
            Tuple of (endpoint_count, intersection_count, endpoint_mask, intersection_mask)
        """
        # Convolve skeleton with neighbor kernel
        neighbors = convolve2d(skeleton, self.neighbor_kernel, mode='same')
        
        # Endpoints have exactly 1 neighbor (10 + 1 = 11)
        endpoints = neighbors == 11
        
        # Intersections have 3+ neighbors (10 + 3 = 13 or more)
        intersections = neighbors >= 13
        
        endpoint_count = int(np.sum(endpoints))
        intersection_count = int(np.sum(intersections))
        
        return endpoint_count, intersection_count, endpoints, intersections
    
    def compute_perimeter(self, segmentation: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute perimeter of vessel segmentation.
        
        Uses edge detection and then counts boundary pixels.
        
        Args:
            segmentation: Binary vessel segmentation
            
        Returns:
            Tuple of (perimeter length, border skeleton)
        """
        # Apply Laplacian edge detection
        derivative = convolve2d(segmentation, self.edge_kernel, mode='same')
        border = derivative > 0
        
        # Skeletonize the border
        border_skeleton = skeletonize(np.ascontiguousarray(border)).astype(np.uint8)
        
        # Count perimeter pixels
        perimeter = float(np.sum(border_skeleton))
        
        return perimeter, border_skeleton
    
    def _find_connected_at_distance(self, skeleton: np.ndarray, 
                                      intersection_points: set,
                                      max_distance: int = 20) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Find points connected to each intersection at exactly max_distance pixels away.
        This matches PVBM's approach of walking 20 pixels from intersections.
        
        Uses recursive traversal matching PVBM's branching2.py behavior exactly.
        PVBM adds neighbors at distance max_distance (when dist == max_size, 
        it adds the neighbor pixels without further recursion).
        
        Args:
            skeleton: Binary skeleton image
            intersection_points: Set of (y, x) coordinates of intersection points
            max_distance: Distance to walk (default 20 like PVBM)
            
        Returns:
            Dictionary mapping intersection points to list of connected points at distance
        """
        connections = {}
        h, w = skeleton.shape
        
        # 8-connectivity directions (same order as PVBM: up, down, left, right, diagonals)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        def recursive_walk(origin, cy, cx, visited, dist):
            """Recursively walk the skeleton exactly like PVBM does."""
            for dy, dx in directions:
                ny, nx = cy + dy, cx + dx
                
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if skeleton[ny, nx] == 1:
                        if dist == max_distance:
                            # At max distance: add this neighbor without recursion
                            if origin not in connections:
                                connections[origin] = []
                            connections[origin].append((ny, nx))
                        else:
                            # Not at max distance: recurse deeper
                            recursive_walk(origin, ny, nx, visited, dist + 1)
        
        for start_y, start_x in intersection_points:
            if skeleton[start_y, start_x] != 1:
                continue
            
            visited = np.zeros_like(skeleton, dtype=bool)
            visited[start_y, start_x] = True
            
            recursive_walk((start_y, start_x), start_y, start_x, visited, 0)
        
        return connections

    def _trace_vessels_bfs(self, skeleton: np.ndarray, 
                           particular_points: set) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
        """
        Trace vessel segments using BFS from particular points.
        
        Args:
            skeleton: Binary skeleton image
            particular_points: Set of (y, x) coordinates of endpoints and intersections
            
        Returns:
            Dictionary mapping start points to list of (end_point, distance) tuples
        """
        connections = {}
        h, w = skeleton.shape
        
        # 8-connectivity directions with distances
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        distances = [1, 1, 1, 1, 
                     np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
        
        for start_y, start_x in particular_points:
            if skeleton[start_y, start_x] != 1:
                continue
                
            visited = np.zeros_like(skeleton, dtype=bool)
            queue = deque([(start_y, start_x, 0.0)])
            visited[start_y, start_x] = True
            
            while queue:
                cy, cx, dist = queue.popleft()
                
                for (dy, dx), step_dist in zip(directions, distances):
                    ny, nx = cy + dy, cx + dx
                    
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and skeleton[ny, nx] == 1:
                        new_dist = dist + step_dist
                        
                        if (ny, nx) in particular_points:
                            # Found another particular point
                            if (start_y, start_x) not in connections:
                                connections[(start_y, start_x)] = []
                            connections[(start_y, start_x)].append(((ny, nx), new_dist))
                            visited[ny, nx] = True
                        else:
                            visited[ny, nx] = True
                            queue.append((ny, nx, new_dist))
        
        return connections
    
    def compute_tortuosity_length(self, skeleton: np.ndarray) -> Tuple[float, float, List[float], List[float], Dict]:
        """
        Compute median tortuosity and total vessel length.
        
        Tortuosity = arc length / chord length for each vessel segment.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            Tuple of (median_tortuosity, total_length, chord_list, arc_list, connections_dict)
        """
        # Find endpoints and intersections
        _, _, endpoints, intersections = self.compute_particular_points(skeleton)
        
        # Get particular point coordinates
        particular = endpoints | intersections
        particular_points = set(zip(*np.where(particular)))
        
        if len(particular_points) < 2:
            return np.nan, float(np.sum(skeleton)), [], [], {}
        
        # Trace vessel connections
        connections = self._trace_vessels_bfs(skeleton, particular_points)
        
        tortuosities = []
        chords = []
        arcs = []
        
        for start_point, end_points in connections.items():
            y1, x1 = start_point
            for end_point, arc_length in end_points:
                y2, x2 = end_point
                
                # Only consider segments longer than 10 pixels
                if arc_length > 10:
                    chord_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if chord_length > 0:
                        tort = arc_length / chord_length
                        tortuosities.append(tort)
                        chords.append(chord_length)
                        arcs.append(arc_length)
        
        median_tort = np.median(tortuosities) if tortuosities else np.nan
        # Divide by 2 because we count each connection from both endpoints
        total_length = sum(chords) / 2 if chords else float(np.sum(skeleton))
        
        return median_tort, total_length, chords, arcs, connections
    
    def compute_branching_angles(self, skeleton: np.ndarray) -> Tuple[float, float, float, Dict, np.ndarray]:
        """
        Compute branching angles at intersection points.
        
        Args:
            skeleton: Binary skeleton image
            
        Returns:
            Tuple of (mean_angle, std_angle, median_angle, angles_dict, centroid_image)
        """
        # Find intersection points only (not endpoints) - matches PVBM
        _, _, endpoints, intersections = self.compute_particular_points(skeleton)
        intersection_points = set(zip(*np.where(intersections)))
        
        angles = {}
        centroid = np.zeros_like(skeleton, dtype=float)
        
        if len(intersection_points) < 1:
            return np.nan, np.nan, np.nan, {}, centroid
        
        # First compute distances from intersection points (for centroid-like weighting)
        # This matches PVBM's compute_distances function
        distance_map = self._compute_distance_from_intersections(skeleton, intersection_points)
        
        # Find connected points at distance 20 from each intersection
        connections = self._find_connected_at_distance(skeleton, intersection_points, max_distance=20)
        
        # First pass: compute all possible angles
        first_pass_angles = {}
        for center_point, neighbor_positions in connections.items():
            if len(neighbor_positions) < 2:
                continue
            
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    p1 = neighbor_positions[i]
                    p2 = neighbor_positions[j]
                    center = np.array(center_point)
                    
                    v1 = np.array(p1) - center
                    v2 = np.array(p2) - center
                    
                    # PVBM uses mean absolute difference > 5
                    mean_dist1 = np.mean(np.abs(v1))
                    mean_dist2 = np.mean(np.abs(v2))
                    
                    if mean_dist1 > 5 and mean_dist2 > 5:
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        
                        if norm1 > 0 and norm2 > 0:
                            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.degrees(np.arccos(cos_angle))
                            
                            key = (center_point, p1, p2)
                            first_pass_angles[key] = angle
        
        # Build connection dictionary from first pass angles
        final_connections = {}
        for key in first_pass_angles.keys():
            center = key[0]
            if center not in final_connections:
                final_connections[center] = []
            for pt in key[1:]:
                if pt not in final_connections[center]:
                    final_connections[center].append(pt)
        
        # Second pass: for intersections with 3+ connections, remove the one 
        # closest to the "centroid" (smallest distance from intersections)
        # This matches PVBM's last_connection_dico logic
        filtered_connections = {}
        for center, neighbors in final_connections.items():
            if len(neighbors) > 2:
                # Find neighbor with minimum distance in distance_map
                min_idx = 0
                min_dist = distance_map.get(neighbors[0], float('inf'))
                for i, pt in enumerate(neighbors):
                    d = distance_map.get(pt, float('inf'))
                    if d < min_dist:
                        min_idx = i
                        min_dist = d
                # Remove the closest point to keep only the two farthest branches
                filtered_neighbors = [n for i, n in enumerate(neighbors) if i != min_idx]
                filtered_connections[center] = filtered_neighbors
            else:
                filtered_connections[center] = neighbors
        
        # Compute final angles from filtered connections
        for center_point, neighbor_positions in filtered_connections.items():
            if len(neighbor_positions) < 2:
                continue
            
            for i in range(len(neighbor_positions)):
                for j in range(i + 1, len(neighbor_positions)):
                    p1 = neighbor_positions[i]
                    p2 = neighbor_positions[j]
                    center = np.array(center_point)
                    
                    v1 = np.array(p1) - center
                    v2 = np.array(p2) - center
                    
                    mean_dist1 = np.mean(np.abs(v1))
                    mean_dist2 = np.mean(np.abs(v2))
                    
                    if mean_dist1 > 5 and mean_dist2 > 5:
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        
                        if norm1 > 0 and norm2 > 0:
                            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.degrees(np.arccos(cos_angle))
                            
                            key = (center_point, p1, p2)
                            # Check for set-equivalent duplicates like PVBM does
                            key_set = frozenset(key)
                            is_duplicate = any(frozenset(k) == key_set for k in angles.keys())
                            if not is_duplicate:
                                angles[key] = angle
                                centroid[center_point[0], center_point[1]] = 1
        
        if not angles:
            return np.nan, np.nan, np.nan, {}, centroid
        
        angle_values = list(angles.values())
        mean_angle = np.mean(angle_values)
        std_angle = np.std(angle_values)
        median_angle = np.median(angle_values)
        
        return mean_angle, std_angle, median_angle, angles, centroid

    def _compute_distance_from_intersections(self, skeleton: np.ndarray, 
                                              intersection_points: set) -> Dict[Tuple[int, int], float]:
        """
        Compute distance from each skeleton pixel to the nearest intersection point.
        This matches PVBM's distance computation for the centroid weighting.
        """
        from collections import deque
        
        h, w = skeleton.shape
        distance_map = {}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for origin in intersection_points:
            visited = np.zeros((h, w), dtype=bool)
            visited[origin] = True
            queue = deque([(origin[0], origin[1], 0)])
            
            while queue:
                cy, cx, dist = queue.popleft()
                current = (cy, cx)
                
                # Update max distance from any intersection
                if current in distance_map:
                    distance_map[current] = max(distance_map[current], dist)
                else:
                    distance_map[current] = dist
                
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and skeleton[ny, nx] == 1:
                        visited[ny, nx] = True
                        queue.append((ny, nx, dist + 1))
        
        return distance_map


# =============================================================================
# Custom Fractal Analysis
# =============================================================================

