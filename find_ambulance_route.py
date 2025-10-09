import xml.etree.ElementTree as ET
import heapq
import sys

class DijkstraForSUMO:
    """
    A class to find the shortest path in a SUMO network using Dijkstra's algorithm.
    The graph is built from the .net.xml file, with junctions as nodes and
    edges weighted by travel time (length / speed).
    """

    def __init__(self, net_file):
        """
        Initializes the router by parsing the network file and building the graph.
        """
        self.net_file = net_file
        self.graph = {}
        self.edge_to_junctions = {}
        self.junction_pair_to_edge = {}
        self._build_graph()

    def _build_graph(self):
        """
        Parses the .net.xml file to build a graph representation.
        Ignores internal edges which are not part of the main road network.
        """
        print(f"Parsing network file: {self.net_file}")
        tree = ET.parse(self.net_file)
        root = tree.getroot()

        for edge in root.findall('edge'):
            # We only consider regular road edges, not internal junction connections
            if edge.get('function') != 'internal':
                edge_id = edge.get('id')
                from_node = edge.get('from')
                to_node = edge.get('to')
                
                # Get the first lane to determine length and speed
                lane = edge.find('lane')
                if lane is not None:
                    length = float(lane.get('length'))
                    speed = float(lane.get('speed'))
                    
                    # Weight is travel time. Avoid division by zero.
                    travel_time = length / speed if speed > 0 else float('inf')

                    # Add edge to the graph
                    if from_node not in self.graph:
                        self.graph[from_node] = {}
                    self.graph[from_node][to_node] = travel_time
                    
                    # Store mappings for later use
                    self.edge_to_junctions[edge_id] = (from_node, to_node)
                    self.junction_pair_to_edge[(from_node, to_node)] = edge_id

        print(f"Graph built successfully with {len(self.graph)} nodes.")

    def find_shortest_path(self, start_edge_id, end_edge_id):
        """
        Calculates the shortest path from a start edge to an end edge.
        """
        if start_edge_id not in self.edge_to_junctions:
            print(f"Error: Start edge '{start_edge_id}' not found in the network.")
            return None, float('inf')
        if end_edge_id not in self.edge_to_junctions:
            print(f"Error: End edge '{end_edge_id}' not found in the network.")
            return None, float('inf')

        # An edge is defined by its start and end junctions
        start_node = self.edge_to_junctions[start_edge_id][0]
        end_node = self.edge_to_junctions[end_edge_id][1]
        
        print(f"\nFinding shortest path from junction '{start_node}' (start of {start_edge_id}) to '{end_node}' (end of {end_edge_id})...")

        # Dijkstra's algorithm initialization
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        previous_nodes = {node: None for node in self.graph}
        
        # Priority queue stores (distance, node)
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # If we've found a shorter path already, skip
            if current_distance > distances[current_node]:
                continue
            
            # If we reached the destination, we can stop early
            if current_node == end_node:
                break

            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items():
                    distance = current_distance + weight
                    
                    # If we found a shorter path to the neighbor
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstruct the path from end to start
        path_junctions = []
        current = end_node
        while current is not None:
            path_junctions.insert(0, current)
            current = previous_nodes[current]
            
        # Check if a path was found
        if path_junctions[0] != start_node:
             return None, float('inf')

        # Convert junction path to edge path
        path_edges = []
        for i in range(len(path_junctions) - 1):
            u, v = path_junctions[i], path_junctions[i+1]
            edge = self.junction_pair_to_edge.get((u, v))
            if edge:
                path_edges.append(edge)

        return path_edges, distances[end_node]


if __name__ == "__main__":
    # The network file to analyze
    net_file = 'hexagon.net.xml'
    
    # --- DEFINE AMBULANCE MISSION HERE ---
    # Example for your ambulance 'amb_1' from hexagon.rou.xml
    start_edge = "e3" 
    # Destination from tripinfo for amb_1 is e11_0, so the edge is e11
    end_edge = "e11"
    
    # Initialize the router and find the path
    router = DijkstraForSUMO(net_file)
    shortest_path, total_time = router.find_shortest_path(start_edge, end_edge)

    # --- PRINT THE RESULTS ---
    print("\n--- Dijkstra's Algorithm Result ---")
    if shortest_path:
        print(f"Optimal route for an ambulance from '{start_edge}' to '{end_edge}':")
        # Print the list of edges as a single space-separated string for easy copying
        print(" ".join(shortest_path))
        print(f"Estimated travel time: {total_time:.2f} seconds")
    else:
        print(f"No path could be found from '{start_edge}' to '{end_edge}'.")