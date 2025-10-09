import os
import sys
import subprocess
import traci
import heapq
import xml.etree.ElementTree as ET

class DijkstraForSUMO:
    def __init__(self, net_file):
        self.net_file = net_file
        self.graph = {}
        self.edge_to_junctions = {}
        self.junction_pair_to_edge = {}
        self._build_graph()

    def _build_graph(self):
        # (This function is unchanged)
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        for edge in root.findall('edge'):
            if edge.get('function') != 'internal':
                edge_id = edge.get('id')
                from_node = edge.get('from')
                to_node = edge.get('to')
                lane = edge.find('lane')
                if lane is not None:
                    length = float(lane.get('length'))
                    speed = float(lane.get('speed'))
                    travel_time = length / speed if speed > 0 else float('inf')
                    if from_node not in self.graph:
                        self.graph[from_node] = {}
                    self.graph[from_node][to_node] = travel_time
                    self.edge_to_junctions[edge_id] = (from_node, to_node)
                    self.junction_pair_to_edge[(from_node, to_node)] = edge_id

    def find_shortest_path(self, start_edge, end_edge):
        # --- THIS FUNCTION IS CORRECTED ---
        if start_edge not in self.edge_to_junctions or end_edge not in self.edge_to_junctions:
            return None, float('inf')

        # The path search must start from the junction AT THE END of the current edge.
        start_node = self.edge_to_junctions[start_edge][1]
        end_node = self.edge_to_junctions[end_edge][1]
        
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        previous_nodes = {node: None for node in self.graph}
        pq = [(0, start_node)]
        
        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances[current_node]:
                continue
            if current_node == end_node:
                break
            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items():
                    distance = dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
                        
        path_from_next_junction = []
        current = end_node
        while current is not None:
            path_from_next_junction.insert(0, current)
            current = previous_nodes[current]
            
        if not path_from_next_junction or path_from_next_junction[0] != start_node:
             return None, float('inf')
             
        path_edges = []
        for i in range(len(path_from_next_junction) - 1):
            u, v = path_from_next_junction[i], path_from_next_junction[i+1]
            edge = self.junction_pair_to_edge.get((u, v))
            if edge: path_edges.append(edge)

        # The final, valid route MUST start with the vehicle's current edge.
        final_route = [start_edge] + path_edges
        return final_route, distances[end_node]

def run_simulation():
    # (This function is now correct because the class is fixed)
    router = DijkstraForSUMO('hexagon.net.xml')
    sumo_gui_binary = os.path.join(os.environ.get("SUMO_HOME", "."), "bin", "sumo-gui")
    sumo_cmd = [sumo_gui_binary, "-c", "hexagon.sumocfg", "--tripinfo-output", "tripinfo_after.xml"]
    traci.start(sumo_cmd)
    rerouted_ambulances = set()
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        departed_ids = traci.simulation.getDepartedIDList()
        for veh_id in departed_ids:
            if "amb" in veh_id and veh_id not in rerouted_ambulances:
                original_route = traci.vehicle.getRoute(veh_id)
                if not original_route: continue
                destination_edge = original_route[-1]
                start_edge = traci.vehicle.getRoadID(veh_id)
                print(f"\nAmbulance '{veh_id}' detected on edge '{start_edge}'! Rerouting...")
                final_route, time = router.find_shortest_path(start_edge, destination_edge)
                if final_route:
                    print(f"  Optimal route found ({time:.2f}s). Applying via TraCI.")
                    traci.vehicle.setRoute(veh_id, final_route)
                    rerouted_ambulances.add(veh_id)
                else:
                    print(f"  Could not find a valid route for '{veh_id}'.")
    traci.close()

if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'.")
    run_simulation()