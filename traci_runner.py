import os
import sys
import traci
import heapq
import xml.etree.ElementTree as ET
import random

# --- DATA DEFINITIONS (Unchanged) ---
PATIENT_POOL = {
    "P01": {"name": "Ram", "condition": "Cardiac Arrest", "keywords": ["Cardiology"]},
    "P02": {"name": "Sita ", "condition": "Multiple Fractures", "keywords": ["Trauma Care", "Orthopedics"]},
    "P03": {"name": "Arjun ", "condition": "Severe Lacerations", "keywords": ["Emergency Care"]},
    "P04": {"name": "Bheema", "condition": "Stroke", "keywords": ["General Medicine"]},
    "P05": {"name": "Lakshman", "condition": "Major Burn Injury", "keywords": ["Emergency Care", "Trauma Care"]},
}
HOSPITALS = {
    "H-01": {"name": "City_General", "specialties": ["General Medicine", "Emergency Care"], "available_beds": 12, "status": "Accepting Patients"},
    "H-02": {"name": "Green_Heart", "specialties": ["Cardiology"], "available_beds": 5, "status": "Accepting Cardiac Patients"},
    "H-04": {"name": "Tumakuru_Trauma", "specialties": ["Trauma Care", "Orthopedics"], "available_beds": 3, "status": "Accepting Patients"}
}
HOSPITAL_TO_EDGE_MAP = {
    "H-01": "e6", "H-02": "e2", "H-04": "e4"
}
# --- DijkstraForSUMO Class (Unchanged) ---
class DijkstraForSUMO:
    def __init__(self, net_file):
        self.net_file = net_file
        self.graph, self.edge_to_junctions, self.junction_pair_to_edge = {}, {}, {}
        self._build_graph()
    def _build_graph(self):
        tree = ET.parse(self.net_file)
        for edge in tree.getroot().findall('edge'):
            if edge.get('function') != 'internal':
                edge_id, from_node, to_node = edge.get('id'), edge.get('from'), edge.get('to')
                lane = edge.find('lane')
                if lane is not None:
                    travel_time = float(lane.get('length')) / float(lane.get('speed'))
                    if from_node not in self.graph: self.graph[from_node] = {}
                    self.graph[from_node][to_node] = travel_time
                    self.edge_to_junctions[edge_id] = (from_node, to_node)
                    self.junction_pair_to_edge[(from_node, to_node)] = edge_id
    def find_shortest_path(self, start_edge, end_edge):
        if start_edge not in self.edge_to_junctions or end_edge not in self.edge_to_junctions: return None, float('inf')
        start_node, end_node = self.edge_to_junctions[start_edge][1], self.edge_to_junctions[end_edge][1]
        distances = {node: float('inf') for node in self.graph}; distances[start_node] = 0
        previous_nodes = {node: None for node in self.graph}; pq = [(0, start_node)]
        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances[current_node]: continue
            if current_node == end_node: break
            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items():
                    distance = dist + weight
                    if distance < distances[neighbor]:
                        distances[neighbor], previous_nodes[neighbor] = distance, current_node
                        heapq.heappush(pq, (distance, neighbor))
        path_nodes = []; current = end_node
        while current is not None: path_nodes.insert(0, current); current = previous_nodes[current]
        if not path_nodes or path_nodes[0] != start_node: return None, float('inf')
        path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        return [start_edge] + [edge for edge in path_edges if edge], distances[end_node]
# --- find_best_hospital Function (Unchanged) ---
def find_best_hospital(patient_keywords):
    best_hospital, max_score = None, -1
    for h_id, h_data in HOSPITALS.items():
        if h_data["available_beds"] > 0:
            score = 10 if any(k in h_data["specialties"] for k in patient_keywords) else 0
            score += 5 if h_data["status"] != "Busy" else 0
            if score > max_score: max_score, best_hospital = score, h_id
    return best_hospital

def run_simulation():
    router = DijkstraForSUMO('hexagon.net.xml')
    sumo_cmd = [os.path.join(os.environ.get("SUMO_HOME", "."), "bin", "sumo-gui"), "-c", "hexagon.sumocfg", "--tripinfo-output", "tripinfo_after.xml"]
    traci.start(sumo_cmd)
    
    available_patients = list(PATIENT_POOL.keys())
    
    # NEW: A queue to hold offloaded tasks. format: { process_at_time: [tasks] }
    task_queue = {}
    COMMUNICATION_DELAY = 3 # Simulate a 3-second delay for the offloading process

    while traci.simulation.getMinExpectedNumber() > 0:
        current_time = traci.simulation.getTime()
        traci.simulationStep()
        
        # --- 1. VEHICLE: OFFLOAD TASK REQUEST ---
        # A newly departed ambulance sends a request to the server.
        for veh_id in traci.simulation.getDepartedIDList():
            if "amb" in veh_id:
                if not available_patients: continue
                
                patient_id = random.choice(available_patients)
                available_patients.remove(patient_id)
                
                # The task is to find a route for this patient.
                task = {"vehicle_id": veh_id, "patient_id": patient_id}
                
                # Schedule the task to be processed after a delay.
                process_time = current_time + COMMUNICATION_DELAY
                if process_time not in task_queue:
                    task_queue[process_time] = []
                task_queue[process_time].append(task)
                
                print(f"\n[Time {current_time:.0f}s] VEHICLE '{veh_id}': Offloading routing task to server.")

        # --- 2. SERVER: PROCESS OFFLOADED TASK ---
        # The server checks if there are any tasks scheduled for the current time.
        if current_time in task_queue:
            for task in task_queue[current_time]:
                veh_id = task["vehicle_id"]
                patient = PATIENT_POOL[task["patient_id"]]
                
                # Check if vehicle still exists in simulation
                if veh_id not in traci.vehicle.getIDList(): continue
                
                print(f"[Time {current_time:.0f}s] SERVER: Received and processing task for '{veh_id}' (Patient: {patient['name']}).")

                best_hospital_id = find_best_hospital(patient["keywords"])
                if not best_hospital_id:
                    print(f"[Time {current_time:.0f}s] SERVER: Task failed for '{veh_id}'. No suitable hospital.")
                    continue

                hospital = HOSPITALS[best_hospital_id]
                destination_edge = HOSPITAL_TO_EDGE_MAP[best_hospital_id]
                start_edge = traci.vehicle.getRoadID(veh_id)
                
                optimal_route, time = router.find_shortest_path(start_edge, destination_edge)

                # --- 3. VEHICLE: RECEIVE RESPONSE ---
                # The server sends the result back to the vehicle.
                if optimal_route:
                    traci.vehicle.setRoute(veh_id, optimal_route)
                    HOSPITALS[best_hospital_id]["available_beds"] -= 1
                    print(f"[Time {current_time:.0f}s] VEHICLE '{veh_id}': Response received. Applying new route to {hospital['name']}.")
                    print(f"  > Bed count at {hospital['name']} is now: {HOSPITALS[best_hospital_id]['available_beds']}")
                else:
                    print(f"[Time {current_time:.0f}s] VEHICLE '{veh_id}': Response failed. Could not compute route.")
            
            del task_queue[current_time] # Clear processed tasks

    traci.close()

if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'.")
    run_simulation()