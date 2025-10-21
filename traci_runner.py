import os
import sys
import traci
import heapq
import random
import xml.etree.ElementTree as ET
from collections import Counter
from scipy.optimize import linear_sum_assignment
import numpy as np
import math # Needed for distance calculation

# --- GA CONFIGURATION (Unchanged) ---
GA_CONFIG = {
    "population_size": 50, "generations": 5, "mutation_rate": 0.3,
    "crossover_rate": 0.8, "tournament_size": 3, "elitism_size": 2
}

# --- SIMULATION DATA (Unchanged) ---
EDGE_POOL = [
    "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12",
    "e1_rev", "e2_rev", "e3_rev", "e4_rev", "e5_rev", "e6_rev",
    "e7_rev", "e8_rev", "e9_rev", "e10_rev", "e11_rev", "e12_rev"
]
PATIENT_PROFILES = [
    {"name": "Ravi Kumar", "condition": "Cardiac Arrest", "keywords": ["Cardiology"], "severity": "high"},
    {"name": "Sita Devi", "condition": "Multiple Fractures", "keywords": ["Trauma Care", "Orthopedics"], "severity": "high"},
    {"name": "Amit Patel", "condition": "Head Injury", "keywords": ["Trauma Care"], "severity": "high"},
    {"name": "Vikram Rathod", "condition": "Chest Pain", "keywords": ["Cardiology", "Emergency Care"], "severity": "high"},
    {"name": "Arjun Singh", "condition": "Severe Lacerations", "keywords": ["Emergency Care"], "severity": "medium"},
    {"name": "Priya Sharma", "condition": "Breathing Difficulty", "keywords": ["General Medicine"], "severity": "medium"},
    {"name": "Anjali Gupta", "condition": "Minor Burn", "keywords": ["General Medicine"], "severity": "low"},
    {"name": "Meera Iyer", "condition": "Sprained Ankle", "keywords": ["Orthopedics"], "severity": "low"},
    {"name": "Suresh Verma", "condition": "Allergic Reaction", "keywords": ["Emergency Care"], "severity": "medium"},
    {"name": "Deepa Mehta", "condition": "Abdominal Pain", "keywords": ["General Medicine"], "severity": "medium"},
]
def generate_random_patients(num_patients):
    patients = {}
    if num_patients > len(PATIENT_PROFILES) or num_patients > len(EDGE_POOL):
        raise ValueError("Cannot generate more unique patients than available profiles/edges.")
    chosen_profiles = random.sample(PATIENT_PROFILES, num_patients)
    chosen_edges = random.sample(EDGE_POOL, num_patients) if num_patients <= len(EDGE_POOL) else random.choices(EDGE_POOL, k=num_patients)
    for i in range(num_patients):
        patient_id = f"P{i+1:02}"; profile = chosen_profiles[i]
        patients[patient_id] = {
            "name": profile["name"], "condition": profile["condition"], "keywords": profile["keywords"],
            "start_edge": chosen_edges[i], "severity": profile["severity"], "status": "pending"
        }
    return patients
ALL_PATIENTS = generate_random_patients(8)
HOSPITALS_TEMPLATE = {
    "H-01": {"name": "City_General", "specialties": ["General Medicine", "Emergency Care"], "initial_beds": 12, "dest_edge": "e6"},
    "H-02": {"name": "Green_Heart", "specialties": ["Cardiology"], "initial_beds": 5, "dest_edge": "e2"},
    "H-04": {"name": "Tumakuru_Trauma", "specialties": ["Trauma Care", "Orthopedics"], "initial_beds": 3, "dest_edge": "e4"}
}
HOSPITALS = {h_id: data.copy() for h_id, data in HOSPITALS_TEMPLATE.items()}
for h_id in HOSPITALS: HOSPITALS[h_id]["available_beds"] = HOSPITALS[h_id]["initial_beds"]
SEVERITY_WEIGHTS = { "high": 1.5, "medium": 1.0, "low": 0.5 }
AMBULANCES = ["amb_1", "amb_2", "amb_3", "amb_4", "amb_5", "amb_6", "amb_7", "amb_8"]


# --- Dijkstra Pathfinding Class (Unchanged) ---
class DijkstraForSUMO:
    def __init__(self, net_file):
        self.net_file = net_file; self.graph, self.edge_to_junctions, self.junction_pair_to_edge = {}, {}, {}
        self._build_graph()
    def _build_graph(self):
        try:
            tree = ET.parse(self.net_file); root = tree.getroot(); junctions_with_edges = set()
            for edge in root.findall('edge'):
                if edge.get('function') != 'internal':
                    edge_id = edge.get('id'); from_node, to_node = edge.get('from'), edge.get('to')
                    if from_node and to_node:
                        junctions_with_edges.add(from_node); junctions_with_edges.add(to_node)
                        if from_node not in self.graph: self.graph[from_node] = {}
                        self.graph[from_node][to_node] = 1
                        self.edge_to_junctions[edge_id] = (from_node, to_node); self.junction_pair_to_edge[(from_node, to_node)] = edge_id
            for node in junctions_with_edges:
                 if node not in self.graph: self.graph[node] = {}
        except FileNotFoundError: raise ValueError(f"Net file '{self.net_file}' not found.")
        except ET.ParseError as e: raise ValueError(f"Failed to parse net file '{self.net_file}': {e}")
    def find_shortest_path_time(self, start_edge, end_edge):
        if not start_edge or not end_edge: return float('inf')
        if start_edge not in self.edge_to_junctions or end_edge not in self.edge_to_junctions: return float('inf')
        _, time = self.find_dynamic_shortest_path(start_edge, end_edge); return time
    def find_dynamic_shortest_path(self, start_edge, end_edge):
        if start_edge == end_edge:
            try: return [start_edge], max(1.0, traci.edge.getTraveltime(start_edge))
            except traci.TraCIException: return [start_edge], 1.0
        start_node_tuple = self.edge_to_junctions.get(start_edge); end_node_tuple = self.edge_to_junctions.get(end_edge)
        if not start_node_tuple or not end_node_tuple: return None, float('inf')
        start_junction, end_junction = start_node_tuple[1], end_node_tuple[0]
        if start_junction not in self.graph: return None, float('inf')
        distances = {node: float('inf') for node in self.graph};
        if start_junction in distances: distances[start_junction] = 0
        else: return None, float('inf')
        previous_nodes = {node: None for node in self.graph}; pq = [(0, start_junction)]
        path_found = False
        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances.get(current_node, float('inf')): continue
            if current_node == end_junction: path_found = True; break
            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    edge_id = self.junction_pair_to_edge.get((current_node, neighbor))
                    if edge_id and neighbor in distances:
                        try: weight = max(1.0, traci.edge.getTraveltime(edge_id))
                        except traci.TraCIException: weight = 1000.0
                        distance = dist + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance; previous_nodes[neighbor] = current_node; heapq.heappush(pq, (distance, neighbor))
        if not path_found or distances.get(end_junction, float('inf')) == float('inf'): return None, float('inf')
        path_nodes = []; current = end_junction
        while current is not None: path_nodes.insert(0, current); current = previous_nodes.get(current)
        if not path_nodes or path_nodes[0] != start_junction: return None, float('inf')
        path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        valid_junction_edges = [edge for edge in path_edges if edge]
        full_edge_path = [start_edge] + valid_junction_edges
        # Correctly handle case where destination is the first junction
        if not valid_junction_edges and end_edge != start_edge:
             full_edge_path.append(end_edge)
        elif valid_junction_edges and full_edge_path[-1] != end_edge:
            full_edge_path.append(end_edge)

        try: first_edge_time = max(1.0, traci.edge.getTraveltime(start_edge))
        except traci.TraCIException: first_edge_time = 1.0

        calculated_path_time = distances.get(end_junction, float('inf'))

        # Refined total time calculation
        total_time = 0
        current_full_path = [start_edge] # Start building path for time calc
        path_valid = True
        for i in range(len(path_nodes) - 1):
             edge = self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1]))
             if edge:
                  current_full_path.append(edge)
                  try: total_time += max(1.0, traci.edge.getTraveltime(edge))
                  except traci.TraCIException: path_valid = False; break
             else: path_valid = False; break # Invalid path segment
        if end_edge not in current_full_path: current_full_path.append(end_edge) # Ensure end edge is considered

        if not path_valid: return None, float('inf')

        # Recalculate total time by summing actual travel times from TraCI for the found path
        total_dynamic_time = 0
        try:
             for edge in current_full_path:
                  total_dynamic_time += max(1.0, traci.edge.getTraveltime(edge))
        except traci.TraCIException:
             return current_full_path, float('inf') # Return path but indicate time error

        return current_full_path, total_dynamic_time


# --- Event-Driven GA Functions (Unchanged) ---
def create_chromosome_event(current_patients):
    hospital_ids = list(HOSPITALS.keys())
    return [random.choice(hospital_ids) for _ in current_patients]
def calculate_fitness_event(chromosome, router, current_patient_ids, current_ambulance_ids):
    specialty_penalty, bed_penalty = 0, 0; num_current_patients = len(current_patient_ids)
    hospital_assignments = Counter(chromosome)
    for i, patient_id in enumerate(current_patient_ids):
        hospital_id = chromosome[i]
        if patient_id not in ALL_PATIENTS or hospital_id not in HOSPITALS: continue
        if not any(spec in HOSPITALS[hospital_id]["specialties"] for spec in ALL_PATIENTS[patient_id]["keywords"]): specialty_penalty += 5000
    for h_id, count in hospital_assignments.items():
        if h_id in HOSPITALS and count > HOSPITALS[h_id]["available_beds"]: bed_penalty += 2000 * (count - HOSPITALS[h_id]["available_beds"])
    current_vehicle_ids = traci.vehicle.getIDList(); valid_ambulance_ids = [amb_id for amb_id in current_ambulance_ids if amb_id in current_vehicle_ids]
    num_ambulances = len(valid_ambulance_ids); amb_edges = []; final_valid_amb_ids = []
    if num_ambulances == 0: return float('inf')
    for amb_id in valid_ambulance_ids:
        try: edge = traci.vehicle.getRoadID(amb_id); amb_edges.append(edge); final_valid_amb_ids.append(amb_id)
        except traci.TraCIException: pass
    valid_ambulance_ids = final_valid_amb_ids; num_ambulances = len(valid_ambulance_ids)
    if num_ambulances == 0: return float('inf')
    num_assignments = min(num_current_patients, num_ambulances); assignable_patient_ids = current_patient_ids[:num_assignments]
    cost_matrix = np.full((num_ambulances, num_assignments), float('inf'))
    for i in range(num_ambulances):
        for j, patient_id in enumerate(assignable_patient_ids):
            hospital_id = chromosome[j]; patient_info = ALL_PATIENTS.get(patient_id); hospital_info = HOSPITALS.get(hospital_id)
            if not patient_info or not hospital_info: continue
            time_to_patient = router.find_shortest_path_time(amb_edges[i], patient_info["start_edge"])
            time_to_hospital = router.find_shortest_path_time(patient_info["start_edge"], hospital_info["dest_edge"])
            if time_to_patient != float('inf') and time_to_hospital != float('inf'):
                mission_time = time_to_patient + time_to_hospital
                weighted_time = mission_time * SEVERITY_WEIGHTS.get(patient_info.get("severity", "medium"), 1.0); cost_matrix[i, j] = weighted_time
    try:

        # Check if matrix is all infinities before calling assignment
        if np.all(cost_matrix == float('inf')):
             total_weighted_time = float('inf')
        else:
             amb_indices, assigned_pat_indices_in_matrix = linear_sum_assignment(cost_matrix)
             # Sum the costs of the assigned pairs, ignore infinite costs if assignment failed for some
             valid_costs = cost_matrix[amb_indices, assigned_pat_indices_in_matrix]
             total_weighted_time = valid_costs[valid_costs != float('inf')].sum()
             if np.all(valid_costs == float('inf')):
                 total_weighted_time = float('inf')
    except ValueError as e:
         return float('inf')
    return total_weighted_time + specialty_penalty + bed_penalty
def selection(population, fitnesses):
    pop_fit_pairs = list(zip(population, fitnesses)); k = min(GA_CONFIG["tournament_size"], len(pop_fit_pairs))
    if not pop_fit_pairs or k == 0: return population[0] if population else []
    tournament = random.sample(pop_fit_pairs, k); return min(tournament, key=lambda x: x[1])[0]
def crossover(parent1, parent2):
    if random.random() > GA_CONFIG["crossover_rate"] or len(parent1) <= 1: return list(parent1), list(parent2)
    point = random.randint(1, len(parent1) - 1); p1 = list(parent1); p2 = list(parent2); return p1[:point] + p2[point:], p2[:point] + p1[point:]
def mutate(chromosome):
    mutated_chromosome = list(chromosome)
    if random.random() < GA_CONFIG["mutation_rate"] and mutated_chromosome:
        idx = random.randrange(len(mutated_chromosome)); mutated_chromosome[idx] = random.choice(list(HOSPITALS.keys()))
    return mutated_chromosome

# --- Mission Management & Dispatch Logic ---
mission_log_counter = 1
def generate_dispatch_report_event(dispatch_details):
    global mission_log_counter
    try:
        with open("dispatch_summary.txt", "a") as f:
            if mission_log_counter == 1:
                f.write("========================================\n")
                f.write("   Event-Driven Dispatch Summary\n")
                f.write("========================================\n\n")
            mission = dispatch_details; patient = ALL_PATIENTS.get(mission['patient_id']); hospital = HOSPITALS.get(mission['hospital_id'])
            if not patient or not hospital: return
            f.write(f"--- Mission {mission_log_counter} (Dispatched at T={mission['dispatch_time']:.1f}s) ---\n")
            f.write(f"  Ambulance: {mission['ambulance_id']}\n\n")
            f.write(f"  Patient Details:\n    - ID:         {mission['patient_id']}\n    - Name:       {patient['name']}\n    - Condition:  {patient['condition']} (Severity: {patient['severity']})\n    - Location:   Edge '{patient['start_edge']}'\n\n") # Log the actual edge
            f.write(f"  Assigned Hospital:\n    - ID:         {mission['hospital_id']}\n    - Name:       {hospital['name']}\n    - Specialties:{', '.join(hospital['specialties'])}\n    - Beds (Current Avail. Before): {mission['beds_at_dispatch']} -> {mission['beds_at_dispatch'] - 1}\n")
            f.write(f"    - Destination:Edge '{hospital['dest_edge']}'\n\n----------------------------------------\n\n")
            mission_log_counter += 1
    except Exception as e: print(f"Error writing report: {e}")

# --- Buffer Zone / Priority Constants and State ---
PRIORITY_DISTANCE_THRESHOLD = 150.0 # *** UPDATED CONSTANT *** Meters before junction to activate priority
PRIORITY_VISUAL_COLOR = (255, 165, 0, 150) # Orange, semi-transparent
prioritized_vehicles = {} # {amb_id: {'tls_id': tls_id, 'visual_id': poly_id}}

# --- Traffic Light Control Logic ---
TLS_IDS = ["n_center", "n1", "n2", "n3", "n4", "n5", "n6"]
MIN_GREEN_TIME = 5; MAX_GREEN_TIME = 40
YELLOW_TIME = 3; ALL_RED_TIME = 2

# !! CRITICAL: VERIFY THESE MAPPINGS AGAINST YOUR hexagon.net.xml !!
# (Mappings assumed unchanged and correct based on previous context)
PHASE_MAP = {
    "n_center": { 0: ["e7_rev_0", "e10_rev_0"], 3: ["e8_rev_0", "e11_rev_0"], 6: ["e9_rev_0", "e12_rev_0"] },
    "n1": { 0: ["e6_0", "e1_rev_0"], 2: ["e7_0"] }, # Green phases 0 and 2
    "n2": { 0: ["e1_0", "e2_rev_0"], 2: ["e8_0"] }, # Green phases 0 and 2
    "n3": { 0: ["e2_0", "e3_rev_0"], 2: ["e9_0"] }, # Green phases 0 and 2
    "n4": { 0: ["e3_0", "e4_rev_0"], 2: ["e10_0"] },# Green phases 0 and 2
    "n5": { 0: ["e4_0", "e5_rev_0"], 2: ["e11_0"] },# Green phases 0 and 2
    "n6": { 0: ["e5_0", "e6_rev_0"], 2: ["e12_0"] } # Green phases 0 and 2
}
YELLOW_PHASE_INDICES = {
    "n_center": { 0: 1, 3: 4, 6: 7 },
    "n1": { 0: 1, 2: 3 }, # Green 0 -> Yellow 1, Green 2 -> Yellow 3
    "n2": { 0: 1, 2: 3 },
    "n3": { 0: 1, 2: 3 },
    "n4": { 0: 1, 2: 3 },
    "n5": { 0: 1, 2: 3 },
    "n6": { 0: 1, 2: 3 }
}
ALL_RED_PHASE_INDICES = { # Index of an all-red phase (or None if missing)
    "n_center": 2,
    "n1": None, # Assuming no dedicated all-red phase from XML
    "n2": None,
    "n3": None,
    "n4": None,
    "n5": None,
    "n6": None
}
DETECTOR_MAP = {
    "e7_rev_0": "det_e7_rev_0", "e8_rev_0": "det_e8_rev_0", "e9_rev_0": "det_e9_rev_0",
    "e10_rev_0": "det_e10_rev_0", "e11_rev_0": "det_e11_rev_0", "e12_rev_0": "det_e12_rev_0",
    "e1_rev_0": "det_e1_rev_0_n1", "e7_0": "det_e7_0_n1", "e6_0": "det_e6_0_n1",
    "e2_rev_0": "det_e2_rev_0_n2", "e8_0": "det_e8_0_n2", "e1_0": "det_e1_0_n2",
    "e2_0": "det_e2_0_n3", "e3_rev_0": "det_e3_rev_0_n3", "e9_0": "det_e9_0_n3",
    "e10_0": "det_e10_0_n4", "e3_0": "det_e3_0_n4", "e4_rev_0": "det_e4_rev_0_n4",
    "e4_0": "det_e4_0_n5", "e5_rev_0": "det_e5_rev_0_n5", "e11_0": "det_e11_0_n5",
    "e5_0": "det_e5_0_n6", "e6_rev_0": "det_e6_rev_0_n6", "e12_0": "det_e12_0_n6",
}

tls_current_phase_indices = {}; tls_phase_start_times = {}; tls_states = {}

def get_phase_demand(tls_id, phase_index):
    demand = 0; lanes = PHASE_MAP.get(tls_id, {}).get(phase_index, [])
    for lane_id in lanes:
        det_id = DETECTOR_MAP.get(lane_id)
        if det_id:
            try: demand += traci.lanearea.getLastStepHaltingNumber(det_id)
            except traci.TraCIException: pass
    return demand

def switch_to_phase(tls_id, new_phase_index, current_time):
    global tls_current_phase_indices, tls_phase_start_times, tls_states
    try:
        # Check if new_phase_index is valid
        program_length = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases)
        if new_phase_index >= program_length:
             print(f"Warn: Invalid phase {new_phase_index} for {tls_id}. Max={program_length-1}. Ignoring."); return

        traci.trafficlight.setPhase(tls_id, new_phase_index)
        tls_current_phase_indices[tls_id] = new_phase_index
        tls_phase_start_times[tls_id] = current_time
        # Determine state
        if new_phase_index in YELLOW_PHASE_INDICES.get(tls_id, {}).values(): tls_states[tls_id] = "YELLOW"
        elif new_phase_index == ALL_RED_PHASE_INDICES.get(tls_id): tls_states[tls_id] = "RED_CLEAR"
        else: tls_states[tls_id] = "GREEN"
    except (traci.TraCIException, KeyError, IndexError) as e: print(f"Error switching TLS {tls_id}: {e}")

# --- Function to activate visual + priority ---
def activate_priority(amb_id, tls_id, current_time):
    global prioritized_vehicles
    if amb_id in prioritized_vehicles: return # Already active

    print(f"PRIORITY: Activating for {amb_id} at {tls_id} (T={current_time:.1f}s). Using SpeedMode 31.")
    poly_id = None

    # Set Speed Mode 31
    try:
        traci.vehicle.setSpeedMode(amb_id, 31)
    except traci.TraCIException as e:
        print(f"Warn: Failed to set speed mode for {amb_id}: {e}")
        return # Cannot activate priority if speed mode fails

    # Add visual cue
    try:
        shape = traci.junction.getShape(tls_id)
        if shape:
            center_x = sum(p[0] for p in shape) / len(shape); center_y = sum(p[1] for p in shape) / len(shape)
            expanded_shape = [(p[0] + (p[0]-center_x)*0.15, p[1] + (p[1]-center_y)*0.15) for p in shape]
            poly_id = f"priority_visual_{tls_id}_{amb_id}"
            traci.polygon.add(poly_id, expanded_shape, PRIORITY_VISUAL_COLOR, layer=15, fill=True)
    except traci.TraCIException as e: print(f"Warn: Viz add failed for {tls_id}: {e}")

    prioritized_vehicles[amb_id] = {'tls_id': tls_id, 'visual_id': poly_id}

# --- Function to deactivate visual + priority ---
def deactivate_priority(amb_id):
    global prioritized_vehicles
    if amb_id not in prioritized_vehicles: return

    state_info = prioritized_vehicles[amb_id]
    print(f"PRIORITY: Deactivating for {amb_id} at {state_info['tls_id']}.")

    # Reset Speed Mode
    try: traci.vehicle.setSpeedMode(amb_id, 0) # Reset to default (obeys lights)
    except traci.TraCIException: pass # Ignore if vehicle gone

    # Remove visual cue
    if state_info.get('visual_id'):
        try: traci.polygon.remove(state_info['visual_id'])
        except traci.TraCIException: pass

    del prioritized_vehicles[amb_id]

# --- REVISED: Check proximity and manage priority ---
def check_ambulance_proximity_and_manage_priority(active_missions, current_time):
    global prioritized_vehicles
    active_amb_ids = list(active_missions.keys())

    # Check currently prioritized vehicles to see if they've cleared
    for amb_id in list(prioritized_vehicles.keys()):
        if amb_id not in active_amb_ids: # Ambulance finished mission entirely
             deactivate_priority(amb_id); continue
        try:
             tls_id = prioritized_vehicles[amb_id]['tls_id']
             amb_edge = traci.vehicle.getRoadID(amb_id)
             # Check if ambulance is past the intersection
             incoming_edges = traci.junction.getIncomingEdges(tls_id)
             if amb_edge not in incoming_edges:
                  # Robust check: Route index past junction edges
                  amb_route = traci.vehicle.getRoute(amb_id); current_route_index = traci.vehicle.getRouteIndex(amb_id)
                  # Internal edges start with ':'
                  junction_edges = incoming_edges
                  junction_edge_indices = [i for i, edge in enumerate(amb_route) if edge in junction_edges]
                  # If no junction edges found in route OR index is past the last one
                  if not junction_edge_indices or current_route_index > max(junction_edge_indices):
                       deactivate_priority(amb_id)
        except traci.TraCIException: deactivate_priority(amb_id); continue # Error, deactivate

    # Check active ambulances for potential new priority requests
    for amb_id, mission in active_missions.items():
        if mission["stage"] == "done" or amb_id in prioritized_vehicles: continue # Skip done/already prioritized

        try:
            # FIX: Get the *current* lane ID, not the first edge in the route
            current_lane = traci.vehicle.getLaneID(amb_id)
            if not current_lane: continue # Skip if no lane (e.g., mid-junction)

            current_edge = traci.vehicle.getRoadID(amb_id)
            junction_id = traci.edge.getToJunction(current_edge) # Junction at end of current edge

            if junction_id in TLS_IDS: # Check if it's a controlled light
                tls_id = junction_id

                # --- THIS IS THE CRITICAL FIX ---
                # Calculate distance to junction
                dist_to_junction = traci.lane.getLength(current_lane) - traci.vehicle.getLanePosition(amb_id)

                # Check if inside the 150m buffer zone
                if dist_to_junction < PRIORITY_DISTANCE_THRESHOLD:
                    # Ambulance is inside the buffer zone, activate priority
                    activate_priority(amb_id, tls_id, current_time)
                # --- END OF FIX ---

        except traci.TraCIException: continue # Ignore errors (e.g., vehicle left sim)


# --- Adaptive Traffic Light Logic (Checks for priority activation) ---
def run_adaptive_tls_logic(tls_id, current_time, prioritized_vehicles_state):
    global tls_current_phase_indices, tls_phase_start_times, tls_states

    # --- Skip if an ambulance has priority at this light ---
    if tls_id in [state['tls_id'] for state in prioritized_vehicles_state.values()]:
        # print(f"TLS {tls_id}: Skipping adaptive logic due to active priority.") # Optional debug
        return

    # Initialize state if not present (Unchanged)
    if tls_id not in tls_states:
        try:
             tls_current_phase_indices[tls_id] = traci.trafficlight.getPhase(tls_id); tls_phase_start_times[tls_id] = current_time
             if tls_current_phase_indices[tls_id] in YELLOW_PHASE_INDICES.get(tls_id, {}).values(): tls_states[tls_id] = "YELLOW"
             elif tls_current_phase_indices[tls_id] == ALL_RED_PHASE_INDICES.get(tls_id): tls_states[tls_id] = "RED_CLEAR"
             else: tls_states[tls_id] = "GREEN"
        except traci.TraCIException: print(f"Warn: Cannot init TLS {tls_id}"); return

    # Get current state (Unchanged)
    current_phase_index = tls_current_phase_indices.get(tls_id, 0); phase_start_time = tls_phase_start_times.get(tls_id, current_time)
    current_state = tls_states.get(tls_id, "GREEN"); phase_elapsed_time = current_time - phase_start_time

    # --- Actuated Logic (Unchanged, handles GREEN, YELLOW, RED_CLEAR) ---
    if current_state == "YELLOW":
        if phase_elapsed_time >= YELLOW_TIME:
            all_red_phase = ALL_RED_PHASE_INDICES.get(tls_id)
            # --- CORRECTED INDENTATION ---
            if all_red_phase is not None:
                switch_to_phase(tls_id, all_red_phase, current_time)
            else: # No all-red? Decide next green immediately
                max_demand = -1; best_next_phase = -1
                for green_phase_idx in PHASE_MAP.get(tls_id, {}).keys():
                    demand = get_phase_demand(tls_id, green_phase_idx)
                    if demand > max_demand: max_demand = demand; best_next_phase = green_phase_idx
                if best_next_phase != -1:
                    switch_to_phase(tls_id, best_next_phase, current_time)
                elif PHASE_MAP.get(tls_id): # Default to first green phase if no demand
                    switch_to_phase(tls_id, list(PHASE_MAP[tls_id].keys())[0], current_time)
        return # Explicit return after handling YELLOW logic
    elif current_state == "RED_CLEAR":
        if phase_elapsed_time >= ALL_RED_TIME:
            max_demand = -1; best_next_phase = -1
            for green_phase_idx in PHASE_MAP.get(tls_id, {}).keys():
                demand = get_phase_demand(tls_id, green_phase_idx)
                if demand > max_demand: max_demand = demand; best_next_phase = green_phase_idx
            if best_next_phase != -1:
                switch_to_phase(tls_id, best_next_phase, current_time)
            # --- CORRECTED INDENTATION ---
            elif PHASE_MAP.get(tls_id): # Default to first green phase
                switch_to_phase(tls_id, list(PHASE_MAP[tls_id].keys())[0], current_time)
        return # Explicit return after handling RED_CLEAR logic
    elif current_state == "GREEN":
        if phase_elapsed_time >= MIN_GREEN_TIME:
            current_phase_demand = get_phase_demand(tls_id, current_phase_index); conflicting_demand = 0
            for other_phase_idx in PHASE_MAP.get(tls_id, {}).keys():
                 if other_phase_idx != current_phase_index: conflicting_demand = max(conflicting_demand, get_phase_demand(tls_id, other_phase_idx))
            switch_decision = False
            if phase_elapsed_time >= MAX_GREEN_TIME: switch_decision = True
            elif current_phase_demand == 0 and conflicting_demand > 0: switch_decision = True
            elif conflicting_demand > current_phase_demand: switch_decision = True
            if switch_decision:
                yellow_phase = YELLOW_PHASE_INDICES.get(tls_id, {}).get(current_phase_index)
                if yellow_phase is not None: switch_to_phase(tls_id, yellow_phase, current_time)
                else:
                     all_red_phase = ALL_RED_PHASE_INDICES.get(tls_id);
                     if all_red_phase is not None: switch_to_phase(tls_id, all_red_phase, current_time)
                     else: # No yellow, no all_red -> decide next green directly
                          max_demand = -1; best_next_phase = -1
                          for green_phase_idx in PHASE_MAP.get(tls_id, {}).keys():
                               if green_phase_idx != current_phase_index:
                                    demand = get_phase_demand(tls_id, green_phase_idx)
                                    if demand > max_demand: max_demand = demand; best_next_phase = green_phase_idx
                          if best_next_phase != -1: switch_to_phase(tls_id, best_next_phase, current_time)
                          elif PHASE_MAP.get(tls_id): switch_to_phase(tls_id, list(PHASE_MAP[tls_id].keys())[0], current_time)

# --- update_active_missions (Manual Stop Version - UPDATED PICKUP TIME) ---
def update_active_missions(active_missions, available_ambulances, router):
    finished_mission_ambulance_ids = []; current_time = traci.simulation.getTime(); ambulance_became_free = False
    for amb_id, mission in list(active_missions.items()):
        current_edge = None
        try:
            if amb_id not in traci.vehicle.getIDList():
                if mission["stage"] != "done":
                    print(f"MISSION COMPLETE: {amb_id} finished (veh removed) for {mission['patient_id']} at T={current_time:.1f}s.")
                    mission["stage"] = "done"
                    # Return bed ONLY if mission involved transport/pickup
                    if mission.get("stage") in ["to_hospital", "pickup"] and mission.get("hospital_id") in HOSPITALS:
                         HOSPITALS[mission["hospital_id"]]["available_beds"] = min(HOSPITALS[mission["hospital_id"]]["available_beds"] + 1, HOSPITALS[mission["hospital_id"]]["initial_beds"])
                    finished_mission_ambulance_ids.append(amb_id)
                continue
            current_edge = traci.vehicle.getRoadID(amb_id)
            if mission["stage"] == "to_patient":
                if current_edge == mission["patient_edge"]:
                    print(f"INFO: {amb_id} reached patient {mission['patient_id']} at T={current_time:.1f}s. Stopping for pickup.")
                    traci.vehicle.setSpeed(amb_id, 0); mission["stage"] = "pickup"; mission["pickup_start_time"] = current_time
                elif current_time % 5 == 0:
                    try: traci.vehicle.changeTarget(amb_id, mission["patient_edge"])
                    except traci.TraCIException: pass # Ignore if route becomes invalid momentarily
            elif mission["stage"] == "pickup":
                pickup_duration = 2.0 # *** UPDATED PICKUP DURATION ***
                if current_time >= mission["pickup_start_time"] + pickup_duration:
                    print(f"INFO: {amb_id} finished pickup for {mission['patient_id']}. Proceeding.")
                    mission["stage"] = "to_hospital"; traci.vehicle.setSpeed(amb_id, -1) # Resume normal speed control
                    # Ensure still ignoring lights if needed (speedMode 31 handles this)
                    if amb_id in prioritized_vehicles: traci.vehicle.setSpeedMode(amb_id, 31)
                    try: traci.vehicle.changeTarget(amb_id, mission["hospital_edge"])
                    except traci.TraCIException: print(f"Warn: Could not set target to {mission['hospital_edge']} for {amb_id}"); pass
                else: traci.vehicle.setSpeed(amb_id, 0) # Keep stopped during pickup
            elif mission["stage"] == "to_hospital":
                if current_edge == mission["hospital_edge"]:
                    print(f"MISSION COMPLETE: {amb_id} delivered patient {mission['patient_id']} at T={current_time:.1f}s.")
                    mission["stage"] = "done"
                    if mission.get("hospital_id") in HOSPITALS:
                         HOSPITALS[mission["hospital_id"]]["available_beds"] = min(HOSPITALS[mission["hospital_id"]]["available_beds"] + 1, HOSPITALS[mission["hospital_id"]]["initial_beds"])
                    finished_mission_ambulance_ids.append(amb_id)
                elif current_time % 5 == 0:
                     try: traci.vehicle.changeTarget(amb_id, mission["hospital_edge"])
                     except traci.TraCIException: pass # Ignore if route becomes invalid momentarily
        except traci.TraCIException as e:
            print(f"TraCI Error updating {amb_id}: {e}. Marking done.")
            if mission["stage"] != "done":
                mission["stage"] = "done"
                # Return bed ONLY if mission involved transport/pickup before error
                if mission.get("stage") in ["to_hospital", "pickup"] and mission.get("hospital_id") in HOSPITALS:
                     HOSPITALS[mission["hospital_id"]]["available_beds"] = min(HOSPITALS[mission["hospital_id"]]["available_beds"] + 1, HOSPITALS[mission["hospital_id"]]["initial_beds"])
                finished_mission_ambulance_ids.append(amb_id)
    for amb_id in finished_mission_ambulance_ids:
        if amb_id in active_missions: del active_missions[amb_id]
        if amb_id not in available_ambulances: available_ambulances.append(amb_id); ambulance_became_free = True
        deactivate_priority(amb_id) # Deactivate priority when mission finishes
    return ambulance_became_free

# --- run_dispatch_cycle (Unchanged) ---
def run_dispatch_cycle(pending_patient_ids, available_ambulance_ids, active_missions, router):
    if not pending_patient_ids or not available_ambulance_ids: return
    current_time = traci.simulation.getTime()
    print(f"\n--- Running Dispatch Cycle at T={current_time:.1f}s ---"); print(f"Pending: {len(pending_patient_ids)}, Available Amb: {len(available_ambulance_ids)}")
    current_vehicle_ids = traci.vehicle.getIDList(); valid_available_amb_ids = [a for a in available_ambulance_ids if a in current_vehicle_ids]
    if not valid_available_amb_ids: print("No valid ambulances in sim."); return
    num_current_patients = len(pending_patient_ids); num_available_ambulances = len(valid_available_amb_ids)
    population = [create_chromosome_event(pending_patient_ids) for _ in range(GA_CONFIG["population_size"])]
    for gen in range(GA_CONFIG["generations"]):
        fitnesses = [calculate_fitness_event(chrom, router, pending_patient_ids, valid_available_amb_ids) for chrom in population]
        valid_pop_fitness = [(p, f) for p, f in zip(population, fitnesses) if f != float('inf')]
        if not valid_pop_fitness: continue
        pop_with_fitness = sorted(valid_pop_fitness, key=lambda x: x[1])
        # print(f"  Dispatch Gen {gen+1}, Best Fitness: {pop_with_fitness[0][1]:.2f}")
        new_population = [chrom for chrom, fit in pop_with_fitness[:GA_CONFIG["elitism_size"]]]
        valid_population = [p for p, f in pop_with_fitness]; valid_fitnesses = [f for p, f in pop_with_fitness]
        num_to_generate = GA_CONFIG["population_size"] - GA_CONFIG["elitism_size"];
        if num_to_generate % 2 != 0: num_to_generate -= 1
        for _ in range(num_to_generate // 2):
            k = min(GA_CONFIG["tournament_size"], len(valid_population))
            parent1 = selection(valid_population, valid_fitnesses) if k > 0 else random.choice(valid_population) if valid_population else []
            parent2 = selection(valid_population, valid_fitnesses) if k > 0 else random.choice(valid_population) if valid_population else []
            if not parent1 or not parent2: continue
            child1, child2 = crossover(parent1, parent2); new_population.extend([mutate(child1), mutate(child2)])
        if (GA_CONFIG["population_size"] - GA_CONFIG["elitism_size"]) % 2 != 0 and new_population: new_population.append(mutate(new_population[0]))
        population = new_population[:GA_CONFIG["population_size"]]
    final_fitnesses = [calculate_fitness_event(chrom, router, pending_patient_ids, valid_available_amb_ids) for chrom in population]
    valid_final_pop_fitness = [(p, f) for p, f in zip(population, final_fitnesses) if f != float('inf')]
    if not valid_final_pop_fitness: print("Dispatch Cycle Error: No valid final plan."); return
    best_plan_chromosome = min(valid_final_pop_fitness, key=lambda x: x[1])[0]
    num_assignments = min(num_current_patients, num_available_ambulances); severity_map = {"high": 0, "medium": 1, "low": 2}
    sorted_pending_patient_ids = sorted(pending_patient_ids, key=lambda p_id: severity_map.get(ALL_PATIENTS.get(p_id, {}).get('severity'), 99))
    assignable_patient_ids = sorted_pending_patient_ids[:num_assignments]; patient_indices_in_chromosome = {p_id: pending_patient_ids.index(p_id) for p_id in assignable_patient_ids}
    amb_edges = [traci.vehicle.getRoadID(amb_id) for amb_id in valid_available_amb_ids]; cost_matrix = np.full((num_available_ambulances, num_assignments), float('inf'))
    for i in range(num_available_ambulances):
        for j, patient_id in enumerate(assignable_patient_ids):
            chromosome_idx = patient_indices_in_chromosome[patient_id]; hospital_id = best_plan_chromosome[chromosome_idx]
            patient_info = ALL_PATIENTS.get(patient_id); hospital_info = HOSPITALS.get(hospital_id);
            if not patient_info or not hospital_info: continue
            time_to_patient = router.find_shortest_path_time(amb_edges[i], patient_info["start_edge"])
            time_to_hospital = router.find_shortest_path_time(patient_info["start_edge"], hospital_info["dest_edge"])
            if time_to_patient != float('inf') and time_to_hospital != float('inf'):
                mission_time = time_to_patient + time_to_hospital; weighted_time = mission_time * SEVERITY_WEIGHTS.get(patient_info.get("severity", "medium"), 1.0)
                cost_matrix[i, j] = weighted_time
    try:
        if np.all(cost_matrix == float('inf')): amb_indices, assigned_pat_indices_in_matrix = [],[]
        else: amb_indices, assigned_pat_indices_in_matrix = linear_sum_assignment(cost_matrix)
    except ValueError: print("Dispatch Cycle Error: Final assignment failed."); return
    assigned_ambulance_ids_this_cycle = set(); dispatched_patient_ids_this_cycle = set()
    print("--- Executing Optimal Assignments ---")
    for amb_idx, matrix_pat_idx in zip(amb_indices, assigned_pat_indices_in_matrix):
        if cost_matrix[amb_idx, matrix_pat_idx] == float('inf'): continue
        amb_id = valid_available_amb_ids[amb_idx]; patient_id = assignable_patient_ids[matrix_pat_idx]
        chromosome_idx = patient_indices_in_chromosome[patient_id]; hospital_id = best_plan_chromosome[chromosome_idx]
        if amb_id in assigned_ambulance_ids_this_cycle or patient_id not in pending_patient_ids: continue
        if HOSPITALS[hospital_id]["available_beds"] <= 0: print(f"Skipping assignment: No beds at {hospital_id} for {patient_id}"); continue
        patient_edge = ALL_PATIENTS[patient_id]["start_edge"]; hospital_edge = HOSPITALS[hospital_id]["dest_edge"]
        beds_before = HOSPITALS[hospital_id]["available_beds"]; dispatch_detail = {'patient_id': patient_id, 'ambulance_id': amb_id, 'hospital_id': hospital_id,'beds_at_dispatch': beds_before, 'dispatch_time': current_time}
        try:
            traci.vehicle.changeTarget(amb_id, patient_edge)
            # DO NOT set speed mode here, let the proximity check handle it
            active_missions[amb_id] = {"stage": "to_patient", "patient_id": patient_id, "patient_edge": patient_edge, "hospital_edge": hospital_edge, "hospital_id": hospital_id}
            HOSPITALS[hospital_id]["available_beds"] -= 1; ALL_PATIENTS[patient_id]["status"] = "assigned"
            assigned_ambulance_ids_this_cycle.add(amb_id); dispatched_patient_ids_this_cycle.add(patient_id); generate_dispatch_report_event(dispatch_detail)
            print(f"  Assigned: {amb_id} -> {patient_id} ({ALL_PATIENTS[patient_id]['severity']}) -> {hospital_id} ({HOSPITALS[hospital_id]['name']})")
        except traci.TraCIException as e:
             print(f"Error dispatching {amb_id} -> {patient_id}: {e}. Reverting."); HOSPITALS[hospital_id]["available_beds"] += 1; ALL_PATIENTS[patient_id]["status"] = "pending"
             if amb_id in active_missions and active_missions[amb_id]["patient_id"] == patient_id: del active_missions[amb_id]
    available_ambulance_ids[:] = [amb for amb in available_ambulance_ids if amb not in assigned_ambulance_ids_this_cycle]
    pending_patient_ids[:] = [pat for pat in pending_patient_ids if pat not in dispatched_patient_ids_this_cycle]
    print(f"--- Dispatch Cycle End: {len(pending_patient_ids)} pending, {len(available_ambulance_ids)} available ---")


# --- Main Simulation Entry Point ---
def run_simulation():
    sumo_cmd = [os.path.join(os.environ.get("SUMO_HOME", "."), "bin", "sumo-gui"),
                "-c", "hexagon.sumocfg",
                "--tripinfo-output", "tripinfo_results.xml",
                "--start", # Start paused
                "--quit-on-end"
               ]
    traci.start(sumo_cmd)

    # Clear/Create summary file & Reset counter
    with open("dispatch_summary.txt", "w") as f: f.write("")
    global mission_log_counter; mission_log_counter = 1

    # Add dynamic POIs
    print("Adding dynamic patient POIs...")
    for patient_id, info in ALL_PATIENTS.items():
        try:
            edge_id = info['start_edge']; lane_id = edge_id + "_0"; length = traci.lane.getLength(lane_id)
            x, y = traci.simulation.convert2D(edge_id, length / 2)
            traci.poi.add(patient_id, x, y, color=(255,0,0,255), poiType="patient", layer=10)
        except traci.TraCIException as e: print(f"Warn: POI add failed {patient_id}: {e}")

    try: router = DijkstraForSUMO('hexagon.net.xml')
    except Exception as e: print(f"CRIT ERR Dijkstra: {e}"); traci.close(); sys.exit(1)

    # Event-Driven State Variables
    pending_patients = list(ALL_PATIENTS.keys())
    available_ambulances = list(AMBULANCES)
    active_missions = {}
    needs_dispatch_run = True

    # Reset hospital beds
    for h_id in HOSPITALS: HOSPITALS[h_id]["available_beds"] = HOSPITALS[h_id]["initial_beds"]

    # Initialize TLS states
    global tls_current_phase_indices, tls_phase_start_times, tls_states
    tls_current_phase_indices = {}; tls_phase_start_times = {}; tls_states = {}
    # Initialize prioritized vehicles state
    global prioritized_vehicles; prioritized_vehicles = {}

    # --- Main Event Loop ---
    step = 0; max_steps = 1000; cooldown_start = -1
    while step < max_steps:
         try:
              traci.simulationStep()
              current_time = traci.simulation.getTime()

              # --- MAIN LOOP ORDER ---
              # 1. Check ambulance proximity & manage priority state (activate/deactivate visuals, setSpeedMode)
              check_ambulance_proximity_and_manage_priority(active_missions, current_time)

              # 2. Run Actuated Traffic Light Logic for ALL lights (will skip if priority active)
              for tls_id in TLS_IDS:
                   # Pass the *correct* priority state dictionary
                   run_adaptive_tls_logic(tls_id, current_time, prioritized_vehicles)

              # 3. Update active ambulance missions (movement, pickup timer, delivery check)
              ambulance_became_free = update_active_missions(active_missions, available_ambulances, router)

              # 4. Check for new patients (placeholder)
              new_patient_added = False

              # 5. Trigger dispatch cycle if needed
              if (needs_dispatch_run or ambulance_became_free or new_patient_added) and pending_patients and available_ambulances:
                  current_vehicle_ids = traci.vehicle.getIDList()
                  valid_available_amb_ids = [a for a in available_ambulances if a in current_vehicle_ids]
                  if valid_available_amb_ids:
                      run_dispatch_cycle(pending_patients, valid_available_amb_ids, active_missions, router)
                      needs_dispatch_run = False

              # 6. Check end condition with cooldown
              if not pending_patients and not active_missions and step > 10:
                   if cooldown_start == -1: print(f"All missions done T={current_time:.1f}s. Cooldown."); cooldown_start = current_time
                   if current_time >= cooldown_start + 10: print("Cooldown finished. Ending."); break
              else: cooldown_start = -1

              step += 1
         except traci.FatalTraCIError: print("TraCI connection lost."); break
         except Exception as e: print(f"!! Runtime Error step {step}: {e}"); break

    print(f"\nSimulation ended at time: {traci.simulation.getTime():.2f}")
    traci.close()

if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ: sys.exit("Declare SUMO_HOME.")
    run_simulation()