import os
import sys
import traci
import heapq
import random
import xml.etree.ElementTree as ET
from collections import Counter
from scipy.optimize import linear_sum_assignment # Step 3A
import numpy as np # Step 3A

# --- GA CONFIGURATION ---
GA_CONFIG = {
    "population_size": 50, # Reduced for event-driven
    "generations": 5,    # Reduced for event-driven
    "mutation_rate": 0.3,
    "crossover_rate": 0.8,
    "tournament_size": 3,
    "elitism_size": 2
}

# --- SIMULATION DATA (DYNAMIC) ---
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
    """Creates a dictionary of random patients with initial 'pending' status."""
    patients = {}
    if num_patients > len(PATIENT_PROFILES) or num_patients > len(EDGE_POOL):
        raise ValueError("Cannot generate more unique patients than available profiles/edges.")
    chosen_profiles = random.sample(PATIENT_PROFILES, num_patients)
    # Ensure unique edges if num_patients <= len(EDGE_POOL)
    chosen_edges = random.sample(EDGE_POOL, num_patients) if num_patients <= len(EDGE_POOL) else random.choices(EDGE_POOL, k=num_patients)

    for i in range(num_patients):
        patient_id = f"P{i+1:02}"
        profile = chosen_profiles[i]
        patients[patient_id] = {
            "name": profile["name"], "condition": profile["condition"],
            "keywords": profile["keywords"], "start_edge": chosen_edges[i],
            "severity": profile["severity"],
            "status": "pending" # Initial status for event-driven logic
        }
    return patients

ALL_PATIENTS = generate_random_patients(8) # Generate all patients for the simulation run

HOSPITALS_TEMPLATE = { # Template for resetting beds
    "H-01": {"name": "City_General", "specialties": ["General Medicine", "Emergency Care"], "initial_beds": 12, "dest_edge": "e6"},
    "H-02": {"name": "Green_Heart", "specialties": ["Cardiology"], "initial_beds": 5, "dest_edge": "e2"},
    "H-04": {"name": "Tumakuru_Trauma", "specialties": ["Trauma Care", "Orthopedics"], "initial_beds": 3, "dest_edge": "e4"}
}
# Working copy for simulation
HOSPITALS = {h_id: data.copy() for h_id, data in HOSPITALS_TEMPLATE.items()}
# Initialize current beds at start
for h_id in HOSPITALS:
    HOSPITALS[h_id]["available_beds"] = HOSPITALS[h_id]["initial_beds"]

SEVERITY_WEIGHTS = { "high": 1.5, "medium": 1.0, "low": 0.5 }
AMBULANCES = ["amb_1", "amb_2", "amb_3", "amb_4", "amb_5", "amb_6", "amb_7", "amb_8"]

# --- Dijkstra Pathfinding Class ---
class DijkstraForSUMO:
    """Calculates shortest paths using dynamic travel times from TraCI."""
    def __init__(self, net_file):
        self.net_file = net_file
        # Static graph for connectivity, dynamic times fetched via TraCI
        self.graph, self.edge_to_junctions, self.junction_pair_to_edge = {}, {}, {}
        self._build_graph()

    def _build_graph(self):
        """Builds the static graph structure from the network file."""
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            junctions_with_edges = set()
            for edge in root.findall('edge'):
                if edge.get('function') != 'internal':
                    edge_id = edge.get('id')
                    from_node, to_node = edge.get('from'), edge.get('to')
                    if from_node and to_node: # Basic check for valid nodes
                        junctions_with_edges.add(from_node)
                        junctions_with_edges.add(to_node)
                        if from_node not in self.graph: self.graph[from_node] = {}
                        self.graph[from_node][to_node] = 1 # Static weight (connectivity only)
                        self.edge_to_junctions[edge_id] = (from_node, to_node)
                        self.junction_pair_to_edge[(from_node, to_node)] = edge_id
            # Ensure all nodes are graph keys
            for node in junctions_with_edges:
                 if node not in self.graph: self.graph[node] = {}
        except FileNotFoundError: raise ValueError(f"Network file '{self.net_file}' not found.")
        except ET.ParseError as e: raise ValueError(f"Failed to parse network file '{self.net_file}': {e}")

    def find_shortest_path_time(self, start_edge, end_edge):
        """Gets the estimated travel time for the shortest path."""
        if not start_edge or not end_edge: return float('inf')
        if start_edge not in self.edge_to_junctions or end_edge not in self.edge_to_junctions:
            return float('inf')
        _, time = self.find_dynamic_shortest_path(start_edge, end_edge)
        return time

    def find_dynamic_shortest_path(self, start_edge, end_edge):
        """Finds shortest path using Dijkstra with live TraCI travel times."""
        if start_edge == end_edge:
            try: return [start_edge], max(1.0, traci.edge.getTraveltime(start_edge)) # Min time 1s
            except traci.TraCIException: return [start_edge], 1.0

        start_node_tuple = self.edge_to_junctions.get(start_edge)
        end_node_tuple = self.edge_to_junctions.get(end_edge)
        if not start_node_tuple or not end_node_tuple: return None, float('inf')

        start_junction, end_junction = start_node_tuple[1], end_node_tuple[0]
        if start_junction not in self.graph: return None, float('inf')

        distances = {node: float('inf') for node in self.graph}
        if start_junction in distances: distances[start_junction] = 0
        else: return None, float('inf') # Should not happen if graph built correctly

        previous_nodes = {node: None for node in self.graph}
        pq = [(0, start_junction)] # (current_distance, current_node)
        path_found = False

        while pq:
            dist, current_node = heapq.heappop(pq)

            if dist > distances.get(current_node, float('inf')): continue
            if current_node == end_junction: path_found = True; break

            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    edge_id = self.junction_pair_to_edge.get((current_node, neighbor))
                    if edge_id and neighbor in distances: # Check if neighbor is valid
                        try:
                            # Use live travel time as weight
                            weight = traci.edge.getTraveltime(edge_id)
                            # Handle potential negative values from SUMO during congestion onset
                            if weight <= 0: weight = 1.0 # Assign a minimal positive weight
                        except traci.TraCIException:
                            weight = 1000.0 # Assign high cost if edge not in TraCI (e.g., deleted?)
                        distance = dist + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            previous_nodes[neighbor] = current_node
                            heapq.heappush(pq, (distance, neighbor))

        if not path_found or distances.get(end_junction, float('inf')) == float('inf'):
            return None, float('inf')

        # Reconstruct path
        path_nodes = []
        current = end_junction
        while current is not None:
            path_nodes.insert(0, current)
            current = previous_nodes.get(current)

        if not path_nodes or path_nodes[0] != start_junction: return None, float('inf')

        # Convert node path to edge path
        path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        valid_junction_edges = [edge for edge in path_edges if edge]

        # Full path includes start edge, intermediate edges, and end edge
        full_edge_path = [start_edge] + valid_junction_edges + [end_edge]

        # Calculate total time (first edge + path distance to start of end_edge)
        try: first_edge_time = max(1.0, traci.edge.getTraveltime(start_edge))
        except traci.TraCIException: first_edge_time = 1.0
        calculated_path_time = distances.get(end_junction, float('inf'))
        total_time = float('inf') if calculated_path_time == float('inf') else first_edge_time + calculated_path_time

        return full_edge_path, total_time


# --- Event-Driven GA Functions ---
def create_chromosome_event(current_patients):
    hospital_ids = list(HOSPITALS.keys())
    return [random.choice(hospital_ids) for _ in current_patients]

def calculate_fitness_event(chromosome, router, current_patient_ids, current_ambulance_ids):
    specialty_penalty, bed_penalty = 0, 0
    num_current_patients = len(current_patient_ids)
    hospital_assignments = Counter(chromosome) # Hospital assignments for *this* chromosome

    # 1. Penalties
    for i, patient_id in enumerate(current_patient_ids):
        hospital_id = chromosome[i]
        if patient_id not in ALL_PATIENTS or hospital_id not in HOSPITALS: continue
        if not any(spec in HOSPITALS[hospital_id]["specialties"] for spec in ALL_PATIENTS[patient_id]["keywords"]):
            specialty_penalty += 5000 # Mismatched specialty penalty
    for h_id, count in hospital_assignments.items():
        if h_id in HOSPITALS and count > HOSPITALS[h_id]["available_beds"]:
            bed_penalty += 2000 * (count - HOSPITALS[h_id]["available_beds"]) # Over capacity penalty

    # 2. Hungarian Cost Matrix
    current_vehicle_ids = traci.vehicle.getIDList() # Get vehicles currently in simulation
    valid_ambulance_ids = [amb_id for amb_id in current_ambulance_ids if amb_id in current_vehicle_ids]
    num_ambulances = len(valid_ambulance_ids)
    if num_ambulances == 0: return float('inf') # No ambulances, cannot assign

    amb_edges = []
    final_valid_amb_ids = [] # Re-validate after getting edges
    for amb_id in valid_ambulance_ids:
        try: edge = traci.vehicle.getRoadID(amb_id); amb_edges.append(edge); final_valid_amb_ids.append(amb_id)
        except traci.TraCIException: pass # Skip if ambulance disappears during check
    valid_ambulance_ids = final_valid_amb_ids
    num_ambulances = len(valid_ambulance_ids)
    if num_ambulances == 0: return float('inf')

    num_assignments = min(num_current_patients, num_ambulances)
    assignable_patient_ids = current_patient_ids[:num_assignments] # Assign up to num_ambulances

    cost_matrix = np.full((num_ambulances, num_assignments), float('inf'))

    for i in range(num_ambulances): # Ambulance index
        for j, patient_id in enumerate(assignable_patient_ids): # Patient index in matrix column
            hospital_id = chromosome[j]
            if patient_id not in ALL_PATIENTS or hospital_id not in HOSPITALS: continue

            patient_info = ALL_PATIENTS[patient_id]
            hospital_info = HOSPITALS[hospital_id]
            time_to_patient = router.find_shortest_path_time(amb_edges[i], patient_info["start_edge"])
            time_to_hospital = router.find_shortest_path_time(patient_info["start_edge"], hospital_info["dest_edge"])

            if time_to_patient != float('inf') and time_to_hospital != float('inf'):
                mission_time = time_to_patient + time_to_hospital
                weighted_time = mission_time * SEVERITY_WEIGHTS.get(patient_info.get("severity", "medium"), 1.0)
                cost_matrix[i, j] = weighted_time

    # 3. Run Hungarian
    try:
        if np.all(cost_matrix == float('inf')): # Check if all costs are infinite
             total_weighted_time = float('inf')
        else:
             amb_indices, assigned_pat_indices = linear_sum_assignment(cost_matrix)
             total_weighted_time = cost_matrix[amb_indices, assigned_pat_indices].sum()
    except ValueError: return float('inf') # Error during assignment

    return total_weighted_time + specialty_penalty + bed_penalty

# --- GA Operators (Selection, Crossover, Mutate) ---
# (Implementations remain the same as previous step)
def selection(population, fitnesses):
    pop_fit_pairs = list(zip(population, fitnesses))
    if not pop_fit_pairs: return population[0] if population else [] # Handle empty population
    k = min(GA_CONFIG["tournament_size"], len(pop_fit_pairs))
    if k == 0: return population[0] if population else []
    tournament = random.sample(pop_fit_pairs, k)
    return min(tournament, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    if random.random() > GA_CONFIG["crossover_rate"] or len(parent1) <= 1:
        return list(parent1), list(parent2) # Return copies
    point = random.randint(1, len(parent1) - 1)
    # Ensure parents are lists for slicing/concatenation
    p1 = list(parent1); p2 = list(parent2)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(chromosome):
    # Mutate a *copy* to avoid modifying elites directly if passed by reference
    mutated_chromosome = list(chromosome)
    if random.random() < GA_CONFIG["mutation_rate"] and mutated_chromosome:
        idx = random.randrange(len(mutated_chromosome))
        mutated_chromosome[idx] = random.choice(list(HOSPITALS.keys()))
    return mutated_chromosome


# --- Mission Management & Dispatch ---
mission_log_counter = 1 # Global counter for report

def generate_dispatch_report_event(dispatch_details):
    """Appends details of a single dispatch event to the summary file."""
    global mission_log_counter
    try:
        with open("dispatch_summary.txt", "a") as f:
            if mission_log_counter == 1:
                f.write("========================================\n")
                f.write("   Event-Driven Dispatch Summary\n")
                f.write("========================================\n\n")

            mission = dispatch_details
            if mission['patient_id'] not in ALL_PATIENTS: return
            patient, hospital = ALL_PATIENTS[mission['patient_id']], HOSPITALS[mission['hospital_id']]

            f.write(f"--- Mission {mission_log_counter} (Dispatched at T={mission['dispatch_time']:.1f}s) ---\n")
            f.write(f"  Ambulance: {mission['ambulance_id']}\n\n")
            f.write(f"  Patient Details:\n    - ID:         {mission['patient_id']}\n    - Name:       {patient['name']}\n    - Condition:  {patient['condition']} (Severity: {patient['severity']})\n    - Location:   Edge '{patient['start_edge']}'\n\n")
            f.write(f"  Assigned Hospital:\n    - ID:         {mission['hospital_id']}\n    - Name:       {hospital['name']}\n    - Specialties:{', '.join(hospital['specialties'])}\n    - Beds (Current Avail. Before): {mission['beds_at_dispatch']} -> {mission['beds_at_dispatch'] - 1}\n") # Show current availability
            f.write(f"    - Destination:Edge '{hospital['dest_edge']}'\n\n----------------------------------------\n\n")
            mission_log_counter += 1
    except Exception as e:
        print(f"Error writing to dispatch_summary.txt: {e}")

# --- Step 4: Implement Traffic Light Preemption ---
def optimize_traffic_lights_for_ambulance(ambulance_id, route_edges):
    """Attempts to set traffic lights green along the ambulance's immediate path."""
    try:
        if not route_edges or len(route_edges) < 1: return

        current_edge = route_edges[0]
        # Junction is at the END of the current edge
        junction_id = traci.edge.getToJunction(current_edge)

        # Check if this junction ID matches any known TLS ID
        tls_ids = traci.trafficlight.getIDList()
        controlled_tls_id = None
        if junction_id in tls_ids:
             controlled_tls_id = junction_id

        # Only proceed if there's a TLS and a next edge in the route
        if controlled_tls_id and len(route_edges) > 1:
            next_edge = route_edges[1]
            links = traci.trafficlight.getControlledLinks(controlled_tls_id)
            if not links: return # No controlled links found for this TLS

            link_index_to_set = -1
            # Find the index corresponding to the connection current_edge -> next_edge
            for i, link_tuple_list in enumerate(links):
                if not link_tuple_list: continue
                for link_tuple in link_tuple_list:
                     if len(link_tuple) >= 3:
                          from_lane, to_lane = link_tuple[0], link_tuple[1]
                          # Check if lanes match the start/end edges
                          if from_lane and to_lane and from_lane.startswith(current_edge) and to_lane.startswith(next_edge):
                               link_index_to_set = i
                               break
                if link_index_to_set != -1: break

            if link_index_to_set != -1:
                # Found the link. Find a phase that makes it green.
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(controlled_tls_id)
                if not logic or not logic[0].phases: return

                best_phase_index = -1
                current_phase_index = traci.trafficlight.getPhase(controlled_tls_id)

                for phase_idx, phase in enumerate(logic[0].phases):
                    if link_index_to_set < len(phase.state):
                        link_state = phase.state[link_index_to_set].lower()
                        if 'g' in link_state: # Look for 'G' or 'g'
                            best_phase_index = phase_idx
                            if link_state == 'g': break # Prefer 'G'

                if best_phase_index != -1 and current_phase_index != best_phase_index:
                    traci.trafficlight.setPhase(controlled_tls_id, best_phase_index)
                    # print(f"TLS PREEMPT: Set {controlled_tls_id} green (Phase {best_phase_index}) for {ambulance_id}")

    except traci.TraCIException as e: pass # Ignore TraCI errors (e.g., no TLS)
    except IndexError as e: pass # Ignore index errors if link index is out of bounds for a phase
    except Exception as e: print(f"Unexpected TLS Preemption Error ({ambulance_id}): {e}")


def update_active_missions(active_missions, available_ambulances, router):
    """Checks status, manages stages, reroutes, and frees up finished ambulances."""
    finished_mission_ambulance_ids = []
    current_time = traci.simulation.getTime()
    ambulance_became_free = False

    for amb_id, mission in list(active_missions.items()):
        current_edge = None
        # --- REPLACE WITH THIS NEW BLOCK ---

        try:
            # Check if ambulance still exists in the simulation
            if amb_id not in traci.vehicle.getIDList():
                if mission["stage"] != "done":
                    print(f"MISSION COMPLETE: {amb_id} finished (vehicle removed) for {mission['patient_id']} at T={current_time:.1f}s.")
                    mission["stage"] = "done"
                    # Restore bed count if vehicle was heading to hospital when it finished
                    if mission.get("stage") == "to_hospital" and mission.get("hospital_id") in HOSPITALS:
                         HOSPITALS[mission["hospital_id"]]["available_beds"] = min(
                              HOSPITALS[mission["hospital_id"]]["available_beds"] + 1,
                              HOSPITALS[mission["hospital_id"]]["initial_beds"]
                         )
                    finished_mission_ambulance_ids.append(amb_id)
                continue

            current_edge = traci.vehicle.getRoadID(amb_id)
            current_route = traci.vehicle.getRoute(amb_id)

            # --- Preemption (Unchanged) ---
            if current_route:
                optimize_traffic_lights_for_ambulance(amb_id, current_route)

            # --- FINAL STAGE MANAGEMENT (MANUAL STOP) ---
            if mission["stage"] == "to_patient":
                if current_edge == mission["patient_edge"]:
                    # Action 1: Arrived. Set speed to 0 and start the timer.
                    print(f"INFO: {amb_id} reached patient {mission['patient_id']} at T={current_time:.1f}s. Stopping for pickup.")
                    traci.vehicle.setSpeed(amb_id, 0) # Command to stop
                    mission["stage"] = "pickup" # New, simplified stage
                    mission["pickup_start_time"] = current_time # Record when stop begins
                
                elif current_time % 5 == 0: # Periodic reroute check
                    traci.vehicle.changeTarget(amb_id, mission["patient_edge"])

            elif mission["stage"] == "pickup":
                # Action: Keep speed at 0 and check if 10 seconds have passed.
                pickup_duration = 10.0
                if current_time >= mission["pickup_start_time"] + pickup_duration:
                    print(f"INFO: {amb_id} finished pickup for {mission['patient_id']}. Proceeding to hospital.")
                    mission["stage"] = "to_hospital"
                    
                    # Action: Restore default speed behavior (allow SUMO to control it again)
                    traci.vehicle.setSpeed(amb_id, -1) # -1 lets SUMO take over
                    
                    # Action: NOW set the new target.
                    traci.vehicle.changeTarget(amb_id, mission["hospital_edge"])
                else:
                    # Ensure speed stays 0 during the wait
                    traci.vehicle.setSpeed(amb_id, 0)

            elif mission["stage"] == "to_hospital":
                if current_edge == mission["hospital_edge"]:
                    print(f"MISSION COMPLETE: {amb_id} delivered patient {mission['patient_id']} at T={current_time:.1f}s.")
                    mission["stage"] = "done"
                    # Restore bed count upon confirmed delivery
                    if mission.get("hospital_id") in HOSPITALS:
                         HOSPITALS[mission["hospital_id"]]["available_beds"] = min(
                              HOSPITALS[mission["hospital_id"]]["available_beds"] + 1,
                              HOSPITALS[mission["hospital_id"]]["initial_beds"]
                         )
                    finished_mission_ambulance_ids.append(amb_id)
                
                elif current_time % 5 == 0: # Periodic reroute check
                   traci.vehicle.changeTarget(amb_id, mission["hospital_edge"])

        except traci.TraCIException as e:
            # ... (Error handling logic remains the same) ...
            print(f"TraCI Error updating mission for {amb_id}: {e}. Marking as done.")
            if mission["stage"] != "done":
                mission["stage"] = "done"
                if mission.get("stage") in ["to_hospital", "pickup"] and mission.get("hospital_id") in HOSPITALS:
                     HOSPITALS[mission["hospital_id"]]["available_beds"] = min(
                          HOSPITALS[mission["hospital_id"]]["available_beds"] + 1,
                          HOSPITALS[mission["hospital_id"]]["initial_beds"]
                     )
                finished_mission_ambulance_ids.append(amb_id)

    # --- Process finished ambulances ---
    for amb_id in finished_mission_ambulance_ids:
        if amb_id in active_missions:
            del active_missions[amb_id]
            if amb_id not in available_ambulances:
                available_ambulances.append(amb_id)
                print(f"EVENT: Ambulance {amb_id} is now available at T={current_time:.1f}s.")
                ambulance_became_free = True

    return ambulance_became_free


def run_dispatch_cycle(pending_patient_ids, available_ambulance_ids, active_missions, router):
    """Runs GA + Hungarian for current state and executes dispatches."""
    if not pending_patient_ids or not available_ambulance_ids: return

    current_time = traci.simulation.getTime()
    print(f"\n--- Running Dispatch Cycle at T={current_time:.1f}s ---")
    print(f"Pending Patients: {len(pending_patient_ids)}, Available Ambulances: {len(available_ambulance_ids)}")

    current_vehicle_ids = traci.vehicle.getIDList()
    valid_available_amb_ids = [amb_id for amb_id in available_ambulance_ids if amb_id in current_vehicle_ids]
    if not valid_available_amb_ids: print("No valid ambulances in sim for dispatch."); return

    num_current_patients = len(pending_patient_ids)
    num_available_ambulances = len(valid_available_amb_ids)

    # --- Run GA ---
    population = [create_chromosome_event(pending_patient_ids) for _ in range(GA_CONFIG["population_size"])]
    for gen in range(GA_CONFIG["generations"]):
        fitnesses = [calculate_fitness_event(chrom, router, pending_patient_ids, valid_available_amb_ids) for chrom in population]
        valid_pop_fitness = [(p, f) for p, f in zip(population, fitnesses) if f != float('inf')]
        if not valid_pop_fitness: continue
        pop_with_fitness = sorted(valid_pop_fitness, key=lambda x: x[1])
        # print(f"  Dispatch Gen {gen+1}, Best Fitness: {pop_with_fitness[0][1]:.2f}") # Optional log

        new_population = [chrom for chrom, fit in pop_with_fitness[:GA_CONFIG["elitism_size"]]]
        valid_population = [p for p, f in pop_with_fitness]
        valid_fitnesses = [f for p, f in pop_with_fitness]
        num_to_generate = GA_CONFIG["population_size"] - GA_CONFIG["elitism_size"]
        if num_to_generate % 2 != 0: num_to_generate -= 1

        for _ in range(num_to_generate // 2):
            k = min(GA_CONFIG["tournament_size"], len(valid_population))
            parent1 = selection(valid_population, valid_fitnesses) if k > 0 else random.choice(valid_population) if valid_population else []
            parent2 = selection(valid_population, valid_fitnesses) if k > 0 else random.choice(valid_population) if valid_population else []
            if not parent1 or not parent2: continue # Skip if selection failed
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        if (GA_CONFIG["population_size"] - GA_CONFIG["elitism_size"]) % 2 != 0 and new_population:
             new_population.append(mutate(new_population[0])) # Mutate copy of best
        population = new_population[:GA_CONFIG["population_size"]]

    # --- Final Assignment ---
    final_fitnesses = [calculate_fitness_event(chrom, router, pending_patient_ids, valid_available_amb_ids) for chrom in population]
    valid_final_pop_fitness = [(p, f) for p, f in zip(population, final_fitnesses) if f != float('inf')]
    if not valid_final_pop_fitness: print("Dispatch Cycle Error: No valid final plan."); return
    best_plan_chromosome = min(valid_final_pop_fitness, key=lambda x: x[1])[0]

    # --- Prepare for Hungarian ---
    num_assignments = min(num_current_patients, num_available_ambulances)
    # Sort pending patients by severity *before* selecting for assignment
    severity_map = {"high": 0, "medium": 1, "low": 2}
    sorted_pending_patient_ids = sorted(pending_patient_ids, key=lambda p_id: severity_map.get(ALL_PATIENTS[p_id]['severity'], 99))
    assignable_patient_ids = sorted_pending_patient_ids[:num_assignments]
    # Map assignable patient ID back to its index in the *original* pending_patient_ids list used for the chromosome
    patient_indices_in_chromosome = {p_id: pending_patient_ids.index(p_id) for p_id in assignable_patient_ids}


    amb_edges = [traci.vehicle.getRoadID(amb_id) for amb_id in valid_available_amb_ids]
    cost_matrix = np.full((num_available_ambulances, num_assignments), float('inf'))

    for i in range(num_available_ambulances):
        for j, patient_id in enumerate(assignable_patient_ids):
            chromosome_idx = patient_indices_in_chromosome[patient_id] # Index in the GA chromosome
            hospital_id = best_plan_chromosome[chromosome_idx]
            if patient_id not in ALL_PATIENTS or hospital_id not in HOSPITALS: continue
            patient_info = ALL_PATIENTS[patient_id]; hospital_info = HOSPITALS[hospital_id]
            time_to_patient = router.find_shortest_path_time(amb_edges[i], patient_info["start_edge"])
            time_to_hospital = router.find_shortest_path_time(patient_info["start_edge"], hospital_info["dest_edge"])
            if time_to_patient != float('inf') and time_to_hospital != float('inf'):
                mission_time = time_to_patient + time_to_hospital
                weighted_time = mission_time * SEVERITY_WEIGHTS.get(patient_info.get("severity", "medium"), 1.0)
                cost_matrix[i, j] = weighted_time

    # --- Run Hungarian ---
    try:
         if np.all(cost_matrix == float('inf')): amb_indices, assigned_pat_indices_in_matrix = [],[]
         else: amb_indices, assigned_pat_indices_in_matrix = linear_sum_assignment(cost_matrix)
    except ValueError: print("Dispatch Cycle Error: Final assignment failed."); return

    # --- Execute Dispatches ---
    assigned_ambulance_ids_this_cycle = set()
    dispatched_patient_ids_this_cycle = set()
    print("--- Executing Optimal Assignments ---")

    for amb_idx, matrix_pat_idx in zip(amb_indices, assigned_pat_indices_in_matrix):
        if cost_matrix[amb_idx, matrix_pat_idx] == float('inf'): continue # Skip invalid assignments

        amb_id = valid_available_amb_ids[amb_idx]
        patient_id = assignable_patient_ids[matrix_pat_idx]
        chromosome_idx = patient_indices_in_chromosome[patient_id]
        hospital_id = best_plan_chromosome[chromosome_idx]

        if amb_id in assigned_ambulance_ids_this_cycle or patient_id not in pending_patient_ids: continue
        if HOSPITALS[hospital_id]["available_beds"] <= 0:
            print(f"Skipping assignment: No beds at {hospital_id} ({HOSPITALS[hospital_id]['name']}) for {patient_id}")
            continue

        patient_edge = ALL_PATIENTS[patient_id]["start_edge"]
        hospital_edge = HOSPITALS[hospital_id]["dest_edge"]
        beds_before = HOSPITALS[hospital_id]["available_beds"]

        dispatch_detail = {
            'patient_id': patient_id, 'ambulance_id': amb_id, 'hospital_id': hospital_id,
            'beds_at_dispatch': beds_before, 'dispatch_time': current_time
        }
        try:
            traci.vehicle.changeTarget(amb_id, patient_edge)
            active_missions[amb_id] = {
                "stage": "to_patient", "patient_id": patient_id,
                "patient_edge": patient_edge, "hospital_edge": hospital_edge,
                "hospital_id": hospital_id
            }
            # Update state immediately
            HOSPITALS[hospital_id]["available_beds"] -= 1
            ALL_PATIENTS[patient_id]["status"] = "assigned"
            assigned_ambulance_ids_this_cycle.add(amb_id)
            dispatched_patient_ids_this_cycle.add(patient_id)
            generate_dispatch_report_event(dispatch_detail)
            print(f"  Assigned: {amb_id} -> {patient_id} ({ALL_PATIENTS[patient_id]['severity']}) -> {hospital_id} ({HOSPITALS[hospital_id]['name']})")

            # Initial preemption call
            current_amb_edge = traci.vehicle.getRoadID(amb_id)
            optimize_traffic_lights_for_ambulance(amb_id, [current_amb_edge, patient_edge])

        except traci.TraCIException as e:
            print(f"Error dispatching {amb_id} -> {patient_id}: {e}. Reverting.")
            HOSPITALS[hospital_id]["available_beds"] += 1
            ALL_PATIENTS[patient_id]["status"] = "pending"
            # Remove from active missions if it was added
            if amb_id in active_missions and active_missions[amb_id]["patient_id"] == patient_id:
                 del active_missions[amb_id]

    # --- Update Global State Lists ---
    # Use list comprehensions for cleaner removal
    available_ambulance_ids[:] = [amb for amb in available_ambulance_ids if amb not in assigned_ambulance_ids_this_cycle]
    pending_patient_ids[:] = [pat for pat in pending_patient_ids if pat not in dispatched_patient_ids_this_cycle]
    print(f"--- Dispatch Cycle End: {len(pending_patient_ids)} patients pending, {len(available_ambulance_ids)} ambulances available ---")


# --- Main Simulation Entry Point ---
def run_simulation():
    """Sets up and runs the event-driven SUMO simulation."""
    sumo_cmd = [os.path.join(os.environ.get("SUMO_HOME", "."), "bin", "sumo-gui"),
                "-c", "hexagon.sumocfg",
                "--tripinfo-output", "tripinfo_results.xml",
                "--start", # Start paused
                "--quit-on-end"
               ]
    traci.start(sumo_cmd)

    # Clear/Create summary file
    with open("dispatch_summary.txt", "w") as f: f.write("")
    global mission_log_counter; mission_log_counter = 1

    # Add dynamic POIs
    print("Adding dynamic patient POIs...")
    for patient_id, info in ALL_PATIENTS.items():
        try:
            edge_id = info['start_edge']
            lane_id = edge_id + "_0"
            length = traci.lane.getLength(lane_id)
            x, y = traci.simulation.convert2D(edge_id, length / 2)
            traci.poi.add(patient_id, x, y, color=(255, 0, 0, 255), poiType="patient", layer=10)
        except traci.TraCIException as e: print(f"Warning: POI add failed for {patient_id}: {e}")

    try: router = DijkstraForSUMO('hexagon.net.xml')
    except Exception as e: print(f"CRITICAL ERROR initializing Dijkstra: {e}"); traci.close(); sys.exit(1)

    # Event-Driven State
    pending_patients = list(ALL_PATIENTS.keys())
    available_ambulances = list(AMBULANCES)
    active_missions = {}
    needs_dispatch_run = True # Flag to run initial dispatch

    # Reset hospital beds
    for h_id in HOSPITALS: HOSPITALS[h_id]["available_beds"] = HOSPITALS[h_id]["initial_beds"]

    # --- Main Event Loop ---
    step = 0
    max_steps = 1000 # Simulation end time from sumocfg
    while step < max_steps: # Use step count or check getMinExpectedNumber
         try:
              traci.simulationStep()
              current_time = traci.simulation.getTime()

              # 1. Update active missions
              ambulance_became_free = update_active_missions(active_missions, available_ambulances, router)

              # 2. Check for new patients (placeholder)
              new_patient_added = False
              # Add logic here if needed

              # 3. Trigger dispatch cycle if needed
              if (needs_dispatch_run or ambulance_became_free or new_patient_added) and pending_patients and available_ambulances:
                  current_vehicle_ids = traci.vehicle.getIDList()
                  valid_available_amb_ids = [amb_id for amb_id in available_ambulances if amb_id in current_vehicle_ids]
                  if valid_available_amb_ids:
                      run_dispatch_cycle(pending_patients, valid_available_amb_ids, active_missions, router)
                      needs_dispatch_run = False # Don't re-run immediately unless another event happens
                  # else: print(f"Time {current_time:.1f}: Dispatch trigger skipped - no valid ambulances in sim.")

              # Check if simulation should end (e.g., all patients served and ambulances idle)
              if not pending_patients and not active_missions and step > 10:
                  if "cooldown_start" not in locals():
                      print("All missions complete. Starting 10s cooldown before shutdown.")
                      cooldown_start = current_time
                  
                  if current_time >= cooldown_start + 10:
                      print("Cooldown finished. Ending simulation.")
                      break

              step += 1

         except traci.FatalTraCIError:
             print("TraCI connection lost. Ending simulation.")
             break
         except Exception as e:
             print(f"!! Runtime Error during simulation step {step}: {e}")
             # Optionally add more detailed error logging or traceback
             break # Stop simulation on unexpected error

    print(f"\nSimulation ended at time: {traci.simulation.getTime():.2f}")
    traci.close()

if __name__ == "__main__":
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'.")
    run_simulation()