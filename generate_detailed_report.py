import xml.etree.ElementTree as ET
import re
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Constants ---
PICKUP_TIME = 2.0 # Seconds
# CONGESTION_FACTOR = 1.20 # No longer needed, using detector data

# --- Simulation Data (Fixed Hospital/Ambulance Info) ---
# (HOSPITALS, AMBULANCES remain the same)
HOSPITALS = {
    "H-01": {"name": "City_General", "dest_edge": "e6"},
    "H-02": {"name": "Green_Heart", "dest_edge": "e2"},
    "H-04": {"name": "Tumakuru_Trauma", "dest_edge": "e4"}
}
AMBULANCES = ["amb_1", "amb_2", "amb_3", "amb_4", "amb_5", "amb_6", "amb_7", "amb_8"]

# --- NEW: Function to parse detector data ---
def parse_detector_output(detector_file="detector_output.xml"):
    """ Parses the detector output file to get average speeds per edge/lane. """
    avg_speeds = {}
    try:
        tree = ET.parse(detector_file)
        root = tree.getroot()
        # Process intervals - simplistic approach: average over all intervals for now
        # More advanced: could average over relevant time periods if needed
        detector_data = {} # {detector_id: [speeds]}
        for interval in root.findall('interval'):
            det_id = interval.get('id')
            mean_speed = float(interval.get('meanSpeed', -1.0))
            if mean_speed >= 0: # Valid speed reading
                if det_id not in detector_data:
                    detector_data[det_id] = []
                detector_data[det_id].append(mean_speed)

        # Calculate average speed for each detector
        for det_id, speeds in detector_data.items():
             if speeds:
                  avg_speeds[det_id] = sum(speeds) / len(speeds)

    except FileNotFoundError:
        print(f"Warning: Detector file '{detector_file}' not found. Using ideal times.")
    except (ET.ParseError, ValueError) as e:
        print(f"Warning: Error parsing detector file '{detector_file}': {e}. Using ideal times.")
    return avg_speeds

# --- Dijkstra Class Modified to Use Detector Data ---
class DijkstraForSUMO:
    """Calculates shortest path using detector-based average travel times."""
    def __init__(self, net_file, avg_speeds):
        self.net_file = net_file
        self.avg_speeds = avg_speeds # Store the parsed detector speeds
        self.graph, self.edge_to_junctions, self.junction_pair_to_edge = {}, {}, {}
        self.edge_details = {} # Stores {'length': L, 'ideal_time': T_ideal}
        self._build_graph()

    def _get_detector_id_for_edge(self, edge_id):
        """ Simple helper to find a matching detector ID (adjust if naming differs) """
        # Assumes detector ID might be like 'det_edge_id_0' or similar based on your file
        possible_det_id = f"det_{edge_id}_0" # Common pattern
        if possible_det_id in self.avg_speeds:
            return possible_det_id
        # Add other potential naming patterns if needed
        # Fallback: check if edge_id itself matches a detector id (less likely)
        if edge_id in self.avg_speeds:
             return edge_id
        return None

    def _build_graph(self):
        # --- Parses network like before, but calculates weights differently ---
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            junctions_with_edges = set()
            max_speed_on_network = 0 # Find max speed for penalty calculation

            # First pass: Get basic details and find max speed
            for edge in root.findall('edge'):
                if edge.get('function') != 'internal':
                    edge_id = edge.get('id')
                    from_node, to_node = edge.get('from'), edge.get('to')
                    lane = edge.find('lane')
                    if lane is not None and from_node and to_node:
                        length = float(lane.get('length')); speed = float(lane.get('speed'))
                        ideal_travel_time = length / speed if speed > 0 else float('inf')
                        self.edge_details[edge_id] = {'length': length, 'ideal_time': ideal_travel_time}
                        if speed > max_speed_on_network: max_speed_on_network = speed
                        # Store basic graph structure info
                        junctions_with_edges.add(from_node); junctions_with_edges.add(to_node)
                        self.edge_to_junctions[edge_id] = (from_node, to_node)
                        self.junction_pair_to_edge[(from_node, to_node)] = edge_id

            if max_speed_on_network <= 0: max_speed_on_network = 13.89 # Default if needed

            # Second pass: Build graph weights using detector data or penalties
            for edge_id, details in self.edge_details.items():
                from_node, to_node = self.edge_to_junctions[edge_id]
                length = details['length']
                ideal_time = details['ideal_time']
                edge_weight = float('inf')

                detector_id = self._get_detector_id_for_edge(edge_id)
                avg_speed = self.avg_speeds.get(detector_id) if detector_id else None

                if avg_speed is not None and avg_speed > 0.1: # Use detector speed if valid
                    edge_weight = length / avg_speed
                elif ideal_time != float('inf'):
                    # No detector or very low speed: Penalize based on ideal time
                    # Make it significantly worse than ideal, maybe 3-5x?
                    edge_weight = ideal_time * 3.0 # Penalty factor
                else:
                    edge_weight = float('inf') # Unusable edge

                # Add to graph
                if from_node not in self.graph: self.graph[from_node] = {}
                self.graph[from_node][to_node] = max(1.0, edge_weight) # Ensure minimum weight > 0

            # Ensure all nodes are graph keys
            for node in junctions_with_edges:
                 if node not in self.graph: self.graph[node] = {}
        except FileNotFoundError: raise ValueError(f"Net file '{self.net_file}' not found.")
        except ET.ParseError as e: raise ValueError(f"Failed to parse net file '{self.net_file}': {e}")

    def find_shortest_path_using_avg_times(self, start_edge, end_edge):
        """
        Finds the shortest path based on DETECTOR AVERAGE travel times.
        Returns: (list_of_edges, total_estimated_distance, total_estimated_time)
        Returns (None, float('inf'), float('inf')) if no path found.
        Estimated time INCLUDES the edge weights (detector-based) but NOT pickup.
        Estimated distance is the length of THIS path.
        """
        if not start_edge or not end_edge: return None, float('inf'), float('inf')

        # Handle trivial case
        if start_edge == end_edge:
             details = self.edge_details.get(start_edge)
             if details:
                 # Calculate estimated time for this single edge
                 detector_id = self._get_detector_id_for_edge(start_edge)
                 avg_speed = self.avg_speeds.get(detector_id)
                 length = details.get('length', 0)
                 ideal_time = details.get('ideal_time', float('inf'))
                 est_time = float('inf')
                 if avg_speed is not None and avg_speed > 0.1:
                      est_time = length / avg_speed
                 elif ideal_time != float('inf'):
                      est_time = ideal_time * 3.0 # Penalty factor
                 
                 return ([start_edge], length, max(1.0, est_time))
             else: return (None, float('inf'), float('inf'))

        # Get start/end junctions
        start_node_tuple = self.edge_to_junctions.get(start_edge)
        end_node_tuple = self.edge_to_junctions.get(end_edge)
        if not start_node_tuple or not end_node_tuple: return None, float('inf'), float('inf')
        start_junction, end_junction = start_node_tuple[1], end_node_tuple[0]
        if start_junction not in self.graph: return None, float('inf'), float('inf')

        # Run Dijkstra using ESTIMATED edge times (from detector averages) as weights
        distances = {node: float('inf') for node in self.graph}; distances[start_junction] = 0
        previous_nodes = {node: None for node in self.graph}; pq = [(0, start_junction)]
        path_found = False
        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances.get(current_node, float('inf')): continue
            if current_node == end_junction: path_found = True; break
            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items(): # weight is estimated time
                    if neighbor in distances:
                         distance = dist + weight
                         if distance < distances[neighbor]:
                              distances[neighbor] = distance; previous_nodes[neighbor] = current_node; heapq.heappush(pq, (distance, neighbor))

        if not path_found or distances.get(end_junction, float('inf')) == float('inf'): return None, float('inf'), float('inf')

        # Reconstruct path nodes
        path_nodes = []; current = end_junction
        while current is not None: path_nodes.insert(0, current); current = previous_nodes.get(current)
        if not path_nodes or path_nodes[0] != start_junction: return None, float('inf'), float('inf')

        # --- Calculate Path Edges, Total Estimated Distance, and Total Estimated Time ---
        junction_path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        valid_junction_edges = [edge for edge in junction_path_edges if edge]
        full_edge_path = [start_edge] + valid_junction_edges
        if start_edge != end_edge and end_edge not in valid_junction_edges:
             full_edge_path.append(end_edge)

        total_est_dist = 0
        total_est_time = 0 # Sum of edge weights along the found path
        path_valid = True
        for edge in full_edge_path:
             details = self.edge_details.get(edge)
             if not details: path_valid = False; break
             total_est_dist += details.get('length', 0)

             # Get the weight used in the graph for this edge
             from_j, to_j = self.edge_to_junctions.get(edge, (None, None))
             edge_weight = self.graph.get(from_j, {}).get(to_j, float('inf'))

             if edge_weight == float('inf'): path_valid = False; break
             total_est_time += edge_weight

        if not path_valid:
            return None, float('inf'), float('inf')

        return full_edge_path, total_est_dist, total_est_time # Time here DOES NOT include pickup yet
def parse_dispatch_summary(filename="dispatch_summary.txt"):
    """Parses the event-driven dispatch summary, including patient location."""
    missions = []
    try:
        with open(filename, "r") as f: content = f.read()
        pattern = re.compile(
            r"--- Mission (\d+) \(Dispatched at T=([\d\.]+)s\) ---"
            r".*?Ambulance:\s*(\w+)"
            r".*?Patient Details:.*?- ID:\s+(\w+)"
            r".*?Location:\s+Edge\s+'([^']+)'"  # Capture the edge name here
            r".*?Assigned Hospital:.*?- ID:\s+([\w-]+)"
            r".*?Beds.*?:\s*(\d+)\s*->\s*(\d+)",
            re.DOTALL
        )
        matches = pattern.finditer(content)
        for match in matches:
            mission_data = {
                'mission_num': int(match.group(1)),
                'dispatch_time': float(match.group(2)),
                'ambulance_id': match.group(3),
                'patient_id': match.group(4),
                'patient_start_edge': match.group(5), # Store the captured edge
                'hospital_id': match.group(6),
                'beds_before': int(match.group(7)),
                'beds_after': int(match.group(8)),
                'beds_info': f"{match.group(7)} -> {match.group(8)}"
            }
            missions.append(mission_data)
    except FileNotFoundError: print(f"Error: File '{filename}' not found."); return None
    except Exception as e: print(f"Error parsing dispatch summary '{filename}': {e}"); return None
    if not missions: print(f"Warning: No missions found in {filename}.")
    return missions

# --- Make sure other function definitions like parse_tripinfo follow ---
def parse_tripinfo(filename, vehicle_ids_of_interest=None):
    """Parses tripinfo XML. If vehicle_ids_of_interest is None, parses all."""
    results = {}
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        for trip in root.findall('tripinfo'):
            veh_id = trip.get('id')
            if vehicle_ids_of_interest is None or veh_id in vehicle_ids_of_interest:
                try:
                     results[veh_id] = {
                         'id': veh_id, 'depart': float(trip.get('depart', 0.0)),
                         'arrival': float(trip.get('arrival', 0.0)), 'duration': float(trip.get('duration', 0.0)),
                         'routeLength': float(trip.get('routeLength', 0.0)), 'timeLoss': float(trip.get('timeLoss', 0.0)),
                         'waitingTime': float(trip.get('waitingTime', 0.0)), 'vType': trip.get('vType', 'unknown')
                     }
                except (ValueError, TypeError) as e: print(f"Warning: Parse error for {veh_id} in {filename}: {e}")
    except FileNotFoundError: print(f"Error: File '{filename}' not found."); return {}
    except ET.ParseError as e: print(f"Error: XML parse error in '{filename}': {e}"); return {}
    return results

# --- Plotting functions ---
def plot_actual_vs_estimated_bars(mission_data):
    """
    Creates bar charts comparing Actual vs. Estimated Time (Detector Avg)
    and Actual vs Estimated Distance (Detector Path) for each completed ambulance mission.
    """
    valid_missions = [m for m in mission_data if m['actual_duration'] is not None and m['estimated_time'] != float('inf')]
    if not valid_missions:
        print("No valid mission data to plot for Actual vs. Estimated.")
        return

    labels = [f"{m['ambulance_id']}\n(P{m['patient_id'][-2:]})" for m in valid_missions]
    estimated_times = [m['estimated_time'] for m in valid_missions]
    actual_times = [m['actual_duration'] for m in valid_missions]
    # 'ideal_dist' key now holds the estimated distance based on detector path
    estimated_dists = [m['ideal_dist'] for m in valid_missions]
    actual_dists = [m['actual_distance'] for m in valid_missions]
    x = np.arange(len(labels)); width = 0.35

    # --- Plot 1: Time Comparison ---
    fig1, ax1 = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    rects1 = ax1.bar(x - width/2, estimated_times, width, label='Estimated Time (Detector Avg)', color='lightblue') # Updated Label
    rects2 = ax1.bar(x + width/2, actual_times, width, label='Actual Time', color='coral')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Ambulance Mission Time: Actual vs. Estimated (Detector Avg)') # Updated Title
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right"); ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(rects1, padding=3, fmt='%.0f'); ax1.bar_label(rects2, padding=3, fmt='%.0f')
    fig1.tight_layout(); plt.savefig("time_comparison.png"); plt.close(fig1)

    # --- Plot 2: Distance Comparison ---
    fig2, ax2 = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    rects3 = ax2.bar(x - width/2, estimated_dists, width, label='Estimated Distance (Detector Path)', color='lightgreen') # Updated Label
    rects4 = ax2.bar(x + width/2, actual_dists, width, label='Actual Distance', color='mediumpurple')
    ax2.set_ylabel('Distance (meters)')
    ax2.set_title('Ambulance Mission Distance: Actual vs. Estimated (Detector Path)') # Updated Title
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right"); ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(rects3, padding=3, fmt='%.0f'); ax2.bar_label(rects4, padding=3, fmt='%.0f')
    fig2.tight_layout(); plt.savefig("distance_comparison.png"); plt.close(fig2)


def plot_pie_charts(df_results):
    """Plots pie charts for ambulance trip time breakdown."""
    amb_data = df_results[df_results['id'].str.contains('amb_', na=False)]
    if amb_data.empty: print("No ambulance data in DataFrame to plot pie charts."); return
    num_amb = len(amb_data); ncols = min(num_amb, 4); nrows = (num_amb + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle('Ambulance Trip Time Breakdown', fontsize=16); axes_flat = axes.flatten()
    for i, (idx, amb) in enumerate(amb_data.iterrows()):
        traffic_waiting_time = max(0, amb['waitingTime'] - PICKUP_TIME)
        driving_time = max(0, amb['duration'] - amb['waitingTime'])
        pickup_time_val = PICKUP_TIME if amb['waitingTime'] >= PICKUP_TIME else amb['waitingTime']
        if driving_time + traffic_waiting_time + pickup_time_val <= 0: sizes = [1, 0, 0]
        else: sizes = [driving_time, traffic_waiting_time, pickup_time_val]
        labels = 'Driving Time', 'Traffic Waiting', 'Pickup Time'; colors = ['skyblue', 'lightcoral', 'lightgrey']
        valid_indices = [k for k, size in enumerate(sizes) if size > 0.01]
        sizes = [sizes[k] for k in valid_indices]; labels = [labels[k] for k in valid_indices]; colors = [colors[k] for k in valid_indices]
        ax = axes_flat[i]
        if sizes:
             ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
             ax.set_title(f"{amb['id']}\nTotal: {amb['duration']:.1f}s"); ax.axis('equal')
        else: ax.set_title(f"{amb['id']}\nNo Data"); ax.axis('off')
    for j in range(i + 1, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig("pie_charts.png"); plt.close(fig)


def plot_ambulance_histogram(df_results):
    """Plots histogram of *ambulance* trip durations."""
    amb_durations = df_results[df_results['id'].str.contains('amb_', na=False)]['duration'].dropna()
    if amb_durations.empty: print("No ambulance data in DataFrame for histogram."); return
    plt.figure(figsize=(10, 6)); plt.hist(amb_durations, bins=8, color='lightgreen', edgecolor='black')
    plt.xlabel('Ambulance Mission Duration (seconds)'); plt.ylabel('Number of Ambulances')
    plt.title('Distribution of Ambulance Mission Durations'); plt.grid(axis='y', alpha=0.75)
    plt.tight_layout(); plt.savefig("duration_histogram.png"); plt.close()


def main():
    # File Paths
    net_file = 'hexagon.net.xml'; summary_file = 'dispatch_summary.txt'
    results_file = 'tripinfo_results.xml'; rou_file = 'hexagon.rou.xml'
    detector_file = 'detector_output.xml' # <-- Add detector file

    print(f"--- Generating Detailed Dispatch Report from {summary_file} & {results_file} ---")

    # --- Parse Input Files ---
    missions = parse_dispatch_summary(summary_file)
    actual_results_dict = parse_tripinfo(results_file)
    avg_speeds = parse_detector_output(detector_file) # <-- Parse detector data

    if not missions: print("Aborting: Could not parse dispatch summary."); return
    if not actual_results_dict: print("Aborting: Could not parse tripinfo results."); return

    # Pass avg_speeds to the router
    try: router = DijkstraForSUMO(net_file, avg_speeds)
    except Exception as e: print(f"Aborting: Failed to init Dijkstra: {e}"); return

    ambulance_starts = {}
    # ... (Parse rou_file - unchanged) ...
    try:
        rou_tree = ET.parse(rou_file)
        for vehicle in rou_tree.findall('.//vehicle'):
            vehicle_id = vehicle.get('id')
            if vehicle_id in AMBULANCES:
                route_elem = vehicle.find('route')
                if route_elem is not None:
                     edges_str = route_elem.get('edges')
                     if edges_str: ambulance_starts[vehicle_id] = edges_str.split()[0]
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"Warning: Could not parse {rou_file} for ambulance start locations: {e}")


    # --- Generate Text Report & Collect Data for Plots ---
    print("\n" + "="*60); print("      Detailed Mission Analysis (Actual vs. Estimated)"); print("="*60)
    mission_plot_data = []

    for mission in missions:
        amb_id = mission['ambulance_id']; patient_id = mission['patient_id']
        hospital_id = mission['hospital_id']; patient_edge = mission['patient_start_edge']
        actual_data = actual_results_dict.get(amb_id)
        hospital_info = HOSPITALS.get(hospital_id); amb_start_edge = ambulance_starts.get(amb_id)

        print(f"\n--- Mission {mission['mission_num']} (Patient {patient_id} by {amb_id}, Dispatched T={mission['dispatch_time']:.1f}s) ---")
        if not hospital_info: print(f"  ERROR: Hospital ID {hospital_id} not found."); continue
        if not patient_edge: print(f"  ERROR: Patient start edge missing."); continue

        hospital_edge = hospital_info['dest_edge']
        print(f"  Patient Location: Edge '{patient_edge}'")
        print(f"  Assigned Hospital: {hospital_info.get('name', 'N/A')} ({hospital_id}), Beds: {mission['beds_info']}")

        # --- Calculation using Detector-based Times ---
        final_estimated_distance = float('inf')
        final_estimated_time = float('inf')

        if amb_start_edge:
            # Find path and times based on average detector data
            path_sp, est_dist_sp, est_time_sp_travel = router.find_shortest_path_using_avg_times(amb_start_edge, patient_edge)
            path_ph, est_dist_ph, est_time_ph_travel = router.find_shortest_path_using_avg_times(patient_edge, hospital_edge)

            if est_time_sp_travel != float('inf') and est_time_ph_travel != float('inf'):
                 # Estimated Distance = Sum of segment distances along path found
                 final_estimated_distance = est_dist_sp + est_dist_ph
                 # Estimated Time = Sum of estimated segment travel times + Pickup Time
                 final_estimated_time = est_time_sp_travel + est_time_ph_travel + PICKUP_TIME
            else:
                 print("  Warning: Could not calculate full estimated path (Start->Patient->Hospital) using detector data.")
                 # Handle partial paths if needed (similar logic to previous version)

        else:
            print(f"  Warning: Could not find start edge for {amb_id}.")
        # --- End of Calculation Update ---

        mission_plot_data.append({
            "ambulance_id": amb_id, "patient_id": patient_id,
            "estimated_time": final_estimated_time, # Store detector-based estimated time
            "ideal_dist": final_estimated_distance, # Store distance of this path (might not be shortest ideal geom.)
            "actual_duration": actual_data['duration'] if actual_data else None,
            "actual_distance": actual_data['routeLength'] if actual_data else None
        })

        if actual_data:
            print(f"  Actual Performance:")
            print(f"    - Duration: {actual_data['duration']:.2f}s | Distance: {actual_data['routeLength']:.2f}m | Wait: {actual_data['waitingTime']:.2f}s | Loss: {actual_data['timeLoss']:.2f}s")
            print(f"  Comparison:")
            # Use updated labels
            print(f"    - Est. Time (Detector Avg): {final_estimated_time:.2f}s vs Actual Time : {actual_data['duration']:.2f}s (Diff: {actual_data['duration'] - final_estimated_time:+.2f}s)")
            print(f"    - Est. Dist (Detector Path) : {final_estimated_distance:.2f}m vs Actual Dist : {actual_data['routeLength']:.2f}m (Diff: {actual_data['routeLength'] - final_estimated_distance:+.2f}m)") # Changed label
        else:
            print("  ERROR: No tripinfo found for this ambulance in results file.")

    print("\n" + "="*60)

    # --- Generate Visual Plots ---
    print("\n--- Generating Visual Plots ---")
    df_results = pd.DataFrame(list(actual_results_dict.values()))
    if df_results.empty: print("Aborting plots: No data loaded into DataFrame."); return

    # Update plot function call to reflect new data meaning
    plot_actual_vs_estimated_bars(mission_plot_data) # This function's labels need updating
    plot_pie_charts(df_results)
    plot_ambulance_histogram(df_results)

    print("\nPlots saved as time_comparison.png, distance_comparison.png, pie_charts.png, duration_histogram.png")


# --- Need to update plot function labels ---
def plot_actual_vs_estimated_bars(mission_data):
    """
    Creates bar charts comparing Actual vs. Estimated Time (Detector Avg)
    and Actual vs Estimated Distance (Detector Path) for each completed ambulance mission.
    """
    valid_missions = [m for m in mission_data if m['actual_duration'] is not None and m['estimated_time'] != float('inf')]
    if not valid_missions:
        print("No valid mission data to plot for Actual vs. Estimated.")
        return

    labels = [f"{m['ambulance_id']}\n(P{m['patient_id'][-2:]})" for m in valid_missions]
    estimated_times = [m['estimated_time'] for m in valid_missions]
    actual_times = [m['actual_duration'] for m in valid_missions]
    # 'ideal_dist' key now holds the estimated distance based on detector path
    estimated_dists = [m['ideal_dist'] for m in valid_missions]
    actual_dists = [m['actual_distance'] for m in valid_missions]
    x = np.arange(len(labels)); width = 0.35

    # --- Plot 1: Time Comparison ---
    fig1, ax1 = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    rects1 = ax1.bar(x - width/2, estimated_times, width, label='Estimated Time (Detector Avg)', color='lightblue') # Updated Label
    rects2 = ax1.bar(x + width/2, actual_times, width, label='Actual Time', color='coral')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Ambulance Mission Time: Actual vs. Estimated (Detector Avg)') # Updated Title
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right"); ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(rects1, padding=3, fmt='%.0f'); ax1.bar_label(rects2, padding=3, fmt='%.0f')
    fig1.tight_layout(); plt.savefig("time_comparison.png"); plt.close(fig1)

    # --- Plot 2: Distance Comparison ---
    fig2, ax2 = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    rects3 = ax2.bar(x - width/2, estimated_dists, width, label='Estimated Distance (Detector Path)', color='lightgreen') # Updated Label
    rects4 = ax2.bar(x + width/2, actual_dists, width, label='Actual Distance', color='mediumpurple')
    ax2.set_ylabel('Distance (meters)')
    ax2.set_title('Ambulance Mission Distance: Actual vs. Estimated (Detector Path)') # Updated Title
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right"); ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(rects3, padding=3, fmt='%.0f'); ax2.bar_label(rects4, padding=3, fmt='%.0f')
    fig2.tight_layout(); plt.savefig("distance_comparison.png"); plt.close(fig2)


if __name__ == "__main__":
    main()
    print("\nReport generation finished.")