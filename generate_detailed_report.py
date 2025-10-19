import xml.etree.ElementTree as ET
import re
import heapq 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # To check if files exist

# --- Data Dictionaries (Should align with traci_runner.py used for the run) ---
# It's best practice to save the actual PATIENTS/HOSPITALS used during the run
# to a file (e.g., JSON) and load it here, instead of hardcoding.
PATIENTS = {
    "P01": {"name": "Ravi Kumar", "start_edge": "e6", "severity": "high"},
    "P02": {"name": "Sita Devi", "start_edge": "e1", "severity": "high"},
    "P03": {"name": "Arjun Singh", "start_edge": "e3_rev", "severity": "medium"},
    "P04": {"name": "Priya Sharma", "start_edge": "e5", "severity": "medium"},
    "P05": {"name": "Amit Patel", "start_edge": "e9", "severity": "high"},
    "P06": {"name": "Anjali Gupta", "start_edge": "e12_rev", "severity": "low"},
    "P07": {"name": "Vikram Rathod", "start_edge": "e8", "severity": "high"},
    "P08": {"name": "Meera Iyer", "start_edge": "e2_rev", "severity": "low"},
    # Add P09, P10 etc. if traci_runner generated more
}
HOSPITALS = {
    "H-01": {"name": "City_General", "dest_edge": "e6"},
    "H-02": {"name": "Green_Heart", "dest_edge": "e2"},
    "H-04": {"name": "Tumakuru_Trauma", "dest_edge": "e4"}
}
AMBULANCES = ["amb_1", "amb_2", "amb_3", "amb_4", "amb_5", "amb_6", "amb_7", "amb_8"]


# --- Dijkstra Pathfinding Class (Static version for ideal estimations) ---
class DijkstraForSUMO:
    """Calculates ideal path time/distance using static network data."""
    def __init__(self, net_file):
        self.net_file = net_file
        self.graph, self.edge_to_junctions, self.junction_pair_to_edge = {}, {}, {}
        self.edge_details = {}
        self._build_graph()

    def _build_graph(self):
        try:
            tree = ET.parse(self.net_file)
            root = tree.getroot()
            junctions_with_edges = set()
            for edge in root.findall('edge'):
                if edge.get('function') != 'internal':
                    edge_id = edge.get('id')
                    from_node, to_node = edge.get('from'), edge.get('to')
                    lane = edge.find('lane')
                    if lane is not None and from_node and to_node:
                        junctions_with_edges.add(from_node); junctions_with_edges.add(to_node)
                        length = float(lane.get('length')); speed = float(lane.get('speed'))
                        travel_time = length / speed if speed > 0 else float('inf')
                        if from_node not in self.graph: self.graph[from_node] = {}
                        self.graph[from_node][to_node] = travel_time # Use ideal time as weight
                        self.edge_to_junctions[edge_id] = (from_node, to_node)
                        self.junction_pair_to_edge[(from_node, to_node)] = edge_id
                        self.edge_details[edge_id] = {'length': length, 'time': travel_time}
            for node in junctions_with_edges:
                 if node not in self.graph: self.graph[node] = {}
        except FileNotFoundError: raise ValueError(f"Net file '{self.net_file}' not found.")
        except ET.ParseError as e: raise ValueError(f"Failed to parse net file '{self.net_file}': {e}")

    def get_path_details(self, start_edge, end_edge):
        """Calculates ideal path time and distance between two edges."""
        if start_edge == end_edge:
             details = self.edge_details.get(start_edge); return (details['time'], details['length']) if details else (float('inf'), float('inf'))
        start_node_tuple = self.edge_to_junctions.get(start_edge); end_node_tuple = self.edge_to_junctions.get(end_edge)
        if not start_node_tuple or not end_node_tuple: return float('inf'), float('inf')
        start_junction, end_junction = start_node_tuple[1], end_node_tuple[0] # Path between junctions
        if start_junction not in self.graph: return float('inf'), float('inf')

        distances = {node: float('inf') for node in self.graph}; distances[start_junction] = 0
        previous_nodes = {node: None for node in self.graph}; pq = [(0, start_junction)]
        path_found = False
        while pq:
            dist, current_node = heapq.heappop(pq)
            if dist > distances.get(current_node, float('inf')): continue
            if current_node == end_junction: path_found = True; break
            if current_node in self.graph:
                for neighbor, weight in self.graph[current_node].items():
                    if neighbor in distances:
                         distance = dist + weight
                         if distance < distances[neighbor]:
                              distances[neighbor] = distance; previous_nodes[neighbor] = current_node; heapq.heappush(pq, (distance, neighbor))

        if not path_found or distances.get(end_junction, float('inf')) == float('inf'): return float('inf'), float('inf')

        path_nodes = []; current = end_junction
        while current is not None: path_nodes.insert(0, current); current = previous_nodes.get(current)
        if not path_nodes or path_nodes[0] != start_junction: return float('inf'), float('inf')

        junction_path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        valid_junction_edges = [edge for edge in junction_path_edges if edge]
        full_edge_path = [start_edge] + valid_junction_edges + [end_edge] # Full path including start/end edges
        total_dist = sum(self.edge_details.get(edge, {}).get('length', 0) for edge in full_edge_path)
        total_time = sum(self.edge_details.get(edge, {}).get('time', 0) for edge in full_edge_path)
        return total_time, total_dist

# --- Parsing Functions ---
def parse_dispatch_summary(filename="dispatch_summary.txt"):
    """Parses the event-driven dispatch summary."""
    missions = []
    try:
        with open(filename, "r") as f: content = f.read()
        # Regex to capture mission number, time, ambulance, patient, hospital, beds
        pattern = re.compile(
            r"--- Mission (\d+) \(Dispatched at T=([\d\.]+)s\) ---"
            r".*?Ambulance:\s*(\w+)"
            r".*?Patient Details:.*?- ID:\s+(\w+)"
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
                'hospital_id': match.group(5),
                'beds_before': int(match.group(6)),
                'beds_after': int(match.group(7)),
                'beds_info': f"{match.group(6)} -> {match.group(7)}"
            }
            missions.append(mission_data)
    except FileNotFoundError: print(f"Error: File '{filename}' not found."); return None
    except Exception as e: print(f"Error parsing dispatch summary '{filename}': {e}"); return None
    if not missions: print(f"Warning: No missions found in {filename}.")
    return missions

def parse_tripinfo(filename, vehicle_ids_of_interest=None):
    """Parses tripinfo XML. If vehicle_ids_of_interest is None, parses all."""
    results = {}
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        for trip in root.findall('tripinfo'):
            veh_id = trip.get('id')
            # If a specific list is given, check if veh_id is in it
            if vehicle_ids_of_interest is None or veh_id in vehicle_ids_of_interest:
                try:
                     results[veh_id] = {
                         'id': veh_id,
                         'depart': float(trip.get('depart', 0.0)),
                         'arrival': float(trip.get('arrival', 0.0)),
                         'duration': float(trip.get('duration', 0.0)),
                         'routeLength': float(trip.get('routeLength', 0.0)),
                         'timeLoss': float(trip.get('timeLoss', 0.0)),
                         'waitingTime': float(trip.get('waitingTime', 0.0)),
                         'vType': trip.get('vType', 'unknown')
                     }
                except (ValueError, TypeError) as e: print(f"Warning: Parse error for {veh_id} in {filename}: {e}")
    except FileNotFoundError: print(f"Error: File '{filename}' not found."); return {}
    except ET.ParseError as e: print(f"Error: XML parse error in '{filename}': {e}"); return {}
    return results

# --- Plotting Functions ---
def plot_comparison(before_data, after_data, ambulance_ids):
    """Generates bar charts comparing metrics before and after."""
    # (Implementation remains the same as previous step)
    metrics_to_plot = {
        'duration': 'Trip Duration (seconds)', 'timeLoss': 'Time Loss (seconds)',
        'waitingTime': 'Waiting Time (seconds)', 'routeLength': 'Route Length (meters)'
    }
    num_amb = len(ambulance_ids)
    if num_amb == 0: print("No common ambulances for before/after comparison."); return

    for metric, title in metrics_to_plot.items():
        before_values = [before_data.get(amb, {}).get(metric, 0) for amb in ambulance_ids]
        after_values = [after_data.get(amb, {}).get(metric, 0) for amb in ambulance_ids]
        x = np.arange(num_amb); width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, num_amb * 1.5), 6))
        rects1 = ax.bar(x - width/2, before_values, width, label='Before Opt.', color='lightcoral')
        rects2 = ax.bar(x + width/2, after_values, width, label='After Opt.', color='skyblue')
        ax.set_ylabel(title.split('(')[-1].replace(')', '').strip().capitalize())
        ax.set_title(f'Comparison: {title}'); ax.set_xticks(x)
        ax.set_xticklabels(ambulance_ids, rotation=45, ha="right"); ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.bar_label(rects1, padding=3, fmt='%.1f'); ax.bar_label(rects2, padding=3, fmt='%.1f')
        fig.tight_layout(); plt.show()

def plot_pie_charts(df):
    """Plots pie charts for ambulance trip time breakdown."""
    # (Implementation remains the same as previous step)
    amb_data = df[df['id'].str.contains('amb_', na=False)]
    if amb_data.empty: print("No ambulance data for pie charts."); return
    num_amb = len(amb_data); ncols = min(num_amb, 4); nrows = (num_amb + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    fig.suptitle('Ambulance Trip Time Breakdown', fontsize=16); axes_flat = axes.flatten()
    for i, (idx, amb) in enumerate(amb_data.iterrows()):
        driving_time = max(0, amb['duration'] - amb['waitingTime']); waiting_time = amb['waitingTime']
        sizes = [1, 0] if driving_time + waiting_time <= 0 else [driving_time, waiting_time]
        labels = 'Driving Time', 'Waiting Time'; ax = axes_flat[i]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        ax.set_title(f"Ambulance ({amb['id']})\nTotal: {amb['duration']:.1f}s"); ax.axis('equal')
    for j in range(i + 1, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


def plot_car_histogram(df):
    """Plots histogram of car trip durations."""
    # (Implementation remains the same as previous step)
    if 'vType' in df.columns: car_durations = df[df['vType'].str.lower() == 'car']['duration'].dropna()
    else: car_durations = df[~df['id'].str.contains('amb_', na=False)]['duration'].dropna()
    if car_durations.empty: print("No car data for histogram."); return
    plt.figure(figsize=(10, 6))
    plt.hist(car_durations, bins=15, color='lightgreen', edgecolor='black')
    plt.xlabel('Trip Duration (seconds)'); plt.ylabel('Number of Cars')
    plt.title('Distribution of Car Trip Durations'); plt.grid(axis='y', alpha=0.75)
    plt.tight_layout(); plt.show()

# --- Main Report Generation ---
def main():
    # File Paths
    net_file = 'hexagon.net.xml'
    summary_file = 'dispatch_summary.txt'
    results_file = 'tripinfo_results.xml'
    before_file = 'tripinfo_before.xml'
    rou_file = 'hexagon.rou.xml' # Needed for start locations

    print(f"--- Generating Detailed Report from {summary_file} & {results_file} ---")

    # Parse Data
    missions = parse_dispatch_summary(summary_file)
    actual_results_dict = parse_tripinfo(results_file) # Parse all vehicles

    if not missions or not actual_results_dict:
        print("Aborting: Missing required input files."); return

    try: router = DijkstraForSUMO(net_file)
    except Exception as e: print(f"Aborting: Failed to init Dijkstra: {e}"); return

    # Get Ambulance Start Edges
    ambulance_starts = {}
    try:
        rou_tree = ET.parse(rou_file)
        for vehicle in rou_tree.findall('.//vehicle'):
            vehicle_id = vehicle.get('id')
            if vehicle_id in AMBULANCES:
                route_elem = vehicle.find('route')
                if route_elem is not None:
                     edges_str = route_elem.get('edges')
                     if edges_str: ambulance_starts[vehicle_id] = edges_str.split()[0]
    except (FileNotFoundError, ET.ParseError): print(f"Warning: Could not parse {rou_file} for start locations.")

    # --- Text Report ---
    print("\n" + "="*55); print("      Detailed Mission Analysis (Actual vs. Ideal)"); print("="*55)
    for mission in missions:
        amb_id = mission['ambulance_id']; patient_id = mission['patient_id']; hospital_id = mission['hospital_id']
        actual_data = actual_results_dict.get(amb_id)
        patient_info = PATIENTS.get(patient_id); hospital_info = HOSPITALS.get(hospital_id)
        amb_start_edge = ambulance_starts.get(amb_id)

        print(f"\n--- Mission {mission['mission_num']} (Patient {patient_id} by {amb_id}, Dispatched T={mission['dispatch_time']:.1f}s) ---")
        if not patient_info or not hospital_info: print("  ERROR: Missing patient/hospital info."); continue

        patient_edge = patient_info['start_edge']; hospital_edge = hospital_info['dest_edge']
        print(f"  Assigned Hospital: {hospital_info.get('name', 'N/A')} ({hospital_id}), Beds: {mission['beds_info']}")

        if actual_data:
            print(f"  Actual Performance:")
            print(f"    - Duration: {actual_data['duration']:.2f}s | Distance: {actual_data['routeLength']:.2f}m | Wait: {actual_data['waitingTime']:.2f}s | Loss: {actual_data['timeLoss']:.2f}s")
        else: print("  ERROR: No tripinfo found for this ambulance."); continue

        # Estimate Ideal Path (Start -> Patient -> Hospital)
        est_time_full, est_dist_full = float('inf'), float('inf')
        if amb_start_edge:
            time_sp, dist_sp = router.get_path_details(amb_start_edge, patient_edge)
            time_ph, dist_ph = router.get_path_details(patient_edge, hospital_edge)
            if time_sp != float('inf') and time_ph != float('inf'):
                est_time_full = time_sp + time_ph; est_dist_full = dist_sp + dist_ph
            else: print("  Warning: Could not calc full ideal path.")
        else: print("  Warning: Missing start edge for ideal estimation.")

        print(f"  Comparison:")
        print(f"    - Ideal Time : {est_time_full:.2f}s vs Actual Time : {actual_data['duration']:.2f}s (Diff: {actual_data['duration'] - est_time_full:+.2f}s)")
        print(f"    - Ideal Dist : {est_dist_full:.2f}m vs Actual Dist : {actual_data['routeLength']:.2f}m (Diff: {actual_data['routeLength'] - est_dist_full:+.2f}m)")

    print("\n" + "="*55)

    # --- Visual Plots ---
    print("\n--- Generating Visual Plots ---")
    df_results = pd.DataFrame(list(actual_results_dict.values()))
    if df_results.empty: print("Aborting plots: No data loaded."); return

    plot_pie_charts(df_results)
    plot_car_histogram(df_results)

    # Before/After Comparison Plot
    if os.path.exists(before_file):
        print(f"--- Generating Before ({before_file}) vs. After ({results_file}) Comparison ---")
        before_stats_dict = parse_tripinfo(before_file, AMBULANCES)
        if before_stats_dict:
            after_stats_dict = {k: v for k, v in actual_results_dict.items() if k in AMBULANCES}
            common_ambulances = sorted(list(set(before_stats_dict.keys()) & set(after_stats_dict.keys())))
            if common_ambulances: plot_comparison(before_stats_dict, after_stats_dict, common_ambulances)
            else: print("No common ambulances found for comparison.")
        else: print(f"Could not parse {before_file}.")
    else: print(f"Info: {before_file} not found. Skipping comparison.")

if __name__ == "__main__":
    main()
    print("\nReport generation finished.")