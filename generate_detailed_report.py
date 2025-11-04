import xml.etree.ElementTree as ET
import re
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import math 

# --- Constants ---
PICKUP_TIME = 2.0 
CONGESTION_FACTOR = 1.15 
FONT_SIZE_AXIS = 14
FONT_SIZE_TITLE = 16

# --- Ambulance ID Renaming Map (Specific IDs) ---
# Maps the simulation IDs to the custom output IDs
AMB_ID_MAP = {
    "amb_1": "amb_501", 
    "amb_3": "amb_503", 
    "amb_4": "amb_504", 
    "amb_5": "amb_505"
}

# --- Simulation Data (Fixed Hospital/Ambulance Info) ---
HOSPITALS = {
    "H-01": {"name": "City_General", "dest_edge": "e6"},
    "H-02": {"name": "Green_Heart", "dest_edge": "e2"},
    "H-04": {"name": "Tumakuru_Trauma", "dest_edge": "e4"}
}
# --- MODIFIED: Updated list of ambulance IDs (4 Ambulances) ---
AMBULANCES = ["amb_1", "amb_3", "amb_4", "amb_5"]


# --- Dijkstra Pathfinding Class (Static version for IDEAL calculations) ---
class DijkstraForSUMO:
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
                        ideal_travel_time = length / speed if speed > 0 else float('inf')
                        if from_node not in self.graph: self.graph[from_node] = {}
                        self.graph[from_node][to_node] = ideal_travel_time
                        self.edge_to_junctions[edge_id] = (from_node, to_node)
                        self.junction_pair_to_edge[(from_node, to_node)] = edge_id
                        self.edge_details[edge_id] = {'length': length, 'time': ideal_travel_time}
            for node in junctions_with_edges:
                    if node not in self.graph: self.graph[node] = {}
        except FileNotFoundError: raise ValueError(f"Net file '{self.net_file}' not found.")
        except ET.ParseError as e: raise ValueError(f"Failed to parse net file '{self.net_file}': {e}")

    def find_shortest_static_path(self, start_edge, end_edge):
        if not start_edge or not end_edge: return None, float('inf'), float('inf')
        if start_edge == end_edge:
             details = self.edge_details.get(start_edge)
             if details: return ([start_edge], details.get('length', float('inf')), details.get('time', float('inf')))
             else: return (None, float('inf'), float('inf'))
        start_node_tuple = self.edge_to_junctions.get(start_edge); end_node_tuple = self.edge_to_junctions.get(end_edge)
        if not start_node_tuple or not end_node_tuple: return None, float('inf'), float('inf')
        start_junction, end_junction = start_node_tuple[1], end_node_tuple[0]
        if start_junction not in self.graph: return None, float('inf'), float('inf')
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
        if not path_found or distances.get(end_junction, float('inf')) == float('inf'): return None, float('inf'), float('inf')
        path_nodes = []; current = end_junction
        while current is not None: path_nodes.insert(0, current); current = previous_nodes.get(current)
        if not path_nodes or path_nodes[0] != start_junction: return None, float('inf'), float('inf')
        junction_path_edges = [self.junction_pair_to_edge.get((path_nodes[i], path_nodes[i+1])) for i in range(len(path_nodes) - 1)]
        valid_junction_edges = [edge for edge in junction_path_edges if edge]
        full_edge_path = [start_edge] + valid_junction_edges
        if start_edge != end_edge and end_edge not in valid_junction_edges:
             full_edge_path.append(end_edge)
        total_ideal_dist = 0; total_ideal_time = 0; path_valid = True
        for edge in full_edge_path:
             details = self.edge_details.get(edge)
             if not details or details.get('time', float('inf')) == float('inf'): path_valid = False; break
             total_ideal_dist += details.get('length', 0); total_ideal_time += details.get('time', 0)
        if not path_valid: return None, float('inf'), float('inf')
        return full_edge_path, total_ideal_dist, total_ideal_time

# --- Parsing Functions (Unchanged) ---
def parse_dispatch_summary(filename="dispatch_summary.txt"):
    missions = [];
    try:
        with open(filename, "r") as f: content = f.read()
        pattern = re.compile(
            r"--- Mission (\d+) \(Dispatched at T=([\d\.]+?)s\) ---" 
            r".*?Ambulance:\s*(\w+)"
            r".*?Patient Details:.*?- ID:\s+(\w+)"
            r".*?Location:\s+Edge\s+'([^']+)'"
            r".*?Assigned Hospital:.*?- ID:\s+([\w-]+)"
            r".*?Beds.*?:\s*(\d+)\s*->\s*(\d+)",
            re.DOTALL
        )
        matches = pattern.finditer(content)
        for match in matches:
            mission_data = {
                'mission_num': int(match.group(1)), 'dispatch_time': float(match.group(2)),
                'ambulance_id': match.group(3), 'patient_id': match.group(4),
                'patient_start_edge': match.group(5), 'hospital_id': match.group(6),
                'beds_before': int(match.group(7)), 'beds_after': int(match.group(8)),
                'beds_info': f"{match.group(7)} -> {match.group(8)}"
            }
            missions.append(mission_data)
    except FileNotFoundError: print(f"Error: File '{filename}' not found."); return None
    except Exception as e: print(f"Error parsing dispatch summary '{filename}': {e}"); return None
    if not missions: print(f"Warning: No missions found in {filename}.")
    return missions

def parse_tripinfo(filename, vehicle_ids_of_interest=None):
    results = {};
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


# --- Plotting Functions (MODIFIED) ---
def plot_actual_vs_estimated_bars(mission_data):
    valid_missions = [m for m in mission_data if m['actual_duration'] is not None and m['estimated_time'] != float('inf')]
    if not valid_missions: print("No valid distance data to plot."); return

    # Map IDs to the 500 series for display
    labels = [f"{AMB_ID_MAP.get(m['ambulance_id'], m['ambulance_id'])}\n(P{m['patient_id'][-2:]})" for m in valid_missions]
    estimated_times = [m['estimated_time'] for m in valid_missions]
    actual_times = [m['actual_duration'] for m in valid_missions]
    ideal_dists = [m['ideal_dist'] for m in valid_missions]
    actual_dists = [m['actual_distance'] for m in valid_missions]
    x = np.arange(len(labels)); width = 0.35

    # --- Plot 1: Time Comparison ---
    fig1, ax1 = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    rects1 = ax1.bar(x - width/2, estimated_times, width, label='Estimated Time', color='lightblue')
    rects2 = ax1.bar(x + width/2, actual_times, width, label='Time Taken by Ga', color='coral')
    
    ax1.set_ylabel('Time (seconds)', fontsize=FONT_SIZE_AXIS); 
    ax1.set_title('Ambulance Response Time: Actual vs. GA Estimate', fontsize=FONT_SIZE_TITLE)
    ax1.set_xticks(x); 
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_AXIS - 4); 
    ax1.legend(fontsize=FONT_SIZE_AXIS - 4)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(rects1, padding=3, fmt='%.0f', fontsize=FONT_SIZE_AXIS - 6); 
    ax1.bar_label(rects2, padding=3, fmt='%.0f', fontsize=FONT_SIZE_AXIS - 6)
    fig1.tight_layout(); plt.savefig("time_comparison.png"); plt.close(fig1)

    # --- Plot 2: Distance Comparison (Actual vs Ideal) ---
    fig2, ax2 = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    rects3 = ax2.bar(x - width/2, ideal_dists, width, label='Ideal Distance', color='lightgreen')
    rects4 = ax2.bar(x + width/2, actual_dists, width, label='Actual Distance', color='mediumpurple')
    
    ax2.set_ylabel('Distance (meters)', fontsize=FONT_SIZE_AXIS); 
    ax2.set_title('Ambulance Travel Distance: Actual vs. Ideal', fontsize=FONT_SIZE_TITLE)
    ax2.set_xticks(x); 
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_AXIS - 4); 
    ax2.legend(fontsize=FONT_SIZE_AXIS - 4)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(rects3, padding=3, fmt='%.0f', fontsize=FONT_SIZE_AXIS - 6); 
    ax2.bar_label(rects4, padding=3, fmt='%.0f', fontsize=FONT_SIZE_AXIS - 6)
    fig2.tight_layout(); plt.savefig("distance_comparison.png"); plt.close(fig2)


def plot_pie_charts(df_results):
    amb_data = df_results[df_results['id'].str.contains('amb_', na=False)].copy() # Use .copy() to avoid SettingWithCopyWarning
    if amb_data.empty: print("No ambulance data in DataFrame to plot pie charts."); return
    
    # Rename IDs for plotting
    amb_data['id'] = amb_data['id'].apply(lambda x: AMB_ID_MAP.get(x, x))
    
    num_amb = len(amb_data); ncols = min(num_amb, 5); nrows = math.ceil(num_amb / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Ambulance Trip Time Breakdown (Actual Times)', fontsize=FONT_SIZE_TITLE); 
    axes_flat = axes.flatten()
    
    for i, (idx, amb) in enumerate(amb_data.iterrows()):
        traffic_waiting_time = max(0, amb['waitingTime'] - PICKUP_TIME)
        driving_time = max(0, amb['duration'] - amb['waitingTime'])
        pickup_time_val = PICKUP_TIME if amb['waitingTime'] >= PICKUP_TIME else amb['waitingTime']
        
        sizes_full = [driving_time, traffic_waiting_time, pickup_time_val]
        labels_full = ['Driving Time', 'Traffic Waiting', 'Pickup Time']
        colors_full = ['skyblue', 'lightcoral', 'lightgrey']

        # Filter out negligible sizes and create label strings with actual time values
        sizes = []; labels = []; colors = []
        for size, label, color in zip(sizes_full, labels_full, colors_full):
            if size > 0.01:
                sizes.append(size)
                labels.append(f"{label} ({size:.1f}s)") # Shows actual time instead of percentage
                colors.append(color)

        ax = axes_flat[i]
        if sizes:
             ax.pie(sizes, labels=labels, startangle=90, colors=colors, 
                    textprops={'fontsize': FONT_SIZE_AXIS - 6})
             ax.set_title(f"{amb['id']}\nTotal: {amb['duration']:.1f}s", fontsize=FONT_SIZE_AXIS - 2); 
             ax.axis('equal')
        else: ax.set_title(f"{amb['id']}\nNo Data", fontsize=FONT_SIZE_AXIS - 2); ax.axis('off')
        
    for j in range(i + 1, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig("pie_charts.png"); plt.close(fig)


def plot_ambulance_histogram(df_results):
    amb_durations = df_results[df_results['id'].str.contains('amb_', na=False)]['duration'].dropna()
    if amb_durations.empty: print("No ambulance data in DataFrame for histogram."); return
    
    plt.figure(figsize=(10, 6)); 
    plt.hist(amb_durations, bins=8, color='lightgreen', edgecolor='black')
    
    plt.xlabel('Ambulance Duration (seconds)', fontsize=FONT_SIZE_AXIS); 
    plt.ylabel('Count of Ambulances', fontsize=FONT_SIZE_AXIS)
    plt.title('Duration Distribution of Ambulances in Single Simulation', fontsize=FONT_SIZE_TITLE); 
    
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout(); plt.savefig("duration_histogram.png"); plt.close()

# --- Main Report Generation Logic (With Hardcoded Start Edge Fix) ---
def main():
    net_file = 'hexagon.net.xml'; summary_file = 'dispatch_summary.txt'
    results_file = 'tripinfo_results.xml'; rou_file = 'hexagon.rou.xml'

    print(f"--- Generating Detailed Dispatch Report from {summary_file} & {results_file} ---")
    
    # --- HARDCODED AMBULANCE START EDGES (REQUIRED FOR CORRECT CALCULATION) ---
    # Used by the Dijkstra router to calculate estimated times.
    ambulance_starts = {
        "amb_1": "e5", 
        "amb_3": "e3", 
        "amb_4": "e7_rev", 
        "amb_5": "e12" # Updated to 'e12' to match the change in rou.xml and traci_runner.py
    }
    # --------------------------------------------------------------------------

    # --- Parse Input Files ---
    missions = parse_dispatch_summary(summary_file)
    actual_results_dict = parse_tripinfo(results_file)
    if not missions: print("Aborting: Could not parse dispatch summary."); return
    if not actual_results_dict: print("Aborting: Could not parse tripinfo results."); return

    # Initialize Dijkstra with IDEAL times
    try: router = DijkstraForSUMO(net_file)
    except Exception as e: print(f"Aborting: Failed to init Dijkstra: {e}"); return
    
    # --- Generate Text Report & Collect Data for Plots ---
    print("\n" + "="*60); print("      Detailed Response Analysis (Actual vs. Estimated)"); print("="*60)
    mission_plot_data = [] 

    for mission in missions:
        amb_id = mission['ambulance_id']; patient_id = mission['patient_id']
        hospital_id = mission['hospital_id']; patient_edge = mission['patient_start_edge']
        actual_data = actual_results_dict.get(amb_id)
        hospital_info = HOSPITALS.get(hospital_id); amb_start_edge = ambulance_starts.get(amb_id)

        # Apply ID mapping for display in the text report
        display_amb_id = AMB_ID_MAP.get(amb_id, amb_id)

        print(f"\n--- Run {mission['mission_num']} (Patient {patient_id} by {display_amb_id}, Dispatched T={mission['dispatch_time']:.1f}s) ---")
        if not hospital_info: print(f"  ERROR: Hospital ID {hospital_id} not found."); continue
        if not patient_edge: print(f"  ERROR: Patient start edge missing."); continue

        hospital_edge = hospital_info['dest_edge']
        print(f"  Patient Location: Edge '{patient_edge}'")
        print(f"  Assigned Hospital: {hospital_info.get('name', 'N/A')} ({hospital_id}), Beds: {mission['beds_info']}")

        # --- CORRECTED Calculation Logic (Calculates GA Estimated Time) ---
        final_ideal_distance = float('inf')
        final_estimated_time = float('inf')
        if amb_start_edge:
            path_sp, ideal_dist_sp, ideal_time_sp = router.find_shortest_static_path(amb_start_edge, patient_edge)
            path_ph, ideal_dist_ph, ideal_time_ph = router.find_shortest_static_path(patient_edge, hospital_edge)
            if ideal_time_sp != float('inf') and ideal_time_ph != float('inf'):
                 patient_edge_details = router.edge_details.get(patient_edge)
                 patient_edge_dist = patient_edge_details.get('length', 0) if patient_edge_details else 0
                 patient_edge_time = patient_edge_details.get('time', 0) if patient_edge_details else 0
                 subtract_dist = patient_edge_dist if amb_start_edge != patient_edge and patient_edge != hospital_edge else 0
                 subtract_time = patient_edge_time if amb_start_edge != patient_edge and patient_edge != hospital_edge else 0
                 final_ideal_distance = max(0, ideal_dist_sp + ideal_dist_ph - subtract_dist)
                 total_ideal_travel_time = max(0, ideal_time_sp + ideal_time_ph - subtract_time)
                 final_estimated_time = (total_ideal_travel_time * CONGESTION_FACTOR) + PICKUP_TIME
            else:
                 print("  Warning: Could not calculate full ideal path (Start->Patient->Hospital).")
                 if ideal_time_sp != float('inf'):
                      final_ideal_distance = ideal_dist_sp; final_estimated_time = (ideal_time_sp * CONGESTION_FACTOR)
                 elif ideal_time_ph != float('inf'):
                      final_ideal_distance = ideal_dist_ph; final_estimated_time = (ideal_time_ph * CONGESTION_FACTOR)
        else:
             print(f"  Warning: Could not find start edge for {amb_id}.")

        mission_plot_data.append({
            "ambulance_id": amb_id, "patient_id": patient_id,
            "estimated_time": final_estimated_time, "ideal_dist": final_ideal_distance,
            "actual_duration": actual_data['duration'] if actual_data else None,
            "actual_distance": actual_data['routeLength'] if actual_data else None
        })

        if actual_data:
            # FIX applied here: Ensuring 'actual_data' is spelled correctly everywhere.
            print(f"  Actual Performance:")
            print(f"    - Duration: {actual_data['duration']:.2f}s | Distance: {actual_data['routeLength']:.2f}m | Wait: {actual_data['waitingTime']:.2f}s | Loss: {actual_data['timeLoss']:.2f}s")
            print(f"  Comparison:")
            print(f"    - GA Est. Time: {final_estimated_time:.2f}s vs Actual Time : {actual_data['duration']:.2f}s (Diff: {actual_data['duration'] - final_estimated_time:+.2f}s)")
            print(f"    - Ideal Distance: {final_ideal_distance:.2f}m vs Actual Distance: {actual_data['routeLength']:.2f}m (Diff: {actual_data['routeLength'] - final_ideal_distance:+.2f}m)")
        else:
            print("  ERROR: No tripinfo found for this ambulance in results file.")

    print("\n" + "="*60)

    # --- Generate Visual Plots ---
    print("\n--- Generating Visual Plots ---")
    df_results = pd.DataFrame(list(actual_results_dict.values()))
    if df_results.empty: print("Aborting plots: No data loaded into DataFrame."); return

    plot_actual_vs_estimated_bars(mission_plot_data)
    plot_pie_charts(df_results)
    plot_ambulance_histogram(df_results)

    print("\nPlots saved as time_comparison.png, distance_comparison.png, pie_charts.png, duration_histogram.png")

if __name__ == "__main__":
    main()