import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_tripinfo(filename, vehicle_ids):
    """Parses a tripinfo XML file and extracts data for specific vehicles."""
    data = {}
    tree = ET.parse(filename)
    root = tree.getroot()
    for trip in root.findall('tripinfo'):
        veh_id = trip.get('id')
        if veh_id in vehicle_ids:
            data[veh_id] = {
                'duration': float(trip.get('duration')),
                'routeLength': float(trip.get('routeLength')),
                'timeLoss': float(trip.get('timeLoss')),
                'waitingTime': float(trip.get('waitingTime'))
            }
    return data

def plot_comparison(before_data, after_data, ambulances):
    """Generates bar charts to compare simulation metrics."""
    metrics_to_plot = {
        'duration': 'Trip Duration (seconds)',
        'timeLoss': 'Time Loss (seconds)',
        'waitingTime': 'Waiting Time (seconds)',
        'routeLength': 'Route Length (meters)'
    }
    
    for metric, title in metrics_to_plot.items():
        before_values = [before_data[amb][metric] for amb in ambulances]
        after_values = [after_data[amb][metric] for amb in ambulances]

        x = np.arange(len(ambulances))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, before_values, width, label='Original Route', color='lightcoral')
        rects2 = ax.bar(x + width/2, after_values, width, label='Dijkstra Route', color='skyblue')

        # Add some text for labels, title and axes ticks
        ax.set_ylabel(title.split(' ')[-1].capitalize())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(ambulances)
        ax.legend()
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    ambulances_to_compare = ['amb_1', 'amb_2']
    
    # Parse data from both files
    print("Parsing tripinfo_before.xml...")
    before_stats = parse_tripinfo('tripinfo_before.xml', ambulances_to_compare)
    
    print("Parsing tripinfo_after.xml...")
    after_stats = parse_tripinfo('tripinfo_after.xml', ambulances_to_compare)

    # Check if we found data for all ambulances
    if len(before_stats) < len(ambulances_to_compare) or len(after_stats) < len(ambulances_to_compare):
        print("\nWarning: Could not find data for all specified ambulances in both files.")
        print("Please ensure the simulations ran long enough for them to finish.")
    else:
        print("\nGenerating comparison plots...")
        plot_comparison(before_stats, after_stats, ambulances_to_compare)