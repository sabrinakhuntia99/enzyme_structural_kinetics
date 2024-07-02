import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

def plot_hulls_for_time_points(conv_peptide_coord, hull_layers):
    # Use list comprehension to directly generate time points and hull numbers
    time_points = [data_point['TimePoint'] for data_point in conv_peptide_coord]
    hull_numbers = [next((j + 1 for j, hull_points in enumerate(hull_layers) if any(
        np.dot(eq[:-1], (data_point['X'], data_point['Y'], data_point['Z'])) + eq[-1] <= 0 for eq in ConvexHull(hull_points).equations
    )), None) for data_point in conv_peptide_coord]

    # Plotting the scatter graph
    plt.figure(figsize=(8, 6))
    plt.scatter(time_points, hull_numbers, marker='o', color='blue')
    plt.xlabel('Time Points')
    plt.ylabel('Detected Hull Number')
    plt.title('Detected Hull Numbers for Peptides over Time')

    # Calculate the maximum number of hull layers
    max_hull_number = len(hull_layers)

    # Set the y-axis limit in increments of 1 up to the maximum hull layer number
    plt.ylim(0, max_hull_number + 1)

    # Set y-axis ticks in increments of 1
    plt.yticks(range(1, max_hull_number + 2))

    plt.grid(True)
    plt.show()

def plot_xyz_coordinates_with_time_and_hulls(data, hull_layers):
    # Create an empty list to store traces
    traces = []

    # Create an empty dictionary to store detected points based on time point
    detected_points = {}

    # Iterate over the data list to group detected points based on time point
    for time_point_data in data:
        # Extract time point name and coordinates
        time_point = time_point_data['TimePoint']
        X = time_point_data['X']
        Y = time_point_data['Y']
        Z = time_point_data['Z']

        # Add coordinates to the detected points dictionary
        if time_point in detected_points:
            detected_points[time_point].append((X, Y, Z))
        else:
            detected_points[time_point] = [(X, Y, Z)]

    # Create traces for detected points and segments for each time point
    for time_point, points in detected_points.items():
        # Sort points based on X coordinate (assuming X represents the detected direction)
        points.sort(key=lambda p: p[0])

        # Extract X, Y, Z coordinates from sorted points
        X, Y, Z = zip(*points)

        # Create scatter trace for detected points
        trace_points = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='markers',
            marker=dict(size=8, color='white', symbol='square'),  # Adjust marker size and color as needed
            name=f'Points at Time Point: {time_point}'  # Include time point name in trace name
        )
        traces.append(trace_points)

        # Create line trace for detected segments
        trace_detected = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(color='white', width=4),  # Adjust line color and width as needed
            name=f'Peptide at Time Point: {time_point}'  # Include time point name in trace name
        )
        traces.append(trace_detected)

    # Define a list of colors for the hulls
    hull_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']

    # Create a trace for each hull layer with a unique color
    for i, hull_points in enumerate(hull_layers):
        # Use modulo operation to repeat colors if there are more hulls than colors
        color_index = i % len(hull_colors)
        color = hull_colors[color_index]

        trace_hull = go.Mesh3d(
            x=hull_points[:, 0],
            y=hull_points[:, 1],
            z=hull_points[:, 2],
            opacity=0.3,  # Adjust opacity as needed
            color=color,  # Assign a unique color to each hull
            name=f'Hull {i + 1}'
        )
        traces.append(trace_hull)

    # Create plot layout

    layout = go.Layout(
        title='Convex Layered Protein Structure with Detected Peptides (Cartesian Coordinates)',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="black", gridcolor="gray", showgrid=False),
            yaxis=dict(title='Y', backgroundcolor="black", gridcolor="gray", showgrid=False),
            zaxis=dict(title='Z', backgroundcolor="black", gridcolor="gray", showgrid=False),
            bgcolor='black',
            aspectmode="cube",
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot
    fig.show()

def plot_triangles_for_time_points(conv_peptide_coord, triangle_layers):
    # Use list comprehension to directly generate time points and triangle numbers
    time_points = [data_point['TimePoint'] for data_point in conv_peptide_coord]
    triangle_numbers = [next((j + 1 for j, triangle_points in enumerate(triangle_layers) if any(
        np.dot(eq[:-1], (data_point['X'], data_point['Y'], data_point['Z'])) + eq[-1] <= 0 for eq in Delaunay(triangle_points).equations
    )), None) for data_point in conv_peptide_coord]

    # Plotting the scatter graph
    plt.figure(figsize=(8, 6))
    plt.scatter(time_points, triangle_numbers, marker='o', color='blue')
    plt.xlabel('Time Points')
    plt.ylabel('Detected Triangle Number')
    plt.title('Detected Triangle Numbers for Peptides over Time')

    # Calculate the maximum number of triangle layers
    max_triangle_number = len(triangle_layers)

    # Set the y-axis limit in increments of 1 up to the maximum triangle layer number
    plt.ylim(0, max_triangle_number + 1)

    # Set y-axis ticks in increments of 1
    plt.yticks(range(1, max_triangle_number + 2))

    plt.grid(True)
    plt.show()

def plot_xyz_coordinates_with_time_and_triangles(data, triangle_layers):
    # Create an empty list to store traces
    traces = []

    # Create an empty dictionary to store detected points based on time point
    detected_points = {}

    # Iterate over the data list to group detected points based on time point
    for time_point_data in data:
        # Extract time point name and coordinates
        time_point = time_point_data['TimePoint']
        X = time_point_data['X']
        Y = time_point_data['Y']
        Z = time_point_data['Z']

        # Add coordinates to the detected points dictionary
        if time_point in detected_points:
            detected_points[time_point].append((X, Y, Z))
        else:
            detected_points[time_point] = [(X, Y, Z)]

    # Create traces for detected points and segments for each time point
    for time_point, points in detected_points.items():
        # Sort points based on X coordinate (assuming X represents the detected direction)
        points.sort(key=lambda p: p[0])

        # Extract X, Y, Z coordinates from sorted points
        X, Y, Z = zip(*points)

        # Create scatter trace for detected points
        trace_points = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='markers',
            marker=dict(size=8, color='white', symbol='square'),  # Adjust marker size and color as needed
            name=f'Points at Time Point: {time_point}'  # Include time point name in trace name
        )
        traces.append(trace_points)

        # Create line trace for detected segments
        trace_detected = go.Scatter3d(
            x=X,
            y=Y,
            z=Z,
            mode='lines',
            line=dict(color='white', width=4),  # Adjust line color and width as needed
            name=f'Peptide at Time Point: {time_point}'  # Include time point name in trace name
        )
        traces.append(trace_detected)

    # Define a list of colors for the triangles
    triangle_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']

    # Create a trace for each triangle layer with a unique color
    for i, triangle_points in enumerate(triangle_layers):
        # Use modulo operation to repeat colors if there are more triangles than colors
        color_index = i % len(triangle_colors)
        color = triangle_colors[color_index]

        trace_triangle = go.Mesh3d(
            x=triangle_points[:, 0],
            y=triangle_points[:, 1],
            z=triangle_points[:, 2],
            opacity=0.3,  # Adjust opacity as needed
            color=color,  # Assign a unique color to each triangle
            name=f'Triangle {i + 1}'
        )
        traces.append(trace_triangle)

    # Create plot layout

    layout = go.Layout(
        title='Delaunay Tessellated Protein Structure with Detected Peptides (Cartesian Coordinates)',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="black", gridcolor="gray", showgrid=False),
            yaxis=dict(title='Y', backgroundcolor="black", gridcolor="gray", showgrid=False),
            zaxis=dict(title='Z', backgroundcolor="black", gridcolor="gray", showgrid=False),
            bgcolor='black',
            aspectmode="cube",
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot
    fig.show()

