import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import RegularPolygon
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Load the data
file_path = '/Users/marcoantoniosahagun/Downloads/DJIFlightRecord_2024-07-08_[13-49-42].csv'
data = pd.read_csv(file_path)

# Extract relevant data
time = data['OSD.flyTime [s]'].values
x_velocity = data['OSD.xSpeed [MPH]'].values
y_velocity = data['OSD.ySpeed [MPH]'].values
yaw = data['OSD.yaw'].values

# Create continuous functions for velocity
vx_func = interp1d(time, x_velocity, kind='linear', fill_value='extrapolate')
vy_func = interp1d(time, y_velocity, kind='linear', fill_value='extrapolate')

# Define the ODE system
def odes(t, y):
    x, y_pos = y
    return [vx_func(t), vy_func(t)]

# Initial conditions
initial_conditions = [0.0, 0.0]

# Time span
t_span = [time[0], time[-1]]
t_eval = np.linspace(time[0], time[-1], len(time))

# Solving the ODE
solution = solve_ivp(odes, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Extracting the results
x_path = solution.y[0]
y_path = solution.y[1]

# Create the figure and subplots
fig, (ax_path, ax_velocity) = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(bottom=0.2, right=0.8)  # Make room for the slider and legend

# Plot for the drone's path
ax_path.plot(y_path, x_path, label='Trajectory')
ax_path.set_xlabel('y position (meters)')
ax_path.set_ylabel('x position (meters)')
ax_path.set_title('Drone Path')
ax_path.axhline(0, color='black', linewidth=0.5)
ax_path.axvline(0, color='black', linewidth=0.5)
ax_path.grid(True)
ax_path.legend()
path_dot, = ax_path.plot([0], [0], 'ro')  # Dot representing current position

# Plot for the drone's velocity and orientation
max_velocity = max(x_velocity.max(), y_velocity.max())
ax_velocity.set_xlim(-max_velocity, max_velocity)
ax_velocity.set_ylim(-max_velocity, max_velocity)
ax_velocity.axhline(0, color='black', linewidth=0.5)
ax_velocity.axvline(0, color='black', linewidth=0.5)
ax_velocity.set_xlabel('Y Speed (MPH)')
ax_velocity.set_ylabel('X Speed (MPH)')
ax_velocity.set_xticks(np.arange(-max_velocity, max_velocity + 1, step=max_velocity / 2))
ax_velocity.set_yticks(np.arange(-max_velocity, max_velocity + 1, step=max_velocity / 2))
ax_velocity.grid(True)
ax_velocity.set_title('Drone Velocity and Orientation')

velocity_arrow = ax_velocity.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label='Horizontal Velocity')
triangle_size = 2.5  
yaw_triangle = RegularPolygon((0, 0), 3, radius=triangle_size * 0.5, orientation=np.pi/6, facecolor='blue', edgecolor='blue', label='Yaw Direction')
ax_velocity.add_patch(yaw_triangle)
yaw_text_at_point = ax_velocity.text(0, 0, '', color='white', fontweight='bold', ha='center', va='center')

# Add text for velocity and yaw
velocity_text = ax_velocity.text(0.05, 0.95, '', transform=ax_velocity.transAxes, color='red', fontsize=10)
yaw_text = ax_velocity.text(0.05, 0.90, '', transform=ax_velocity.transAxes, color='blue', fontsize=10)

# Update function for slider
def update(val):
    frame = int(val)
    
    # Update path dot
    path_dot.set_data(y_path[frame], x_path[frame])
    
    # Update velocity arrow
    velocity_arrow.set_UVC(y_velocity[frame], x_velocity[frame])
    velocity_magnitude = np.sqrt(x_velocity[frame] ** 2 + y_velocity[frame] ** 2)
    velocity_text.set_text(f'Velocity: {velocity_magnitude:.2f} MPH')
    
    # Update yaw triangle
    yaw_rad = np.radians(yaw[frame])
    yaw_triangle.orientation = -yaw_rad  # Negative because matplotlib uses opposite rotation direction
    yaw_text.set_text(f'Yaw: {yaw[frame]:.0f}°')

    # Update yaw text at triangle point
    text_distance = triangle_size * 0.7  # Adjust this value to position the text
    text_x = text_distance * np.sin(yaw_rad)
    text_y = text_distance * np.cos(yaw_rad)
    yaw_text_at_point.set_position((text_x, text_y))
    yaw_text_at_point.set_text(f'{yaw[frame]:.0f}°')
    yaw_text_at_point.set_color('blue')

    # Redraw the canvas
    fig.canvas.draw_idle()

# Create the slider
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(time) - 1, valinit=0, valstep=1)

# Connect the slider to the update function
slider.on_changed(update)

# Initial plot update
update(0)

plt.show()
