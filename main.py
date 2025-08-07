import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Base Traffic Parameters
rt = 0.1   # Mean reaction time
a = 5      # Mean acceleration
b = -5     # Mean deceleration
fd = 10    # Desired following distance
tol = 0.1  # Tolerance
sl = 0.7 * np.sqrt(-1 * fd * b)  # Speed limit (from physics)

# Std dev for variability
rt_std = 0.01
a_std = 0.1
b_std = 0.1

# Simulation parameters
w = 100                # Road width
N_cars = int(w / fd)   # Number of cars
dt = 0.1               # Time step
fps = 15               # Frames per second
duration = 10           # seconds
n_frames = fps * duration

# Initial state
X = np.sort(w * np.random.rand(N_cars))  # Car positions (sorted)
V = 2 * np.ones(N_cars)                  # Initial velocities

# Individual parameters per car
RT = np.clip(np.random.normal(rt, rt_std, N_cars), 0.01, None)
A = np.clip(np.random.normal(a, a_std, N_cars), 0, None)
B = np.clip(np.random.normal(b, b_std, N_cars), None, 0)

# Stoplight parameters
stoplight_x = 50
green_time = 2
red_time = 2
cycle_time = green_time + red_time
approach_zone = 10.0

def stoplight_state(t):
    """Return True if green, False if red."""
    return (t % cycle_time) < green_time

def decision(dx, v, a_i, b_i):
    if (dx - fd) > tol:
        return a_i
    elif (dx - fd) < -tol:
        return b_i
    else:
        return 0

def moving_average_density_velocity(x_vals, X, V, window):
    densities = []
    avg_velocities = []
    for x in x_vals:
        dists = np.abs((X - x + w/2) % w - w/2)
        mask = dists <= window / 2
        num = np.sum(mask)
        densities.append(num / window)
        avg_velocities.append(np.mean(V[mask]) if num > 0 else 0)
    return np.array(densities), np.array(avg_velocities)

# Set up plot
x_vals = np.linspace(0, w, 500)
car_y_level = -0.05
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
fig.suptitle(f"Traffic Simulation | Speed Limit: {sl:.1f}", fontsize=14, fontweight='bold')

# Top subplot (Density)
scat = axs[0].scatter(X, np.full_like(X, car_y_level), s=25, color='green', label='Cars')
stoplight_line = axs[0].axvline(stoplight_x, color='green', linestyle='--', linewidth=2, alpha=0.6)
density_line, = axs[0].plot([], [], 'red', label='Density')
jam_fill = axs[0].fill_between(x_vals, 0, 0)
axs[0].set_ylim(car_y_level - 0.05, 2)
axs[0].set_xlim(0, w)
axs[0].set_ylabel("Density")         # <-- Label added
axs[0].legend(loc="upper right")

# Bottom subplot (Velocity)
velocity_line, = axs[1].plot([], [], 'blue', label='Velocity')
axs[1].set_ylim(0, sl + 1)
axs[1].set_ylabel("Velocity")        # <-- Label added
axs[1].set_xlabel("Position")        # <-- Common x-axis label
axs[1].legend(loc="upper right")

t = 0

def update(frame):
    global X, V, A, B, RT, t, jam_fill

    green_light = stoplight_state(t)

    for i in range(N_cars):
        x = X[i]
        v = V[i]
        dx = (X[(i + 1) % N_cars] + (w if i == N_cars - 1 else 0)) - X[i]

        a_i = A[i]
        b_i = B[i]
        rt_i = RT[i]

        acc = decision(dx, v, a_i, b_i)

        dx_light = (stoplight_x - x + w) % w
        if not green_light and dx_light < approach_zone:
            if dx_light < 0.5 and v < 0.5:
                acc = -v / dt
            else:
                acc = b_i

        dv = acc * dt
        v_old = v
        v += dv
        v = np.clip(v, 0, sl)
        x += v * dt + v_old * rt_i
        x %= w

        X[i] = x
        V[i] = v

    sort_idx = np.argsort(X)
    X, V, A, B, RT = X[sort_idx], V[sort_idx], A[sort_idx], B[sort_idx], RT[sort_idx]

    density, avg_velocity = moving_average_density_velocity(x_vals, X, V, 2 * fd)

    scat.set_offsets(np.column_stack((X, np.full_like(X, car_y_level))))
    density_line.set_data(x_vals, density)
    velocity_line.set_data(x_vals, avg_velocity)
    stoplight_line.set_color('green' if green_light else 'red')

    for coll in axs[0].collections[1:]:
        coll.remove()

    jam_mask = (density > 0.5 / fd) & (avg_velocity < 0.3 * sl)
    axs[0].fill_between(
        x_vals, 0, density, where=jam_mask,
        color='red', alpha=0.1, step='mid'
    )

    axs[0].set_ylim(car_y_level - 0.05, max(2, np.max(density) * 1.1))
    t += dt

ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps)
gif_path = "traffic_simulation.gif"
ani.save(gif_path, writer=PillowWriter(fps=fps))