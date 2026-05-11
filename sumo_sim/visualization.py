import os
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List
from sumo_sim.control_loop import ControlResult

matplotlib.use("Agg")  

VEHICLE_COLORS = {
    10: ("#2196F3", "inside_1"),
    11: ("#03A9F4", "inside_2"),
    20: ("#FF5722", "outside_1"),
    21: ("#FF9800", "outside_2"),
    30: ("#4CAF50", "ramp_1"),
}

def plot_trajectories(results: List[ControlResult], save_path: str = "outputs/trajectories.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    time_series: dict = {}
    pos_series:  dict = {}
    lane_series: dict = {}

    for r in results:
        for v in r.vehicle_states:
            vid = v.vehicle_id
            if vid not in time_series:
                time_series[vid] = []
                pos_series[vid]  = []
                lane_series[vid] = []
            time_series[vid].append(r.time)
            pos_series[vid].append(v.position)
            lane_series[vid].append(v.lane_id)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Case Study 1", fontsize=14, fontweight="bold")

    gbd_time = next((r.time for r in results if r.gbd_triggered), None)

    ax1 = axes[0]
    for vid, times in time_series.items():
        col, label = VEHICLE_COLORS.get(vid, ("#999", f"v{vid}"))
        style = "--" if vid == 30 else "-"  
        ax1.plot(times, pos_series[vid], color=col, linestyle=style, linewidth=2, label=label)

    if gbd_time:
        ax1.axvline(gbd_time, color="red", linestyle=":", linewidth=1.5, label=f"GBD triggered (t={gbd_time:.1f}s)")

    ax1.set_ylabel("Position [m]", fontsize=11)
    ax1.set_title("Longitudinal Trajectories", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for vid, times in time_series.items():
        col, label = VEHICLE_COLORS.get(vid, ("#999", f"v{vid}"))
        style = "--" if vid == 30 else "-"
        ax2.plot(times, lane_series[vid], color=col,
                 linestyle=style, linewidth=2, label=label)

    if gbd_time:
        ax2.axvline(gbd_time, color="red", linestyle=":", linewidth=1.5)

    ax2.set_xlabel("Time [s]", fontsize=11)
    ax2.set_ylabel("Lane ID", fontsize=11)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Main Lane (0)", "On-Ramp (1)"])
    ax2.set_title("Lateral Movement (Lane Change)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved: {save_path}")

def plot_convergence(solver_history, save_path: str = "outputs/convergence.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    iterations   = [it.iteration    for it in solver_history]
    lower_bounds = [it.lower_bound  for it in solver_history]
    upper_bounds = [it.upper_bound  for it in solver_history]
    gaps         = [it.gap          for it in solver_history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("GBD Convergence Plot — LB & UB",
                 fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(iterations, lower_bounds, "b-o", linewidth=2,
             markersize=8, label="Lower Bound (LB)")
    ax1.plot(iterations, upper_bounds, "r-s", linewidth=2,
             markersize=8, label="Upper Bound (UB)")
    ax1.fill_between(iterations, lower_bounds, upper_bounds,
                     alpha=0.15, color="gray", label="Gap region")

    # Her noktaya değer yaz
    for i, (lb, ub) in enumerate(zip(lower_bounds, upper_bounds)):
        ax1.annotate(f"LB={lb:.1f}", (iterations[i], lb),
                     textcoords="offset points", xytext=(0, -18),
                     fontsize=8, color="blue", ha="center")
        ax1.annotate(f"UB={ub:.1f}", (iterations[i], ub),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=8, color="red", ha="center")

    ax1.set_ylabel("Objective Value", fontsize=11)
    ax1.set_title(f"LB and UB Convergence  |  {len(iterations)} iterations - CONVERGED", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    valid_gaps = [max(g, 1e-10) for g in gaps]
    ax2.semilogy(iterations, valid_gaps, "g-^", linewidth=2, markersize=8, label="Gap = UB - LB")
    ax2.axhline(1e-3, color="red", linestyle="--", linewidth=1.5, label="ε = 1e-3")

    for i, g in enumerate(gaps):
        ax2.annotate(f"{g:.2e}", (iterations[i], max(g, 1e-10)),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=8, color="green")

    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Gap (log scale)", fontsize=11)
    ax2.set_title("Convergence Speed", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved: {save_path}")

def plot_time_headways(results: List[ControlResult],save_path: str = "outputs/time_headways.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pairs = [
        (20, 30, "outside_1 → ramp_1"),   
        (21, 20, "outside_2 → outside_1"), 
        (10, 11, "inside_1 → inside_2"),   
    ]

    h_min = 1.72  

    positions: dict = {}
    velocities: dict = {}
    times: list = []

    for r in results:
        if r.time not in times:
            times.append(r.time)
        for v in r.vehicle_states:
            if v.vehicle_id not in positions:
                positions[v.vehicle_id]  = {}
                velocities[v.vehicle_id] = {}
            positions[v.vehicle_id][r.time]  = v.position
            velocities[v.vehicle_id][r.time] = v.velocity

    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Case Study 1 — Time Headway Graphs", fontsize=14, fontweight="bold")

    gbd_time = next((r.time for r in results if r.gbd_triggered), None)

    for ax, (leader_id, follower_id, pair_label) in zip(axes, pairs):
        headways = []
        valid_times = []

        for t in times:
            if (t in positions.get(leader_id, {}) and
                t in positions.get(follower_id, {}) and
                t in velocities.get(follower_id, {})):

                dx    = positions[leader_id][t] - positions[follower_id][t]
                v_f   = velocities[follower_id][t]
                if v_f > 0.1:
                    hw = dx / v_f
                    headways.append(max(hw, 0))
                    valid_times.append(t)

        col_l, _ = VEHICLE_COLORS.get(leader_id,   ("#999", ""))
        col_f, _ = VEHICLE_COLORS.get(follower_id, ("#999", ""))

        ax.plot(valid_times, headways, color=col_f, linewidth=2, label=f"hw({pair_label})")
        ax.axhline(h_min, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label=f"h_min = {h_min:.2f}s")

        if gbd_time:
            ax.axvline(gbd_time, color="purple", linestyle=":",
                       linewidth=1.5, alpha=0.6, label="GBD trigger")

        ax.fill_between(valid_times, 0, h_min,
                        alpha=0.08, color="red", label="Unsafe region")
        ax.set_ylabel("Headway [s]", fontsize=10)
        ax.set_title(pair_label, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time [s]", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Time headway graph saved: {save_path}")