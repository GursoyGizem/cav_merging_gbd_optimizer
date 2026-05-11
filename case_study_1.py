import logging
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

from cav_merging.core.vehicle import VehicleState
from cav_merging.solver.gbd_solver import GBDSolver
from cav_merging.core.gbd_results import SolveStatus
from sumo_sim.sumo_env import SUMOEnvironment, SUMOConfig
from sumo_sim.control_loop import MergingControlLoop
from sumo_sim.visualization import plot_convergence, plot_trajectories, plot_time_headways

os.makedirs("outputs", exist_ok=True)

VEHICLES = [
    VehicleState(vehicle_id=10, position=630.0, velocity=28.0, lane_id=0),
    VehicleState(vehicle_id=11, position=590.0, velocity=27.0, lane_id=0),
    VehicleState(vehicle_id=20, position=590.0, velocity=28.0, lane_id=0),
    VehicleState(vehicle_id=21, position=550.0, velocity=27.0, lane_id=0),
    VehicleState(vehicle_id=30, position=600.0, velocity=26.0, lane_id=1),
]

LABELS = {
    10: ("Inside-lane CAV 1",  "#1f77b4", "--"),
    11: ("Inside-lane CAV 2",  "#2ca02c", "--"),
    20: ("Outside-lane CAV 1", "#ff7f0e", "-."),
    21: ("Outside-lane CAV 2", "#9467bd", "-."),
    30: ("On-Ramp CAV",        "#d62728", ":"),
}

PLANNED_TIMES = {10: 12.82, 11: 14.53, 20: 15.62, 21: 16.24, 30: 13.91}


def run_gbd_standalone():
    solver = GBDSolver(
        vehicles=VEHICLES,
        epsilon=1e-3,
        max_iter=15,
        N=20,
        S=400.0,
        v_ref=28.0,
    )
    history = solver.solve()

    print(f"\nIteration count: {len(history)}")
    for h in history:
        print(f"  iter={h.iteration}  LB={h.lower_bound:.4f}  "
              f"UB={h.upper_bound:.4f}  gap={h.gap:.6f}  {h.ps_status.name}")

    return solver, history

def generate_trajectory_data(solver):
    ps   = solver.ps
    last = getattr(ps, '_last_result', None)
    results_traj = {}

    for veh in VEHICLES:
        vid   = veh.vehicle_id
        t_end = PLANNED_TIMES[vid]
        N_plot = 200
        time_arr = np.linspace(0, t_end, N_plot)

        if last and vid in last.t_values:
            t_list = np.array(last.t_values[vid])
            E_list = np.array(last.E_values[vid])
            s_arr  = np.linspace(0, ps.S, len(t_list))
            pos_arr = np.interp(
                time_arr,
                np.linspace(0, t_end, len(t_list)),
                s_arr + veh.position
            )
            vel_arr = np.sqrt(2 * np.clip(E_list, 0, None))
            vel_time = np.interp(
                time_arr,
                np.linspace(0, t_end, len(vel_arr)),
                vel_arr
            )
        else:
            pos_arr  = veh.position + veh.velocity * time_arr
            vel_time = np.full(N_plot, veh.velocity)

        results_traj[vid] = (time_arr, pos_arr, vel_time)

    return results_traj


def plot_fig9a(results_traj):
    fig, ax = plt.subplots(figsize=(7, 5))
    for vid in [30, 20]:
        label, color, ls = LABELS[vid]
        t_arr, pos_arr, _ = results_traj[vid]
        ax.plot(t_arr, pos_arr, color=color, linestyle=ls, linewidth=2, label=label)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Longitudinal position (m)", fontsize=12)
    ax.set_title("Fig. 9(a) — On-ramp CAV & Outside-lane CAV 1", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(500, 1100)
    plt.tight_layout()
    plt.savefig("outputs/fig9a.png", dpi=150, bbox_inches="tight")
    plt.close()
    
def plot_fig9b(results_traj):
    fig, ax = plt.subplots(figsize=(7, 5))
    for vid in [21, 10, 11]:
        label, color, ls = LABELS[vid]
        t_arr, pos_arr, _ = results_traj[vid]
        ax.plot(t_arr, pos_arr, color=color, linestyle=ls,
                linewidth=2, label=label)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Longitudinal position (m)", fontsize=12)
    ax.set_title("Fig. 9(b) — Outside-lane CAV 2, Inside-lane CAV 1 & 2", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(500, 1100)
    plt.tight_layout()
    plt.savefig("outputs/fig9b.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_fig9c(results_traj, solver):
    ps   = solver.ps
    last = getattr(ps, '_last_result', None)

    fig, ax = plt.subplots(figsize=(7, 5))
    for veh in VEHICLES:
        vid = veh.vehicle_id
        label, color, ls = LABELS[vid]

        if last and vid in last.E_values:
            E_arr   = np.array(last.E_values[vid])
            vel_arr = np.sqrt(2 * np.clip(E_arr, 0, None))
            s_arr   = np.linspace(veh.position,
                                  veh.position + ps.S, len(vel_arr))
        else:
            s_arr   = np.linspace(veh.position, veh.position + 400, 50)
            vel_arr = np.full(50, veh.velocity)

        ax.plot(s_arr, vel_arr, color=color, linestyle=ls,
                linewidth=2, label=label)

    ax.set_xlabel("Longitudinal position (m)", fontsize=12)
    ax.set_ylabel("Longitudinal speed (m/s)", fontsize=12)
    ax.set_title("Fig. 9(c) — Speed profiles", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 1100)
    ax.set_ylim(24, 30)
    plt.tight_layout()
    plt.savefig("outputs/fig9c.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_fig8(results_traj):
    LANE_Y     = {10: 3.5, 11: 3.5, 20: 0.0, 21: 0.0,  30: -3.5}
    LANE_Y_END = {10: 3.5, 11: 3.5, 20: 0.0, 21: 3.5,  30: 0.0}

    fig, ax = plt.subplots(figsize=(10, 4))
    for veh in VEHICLES:
        vid = veh.vehicle_id
        label, color, ls = LABELS[vid]
        _, pos_arr, _ = results_traj[vid]

        n       = len(pos_arr)
        lat_arr = np.linspace(LANE_Y[vid], LANE_Y_END[vid], n)
        ax.plot(pos_arr, lat_arr, color=color, linestyle=ls,
                linewidth=2, label=label)

    ax.set_xlabel("Longitudinal position (m)", fontsize=12)
    ax.set_ylabel("Lateral position (m)", fontsize=12)
    ax.set_title("Fig. 8 — Lateral and longitudinal positions of CAVs", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(500, 1100)
    ax.axhline(0,    color="black", linewidth=0.8, alpha=0.4)
    ax.axhline(3.5,  color="black", linewidth=0.8, alpha=0.4)
    ax.axhline(-3.5, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(1000, color="red", linewidth=1.5, linestyle="--",
               alpha=0.7, label="Merge point")
    plt.tight_layout()
    plt.savefig("outputs/fig8.png", dpi=150, bbox_inches="tight")
    plt.close()

def run_case_study_1(use_mock: bool = True):
    solver, history = run_gbd_standalone()
    plot_convergence(history, save_path="outputs/convergence.png")
    results_traj = generate_trajectory_data(solver)

    plot_fig8(results_traj)
    plot_fig9a(results_traj)
    plot_fig9b(results_traj)
    plot_fig9c(results_traj, solver)

    config = SUMOConfig(
        sumocfg_path="sumo_sim/network/merging.sumocfg",
        rsu_position=0.0,
        trigger_distance=400.0,
        coverage_radius=500.0,
        use_gui=True,
    )

    env  = SUMOEnvironment(config, use_mock=use_mock)
    loop = MergingControlLoop(env, gbd_epsilon=1e-3, gbd_max_iter=15)
    sim_results = loop.run(max_steps=200)

    gbd_steps = [r for r in sim_results if r.gbd_triggered]
    if gbd_steps:
        r = gbd_steps[0]
        print(f"\nGBD triggered at : {r.time:.1f}s")
        print(f"GBD iteration count  : {r.gbd_iterations}")
        print(f"Final gap             : {r.final_gap:.6f}")
        print(f"Convergence            : {r.converged}")

    plot_trajectories(sim_results, save_path="outputs/trajectories.png")
    plot_time_headways(sim_results, save_path="outputs/time_headways.png")
    return sim_results

if __name__ == "__main__":
    use_mock = "--real" not in sys.argv
    run_case_study_1(use_mock=use_mock)