from core.solver import WesterveltSolver, WesterveltParams
from core.postprocessing import plot_stability_scan


def run_westervelt_experiment():
    # Paramètres
    params = WesterveltParams(
        c=1500,
        rho0=1000,
        beta=4.8,
        mu_v=6e-6,
        dx=2.8e-5,
        dt=1.8e-8,
        nx=2000,
        nt=1200,
        scheme="semi_implicit",
    )

    solver = WesterveltSolver(params)
    solver.initialize(u0_type="gaussian")

    # Temps de snapshot
    times = [0.0, 1e-6, 2e-6, 4e-6, 6e-6, 8e-6]
    print("Demarrage de la simulation semi-implicite...")
    snapshots = solver.run_with_snapshots(times, store_energy=True)
    print("Simulation terminée.")

    # Visualisation
    solver.plot_snapshots(snapshots)
    solver.plot_energy()

    dt_values = [1.2e-8, 1.8e-8, 2.5e-8, 3.0e-8]
    amp_values = [0.5, 1.0, 1.5]
    stability = solver.run_stability_scan(dt_values, amp_values, blowup_threshold=1e5)

    stable_count = sum(1 for r in stability if r["stable"])
    print(f"Configurations stables: {stable_count}/{len(stability)}")
    plot_stability_scan(stability)


if __name__ == "__main__":
    run_westervelt_experiment()
