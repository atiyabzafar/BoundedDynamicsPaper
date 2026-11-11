import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import concurrent.futures
import multiprocessing
import csv

# ---------------------------------------------------------
# Solver And Helper Functions (DO NOT EDIT)
# ---------------------------------------------------------

def normalize_population(y, eps=1e-12):
    """
    Takes population and divides by the sum.
    Also keep it within bound for numerics
    """
    y = np.asarray(y, dtype=float).copy()
    y = np.clip(y, 0.0, None)
    s = y.sum()
    if s <= eps or not np.isfinite(s):
        return np.full_like(y, 1.0 / len(y), dtype=float)
    return y / s

def random_static_network(N, p_link, seed=None):
    """
    Generating Weighted-random Erdos Reniyi directed network 
    """
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < p_link).astype(float)
    np.fill_diagonal(A, 0.0)
    return A

def glv_fitness(r, B, y):
    """
    Fitness function for the generalised lotka volterra model
    f_i=r_i+sum_j b_ijy_j
    """
    return r + B @ y

def bounded_fitness_old(r, A, Alpha, y, eps=1e-12):
    """
    Fitness function for the bounded model
    f_i = r_i + \sum_j \alpha_{ij} y_j / (y_i + y_j)
    """
    denom = y[:, None] + y[None, :] + eps
    ratio = y[None, :] / denom
    W = A 
    return r + np.sum(W * ratio, axis=1)


def bounded_fitness(r, A, Alpha, y, eps=1e-12):
    """
    Fitness function for the Bounded Replicator model.
    f_i = r_i + \sum_j \alpha_{ij} y_j / (y_i + y_j)
    """
    denom = y[:, None] + y[None, :] + eps
    ratio = y[None, :] / denom

    return r + np.sum(Alpha * ratio, axis=1)

def rhs_static_network(t, y, model, A, r, B=None, Alpha=None, c=1.0):
    """
    Defining the dynamical system for the replicator equation with their own fitness function
    RHS of:
    $$
    \dot{y}_i=y_i\left[ f_i(y) -\sum_k \left(\frac{y_k}{c}\right) f_k(y) \right]
    $$
    """
    y_safe = np.clip(y, 0.0, None)

    # First, calculate the fitness for all species
    if model == "GLV":
        f = glv_fitness(r, B, y_safe)
    elif model == "Bounded":
        f = bounded_fitness(r, A, Alpha, y_safe)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Calculate the coupled dilution term: sum(y_k * f_k) / c
    dilution_term = np.dot(y_safe, f) / c

    return y_safe * (f - dilution_term)

def count_survivors(y, threshold=1e-6):
    """
    Helper function to count the number of surviving species based on the predefined threshold
    """
    y = np.asarray(y, dtype=float)
    return int(np.sum(y > threshold))


def solve_static_network_with_survival(
    model="GLV", N=20, p_link=0.25, c=1.0, seed=42, T=100.0, r_min=0.1, r_max=1.0, survival_threshold=1e-6, alpha_scale=1.0, beta_scale=50
):
    """
    So;ves the dynamical system for a random network for a seed and link probability
    """
    rng = np.random.default_rng(seed)
    
    # Generate network A first (0 or 1, diagonal 0)
    A = (rng.random((N, N)) < p_link).astype(float)
    np.fill_diagonal(A, 0.0)
    
    # Generate r_i which remains the same for the network
    r = rng.uniform(r_min, r_max, size=N)
    
    y0 = normalize_population(rng.random(N))

    # Calculate Alpha and Beta exactly
    Alpha = A.copy()
    Alpha[A != 0] = alpha_scale
    
    #B = Alpha * (N / (2.0 * c))  # beta_ij = alpha_ij * (S / 2c)
    B = Alpha * beta_scale

    if model == "GLV":
        rhs = lambda t, y: rhs_static_network(t, y, "GLV", A, r, B=B, c=c)
    elif model == "Bounded":
        rhs = lambda t, y: rhs_static_network(t, y, "Bounded", A, r, Alpha=Alpha, c=c)

    def event_equilibrium(t, y):
        dydt = rhs(t, y)
        y_safe = np.clip(y, 1e-12, None)
        rel_change = np.abs(dydt / y_safe)
        return np.max(rel_change) - 1e-8

    event_equilibrium.terminal = True
    event_equilibrium.direction = -1

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, T),
        y0=y0,
        method='BDF',
        events=event_equilibrium,
        rtol=1e-6,
        atol=1e-9,
        max_step=1.0,
    )

    y_final = normalize_population(sol.y[:, -1])
    
    return {
        "model": model,
        "p_link": p_link,
        "y_final": y_final,
        "r": r
    }

# ---------------------------------------------------------
# Parallel Execution Setup
# ---------------------------------------------------------

def run_simulation_task(args):
    """
    Runs the simulation for the given model and its parameters.
    Helps in parallel multi thread processing. 
    Takes a list of arguments which include the following in order,
    the model you are solving for ("GLV" for SRD and "bounded" for RRD)
    N: number of species in the network
    p_link: The connectance value, or the link probability
    c: Carrying capacity value 
    seed: seed used to generate this specific run
    T: Final time for integrator 
    threshold: Existence threshold beyond which we consider a species to be existing
    r_min: the minimum of the uniform distribution for sampling intinsic growth rate
    r_max: the maximum of the same uniform distribution
    """
    model, N, p_link, c, seed, T, threshold, r_min, r_max, beta_scale, alpha_scale = args
    try:
        res = solve_static_network_with_survival(
            model=model, N=N, p_link=p_link, c=c, seed=seed, T=T, 
            survival_threshold=threshold, r_min=r_min, r_max=r_max,
            alpha_scale = alpha_scale , beta_scale=beta_scale
        )
        y_final = res["y_final"]
        r = res["r"]
        
        # Isolate surviving species to analyze their intrinsic growth rates
        survivors_mask = y_final > threshold
        surviving_r = r[survivors_mask]
        
        return {
            "model": model,
            "p_link": p_link,
            "seed": seed,
            "survivors": int(np.sum(survivors_mask)),
            "mean_r_survivors": float(np.mean(surviving_r)) if len(surviving_r) > 0 else 0.0,
            "survivors_mask": str(survivors_mask.tolist()),
            "r_list": str(r.tolist()) # Saves exact array for detailed analysis
        }
    except Exception as e:
        print(f"Simulation failed for {model} at p={p_link}, seed={seed}: {e}")
        return None

if __name__ == '__main__':
    ####################################################
    # Sweep Parameters (EDIT THESE FOR RUNNING SIM)
    ######################################################
    p_values = np.linspace(0, 1.0, 101) # 101 values of p including 0
    realizations_per_p = 50 # 50 networks generated for each p value
    survival_threshold = 1e-4 # threshold for exiitence
    N = 100
    T = 2000.0
    c = 1.0
    r_min = 0.01
    r_max = 1.0

    ###########################################################
    alpha_scale=1.0
    beta_scale = N / (2 * c)
    ###########################################################
    #csv_filename = "simulation_results_full_withallspecies.csv"
    
    fieldnames = [
        "model", "p_link", "seed", "survivors",  
        "mean_r_survivors", "survivors_mask", "r_list"
    ]

    output_stem = (
                f"N{N}_T{T:g}_c{c:g}_r{r_min:g}-{r_max:g}_"
                f"threshold{survival_threshold:g}_reps{realizations_per_p}_"
                f"p{p_values[0]:g}-{p_values[-1]:g}_pcount{len(p_values)}_"
                f"betascale_{beta_scale}_alphascale_{alpha_scale}"
            )
    csv_filename = f"simulation_results_full_withallspecies_{output_stem}.csv"

    # Initialize CSV
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Generate tasks
    tasks = []
    for model in ["GLV", "Bounded"]:
        for p_link in p_values:
            for seed_offset in range(realizations_per_p):
                tasks.append((model, N, p_link, c, 42 + seed_offset, T, survival_threshold, r_min, r_max, beta_scale, alpha_scale))

    print(f"Total simulations to run: {len(tasks)}")
    
    # Execute and save incrementally
    raw_results = []
    max_workers = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Running on {max_workers} CPU cores. Saving live to {csv_filename}...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_simulation_task, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                raw_results.append(result)
                
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)

    print("\nSweep Complete. All data saved.")
    
    #Load and aggregate data for plotting
    df_raw = pd.read_csv(csv_filename)
    
    sweep_summary = df_raw.groupby(["model", "p_link"]).agg(
        mean_survivors=('survivors', 'mean'),
        sem_survivors=('survivors', lambda x: x.sem(ddof=1) if len(x) > 1 else 0.0),
    ).reset_index()

    print("\nPlotting Results...")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, grp in sweep_summary.groupby("model"):
        ax.errorbar(
            grp["p_link"], grp["mean_survivors"], yerr=grp["sem_survivors"],
            fmt='o-', capsize=5, lw=2, label=model_name,
        )
    ax.set_xlabel("Wiring probability p_link")
    ax.set_ylabel("Mean number of surviving species")
    ax.set_title(f"Mean survivors vs p_link across {realizations_per_p} networks per p")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/Diversity_{output_stem}.png")    
    plt.show()