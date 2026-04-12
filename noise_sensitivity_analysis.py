import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate_lqg
from metrics import summarize_run
from monte_carlo import CFG

def main():
    # Define range for measurement noise (rad)
    # Nominal sigma_v is 0.05
    sigma_v_range = np.linspace(0.01, 0.25, 12)
    
    results = {
        'lqg': {'mean_cost': [], 'std_cost': [], 'mean_error': []},
        'pd_noisy': {'mean_cost': [], 'std_cost': [], 'mean_error': []}
    }
    
    n_trials = 50  # Number of trials per noise level for statistical significance
    
    print(f"Starting Noise Level Sweep ({n_trials} trials per point)...")
    
    for sigma_v in sigma_v_range:
        print(f"  Testing sigma_v = {sigma_v:.3f} rad")
        
        # Temporary storage for this noise level
        level_costs_lqg = []
        level_errors_lqg = []
        level_costs_pd = []
        level_errors_pd = []
        
        # Update config for this level
        current_cfg = copy.deepcopy(CFG)
        current_cfg['measurement']['sigma_v'] = float(sigma_v)
        
        for seed in range(n_trials):
            # Run LQG
            res_lqg = simulate_lqg(current_cfg, seed=seed, controller_mode='lqg', estimator='kf')
            m_lqg = summarize_run(res_lqg['t'], res_lqg['x'], res_lqg['u'], 
                                  Q=current_cfg['lqr']['Q'], Ru=current_cfg['lqr']['Ru'])
            level_costs_lqg.append(m_lqg['lqr_cost'])
            level_errors_lqg.append(m_lqg['rms_theta'])
            
            # Run PD (Noisy)
            res_pd = simulate_lqg(current_cfg, seed=seed, controller_mode='pd_noisy', estimator='kf')
            m_pd = summarize_run(res_pd['t'], res_pd['x'], res_pd['u'], 
                                 Q=current_cfg['lqr']['Q'], Ru=current_cfg['lqr']['Ru'])
            level_costs_pd.append(m_pd['lqr_cost'])
            level_errors_pd.append(m_pd['rms_theta'])
            
        # Store aggregated results
        results['lqg']['mean_cost'].append(np.mean(level_costs_lqg))
        results['lqg']['std_cost'].append(np.std(level_costs_lqg))
        results['lqg']['mean_error'].append(np.mean(level_errors_lqg))
        
        results['pd_noisy']['mean_cost'].append(np.mean(level_costs_pd))
        results['pd_noisy']['std_cost'].append(np.std(level_costs_pd))
        results['pd_noisy']['mean_error'].append(np.mean(level_errors_pd))

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Total Quadratic Cost J
    ax1.set_xlabel('Measurement Noise Std Dev σ_v (rad)')
    ax1.set_ylabel('Mean Quadratic Cost J', color='tab:blue')
    ax1.plot(sigma_v_range, results['lqg']['mean_cost'], 'o-', label='LQG Cost J', color='tab:blue', linewidth=2)
    ax1.plot(sigma_v_range, results['pd_noisy']['mean_cost'], 's--', label='PD Cost J', color='tab:cyan', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for RMS Pitch Error
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean RMS Pitch Error (rad)', color='tab:red')
    ax2.plot(sigma_v_range, results['lqg']['mean_error'], '^-', label='LQG RMS Error', color='tab:red', linewidth=2)
    ax2.plot(sigma_v_range, results['pd_noisy']['mean_error'], 'x--', label='PD RMS Error', color='tab:orange', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Noise Sensitivity Sweep: LQG vs PD Performance')
    fig.tight_layout()
    
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    os.makedirs('outputs/sensitivity', exist_ok=True)
    plt.savefig('outputs/sensitivity/noise_sweep.png', dpi=200)
    print("\nResults saved to outputs/sensitivity/noise_sweep.png")

if __name__ == '__main__':
    main()