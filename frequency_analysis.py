import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dynamics import continuous_matrices
from control import lqr_continuous

def main():
    # --- 1. Setup Plant and Controller ---
    # Nominal parameters from report
    a, b, c = 0.5, 0.8, 1.0
    Q = np.diag([100.0, 25.0])
    Ru = 3.7
    
    A, B, G = continuous_matrices(a, b, c, sigma=0.1)
    K, _ = lqr_continuous(A, B, Q, Ru)
    K = K.reshape(1, -1)
    
    # --- 2. Define Transfer Functions ---
    # Plant: G(s) = C(sI - A)^-1 B
    # For pitch angle control, we measure theta (index 0)
    C_meas = np.array([[1.0, 0.0]])
    sys_plant = signal.StateSpace(A, B, C_meas, 0)
    
    # Open-loop LQR Return Ratio: L(s) = K(sI - A)^-1 B
    # This represents the loop broken at the plant input
    sys_lqr_loop = signal.StateSpace(A, B, K, 0)
    
    # --- 3. Frequency Response ---
    w = np.logspace(-1, 2, 500)
    w_out, mag, phase = signal.bode(sys_lqr_loop, w)
    
    # --- 4. Compute Margins ---
    # Convert back from dB/degrees for margin calculation if needed, 
    # but scipy.signal.margin is better for this.
    # We'll use the Transfer Function form for margin calculation
    num, den = signal.ss2tf(A, B, K, 0)
    tf_loop = signal.TransferFunction(num[0], den)
    
    # --- 5. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Magnitude Plot
    ax1.semilogx(w_out, mag, linewidth=2, color='blue')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Bode Plot: LQR Open-Loop Return Ratio L(s)')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    # Phase Plot
    ax2.semilogx(w_out, phase, linewidth=2, color='red')
    ax2.axhline(-180, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Phase (deg)')
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    # Mark Gain Crossover Frequency
    idx_cross = np.argmin(np.abs(mag))
    w_gc = w_out[idx_cross]
    pm = phase[idx_cross] + 180
    
    ax1.plot(w_gc, 0, 'ko')
    ax2.plot(w_gc, phase[idx_cross], 'ko')
    ax2.annotate(f'Phase Margin: {pm:.1f}°\nat {w_gc:.2f} rad/s', 
                 xy=(w_gc, phase[idx_cross]), xytext=(w_gc*1.5, phase[idx_cross]+20),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    os.makedirs('outputs/frequency', exist_ok=True)
    plt.savefig('outputs/frequency/lqr_bode.png', dpi=200)
    
    print("-" * 30)
    print("FREQUENCY DOMAIN RESULTS")
    print("-" * 30)
    print(f"Gain Crossover Freq: {w_gc:.3f} rad/s")
    print(f"Phase Margin:        {pm:.2f} degrees")
    print("-" * 30)
    print("Saved plot to outputs/frequency/lqr_bode.png")

if __name__ == '__main__':
    main()