import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
Er = 4.6
TanD = 0.03
FreqForEpsrTanD = 1e9 # 1 GHz
HighFreqForTanD = 1e12 # 1 THz
LowFreqForTanD = 1e3 # 1 kHz

# Define frequency points for calculation
freq = np.logspace(np.log10(LowFreqForTanD), np.log10(HighFreqForTanD), num=1000)

# Calculate permittivity using the model
ep_r = np.zeros_like(freq, dtype=np.complex128)
tand = np.zeros_like(freq, dtype=np.float64)
f_epr_tand = FreqForEpsrTanD * TanD
f_high = HighFreqForTanD
f_low = LowFreqForTanD
for i, f in enumerate(freq):
    k = np.log((f_high + 1j * f_epr_tand) / (f_low + 1j * f_epr_tand))
    fd = np.log((f_high + 1j * f) / (f_low + 1j * f))
    ep_d = -TanD * Er / np.imag(k)
    ep_inf = Er * (1 + TanD * np.real(k) / np.imag(k))
    ep_r[i] = ep_inf + ep_d * fd
    tand[i] = -np.imag(ep_r[i]) / np.real(ep_r[i])

# Print the complex permittivity at FreqForEpsrTanD
print(f"Complex permittivity at {FreqForEpsrTanD/1e9} GHz: {ep_r[np.argmin(np.abs(freq - FreqForEpsrTanD))]:.3f}")

# Plot Er and TanD on separate plots with grid and logarithmic x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.5)

# Plot Er on the first plot
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Er')
ax1.set_xscale('log')
ax1.grid(True, which='both')
ax1.plot(freq, np.real(ep_r))

# Plot TanD on the second plot
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('TanD')
ax2.set_xscale('log')
ax2.grid(True, which='both')
ax2.plot(freq, tand)

# Show the plots
ax1.set_title(f"Er vs Frequency (Er={Er}, TanD={TanD})")
ax2.set_title(f"TanD vs Frequency (Er={Er}, TanD={TanD})")
plt.show()