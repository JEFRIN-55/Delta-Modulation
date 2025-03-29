# Delta Modulation (DM)
  JEFRIN INOLA J (212223060104).

## Aim  
To perform Delta Modulation (DM) on a continuous signal and analyze its reconstruction.  

## Tools Required  
- Python (colab)
- NumPy  
- Matplotlib  
- SciPy  

## Program  

### Delta Modulation and Demodulation  
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parameters
fs = 10000  # Sampling frequency
f = 10  # Signal frequency
T = 1  # Duration in seconds
delta = 0.1  # Step size

# Generate time vector
t = np.arange(0, T, 1/fs)

# Generate message signal (analog signal)
message_signal = np.sin(2 * np.pi * f * t)  # Sine wave as input signal

# Delta Modulation Encoding
encoded_signal = []
dm_output = [0]  # Initial value of the modulated signal
prev_sample = 0

for sample in message_signal:
    if sample > prev_sample:
        encoded_signal.append(1)
        dm_output.append(prev_sample + delta)
    else:
        encoded_signal.append(0)
        dm_output.append(prev_sample - delta)
    prev_sample = dm_output[-1]

# Delta Demodulation (Reconstruction)
demodulated_signal = [0]
for bit in encoded_signal:
    if bit == 1:
        demodulated_signal.append(demodulated_signal[-1] + delta)
    else:
        demodulated_signal.append(demodulated_signal[-1] - delta)

# Convert to numpy array
demodulated_signal = np.array(demodulated_signal)

# Apply a low-pass Butterworth filter
def low_pass_filter(signal, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

filtered_signal = low_pass_filter(demodulated_signal, cutoff_freq=20, fs=fs)

# Plotting the Results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, message_signal, label='Original Signal', linewidth=1)
plt.title("Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.step(t, dm_output[:-1], label='Delta Modulated Signal', where='mid')
plt.title("Delta Modulated Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal[:-1], label='Demodulated & Filtered Signal', linestyle='dotted', linewidth=1, color='r')
plt.title("Demodulated & Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```

## Output Waveforms  
- **Message Signal:** The original analog sine wave before Delta Modulation.
  
  ![image](https://github.com/user-attachments/assets/a71360a1-d8ed-4796-bdd7-4eaaa30568a8)

- **Delta Modulated Signal:** The stepwise encoded representation of the signal.
  
  ![image](https://github.com/user-attachments/assets/906099aa-7994-43fc-b3c6-3d60df3167d4)
  

- **Demodulated & Filtered Signal:** The reconstructed signal after filtering.
  
  ![image](https://github.com/user-attachments/assets/daad0a62-7937-44a8-a450-8d98c17bdd76)

## Results  
- The analog message signal was successfully sampled and encoded using Delta Modulation.  
