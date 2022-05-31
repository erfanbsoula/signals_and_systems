import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

sampling_time_duration = 3
sampling_frequency = 1000
signal_length = sampling_time_duration * sampling_frequency
faundamental_angular_freq = complex(0, 2*pi)
t = np.arange(0, sampling_time_duration, 1/sampling_frequency)
y = (1 + np.sin(2*pi*12*t)) * np.cos(np.sin(2*pi*25*t) + t)

t_normalized = t / sampling_time_duration
dft = np.zeros(signal_length, dtype=np.complex_)
for k in range(signal_length):
    complex_sine_wave = np.exp(-faundamental_angular_freq * k * t_normalized)
    dft[k] = np.dot(y, complex_sine_wave)

freqs = np.linspace(0, sampling_frequency, signal_length, endpoint=False)

inverse_dft = np.zeros(signal_length, dtype=np.complex_)
shots = np.zeros((30, 2, signal_length), dtype=np.complex_)

shot_index = 0
update_period = 20
counter = 0

for k in range(signal_length):
    complex_sine_wave = np.exp(faundamental_angular_freq * k * t_normalized)
    inverse_dft += (complex_sine_wave * dft[k])/signal_length
    counter += 1

    if k >= 200 and k <= 2800:
        update_period = 500
    else:
        update_period = 20

    if counter >= update_period or k == signal_length-1:
        counter = 0
        shots[shot_index][0] = inverse_dft
        shots[shot_index][1][:k+1] = dft[:k+1] / signal_length
        shot_index += 1

fig = plt.figure(figsize=(6.5, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

plt.ion()
fig.show()
fig.canvas.draw()

for k in range(shot_index):
    ax1.clear()
    ax2.clear()
    ax1.plot(t, y, label="Original Signal")
    ax1.plot(t, np.real(shots[k][0]), label="Reconstructed")
    ax2.plot(freqs, np.abs(shots[k][1]))
    ax1.legend(loc='best', prop={'size': 8})
    ax1.set_title("Time-Domain Signal")
    ax1.set_xlabel("time", color="orange")
    ax2.set_title("Fourier Transform")
    ax2.set_xlabel("frequency", color="orange")
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.2)

plt.pause(10)
print("done")