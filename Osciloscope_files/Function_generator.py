import numpy as np
import matplotlib.pyplot as plt

# Parâmetros globais
N = 8192  # 8192 pontos
T = 1e-6      # Duração - 1 microsegundo
t = np.linspace(0, 1e-6, N, endpoint=False)   # Array de tempo global

def chirp_gen(C_freq, Bw, sweep_time):

    """Gera um sinal chirp linear decrescente
    
    Args:
        C_freq: Frequência central (Hz)
        Bw: Bandwidth (Hz)
        sweep_time: Tempo de varredura (s) (tempo até alcançar a frequência desejada)
    """
    phase = 2*np.pi * (C_freq*t + (Bw/(2*sweep_time))*t*(sweep_time - t))
    return np.cos(phase)

signal = chirp_gen(868e6, 800e6, T)

# FFT
fft_signal = np.fft.fft(signal)

fig, axs = plt.subplots(2)
fig.suptitle('Sinal e sua FFT')
axs[0].plot(t, signal)
axs[1].plot(t,fft_signal)
plt.show()


