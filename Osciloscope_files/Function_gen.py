import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from ChirpGen import chirp_gen

# Parâmetros padrão
DEFAULT_N_PER_CHIRP = 2048
DEFAULT_T_PER_CHIRP = 1e-6
DEFAULT_NUM_CHIRPS = 1024


def generate_chirp_array(
    C_freq,
    Bw,
    num_chirps=DEFAULT_NUM_CHIRPS,
    N_per_chirp=DEFAULT_N_PER_CHIRP,
    T_per_chirp=DEFAULT_T_PER_CHIRP,
):
    """Gera um array de múltiplos chirps concatenados

    Args:
        C_freq: Frequência central (Hz)
        Bw: Bandwidth (Hz)
        num_chirps: Número de chirps a gerar
        N_per_chirp: Número de pontos por chirp
        T_per_chirp: Duração de cada chirp (s)

    Returns:
        signal: Array com todos os chirps concatenados
        t_total: Array de tempo total
    """
    if num_chirps <= 0:
        raise ValueError("num_chirps deve ser maior que 0")

    # Gera um chirp modelo usando a função do ChirpGen.py
    single_chirp = chirp_gen(C_freq, Bw, N=N_per_chirp, T=T_per_chirp)

    """ 

    Replica o chirp num_chirps vezes 
    (não sei se está correto pois não existe 
    "volta à frequência inicial")

    """
    signal = np.tile(single_chirp, num_chirps)

    # Array de tempo total
    N_total = N_per_chirp * num_chirps
    T_total = T_per_chirp * num_chirps
    t_total = np.linspace(0, T_total, N_total, endpoint=False)

    return signal, t_total


if __name__ == "__main__":
    # Configuração
    C_FREQ = 470e6
    BW = 140e6
    N_PER_CHIRP = DEFAULT_N_PER_CHIRP
    T_PER_CHIRP = DEFAULT_T_PER_CHIRP
    NUM_CHIRPS = DEFAULT_NUM_CHIRPS

    # Gera array de chirps
    signal, t = generate_chirp_array(
        C_FREQ,
        BW,
        num_chirps=NUM_CHIRPS,
        N_per_chirp=N_PER_CHIRP,
        T_per_chirp=T_PER_CHIRP,
    )

    N_total = signal.size
    T_total = NUM_CHIRPS * T_PER_CHIRP
    Fs = N_PER_CHIRP / T_PER_CHIRP

    print("Sinal gerado:")
    print(f"  Número de chirps: {NUM_CHIRPS}")
    print(f"  Duração por chirp: {T_PER_CHIRP*1e6:.1f} µs")
    print(f"  Duração total: {T_total*1e3:.2f} ms")
    print(f"  Total de pontos: {N_total}")
    print(f"  Taxa de amostragem: {Fs/1e9:.3f} GHz")

    # FFT
    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)
    freqs = np.fft.fftfreq(N_total, 1/Fs)

    # Espectrograma
    f, t_spec, Sxx = spectrogram(signal, fs=Fs, nperseg=128)

    # Imagens
    fig = plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(t*1e3, signal)
    plt.title(f'Sinal no tempo ({NUM_CHIRPS} chirps)')
    plt.xlabel('Tempo [ms]')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(freqs[:N_total//2]/1e6, magnitude[:N_total//2])
    plt.title('Domínio da frequência (Magnitude)')
    plt.xlim(0, 1000)
    plt.xlabel('Frequência [MHz]')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.pcolormesh(t_spec*1e3, f/1e6, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequência [MHz]')
    plt.xlabel('Tempo [ms]')
    plt.title('Espectrograma')
    plt.colorbar(label='Potência [dB]')

    plt.tight_layout()
    plt.show()
