import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Parâmetros padrão
DEFAULT_N = 2048  # 8192 pontos
DEFAULT_T = 1e-6  # Duração - 1 microsegundo

def chirp_gen(C_freq, Bw, N=DEFAULT_N, T=DEFAULT_T, return_time=False):

    """Gera um sinal chirp linear que varre toda a banda durante o tempo T.

    Args:
        C_freq: Frequência central (Hz)
        Bw: Bandwidth (Hz)
        N: Número de amostras
        T: Duração total do sinal (s)
        return_time: Se True, retorna também o vetor de tempo

    Returns:
        signal ou (t, signal)
    """
    # Validação de parâmetros
    if N <= 1:
        raise ValueError("N deve ser maior que 1")
    if T <= 0:
        raise ValueError("T deve ser maior que 0")
    if Bw < 0:
        raise ValueError("Bw não pode ser negativo")

    # Verifica critério de Nyquist
    f_max = C_freq + Bw/2
    Fs = N/T
    if Fs < 2 * f_max:
        raise ValueError(
            f"Violação do critério de Nyquist!\n"
            f"Frequência máxima do sinal: {f_max/1e6:.2f} MHz\n"
            f"Frequência de amostragem: {Fs/1e9:.3f} GHz\n"
            f"Mínimo requerido: {2*f_max/1e9:.3f} GHz\n"
            f"Aumente N para pelo menos {int(np.ceil(2*f_max*T))} pontos"
        )
    
    # Base de tempo local
    t = np.linspace(0, T, N, endpoint=False)

    # Frequência inicial e taxa de varredura
    f0 = C_freq - Bw/2  # Frequência inicial
    k = Bw / T  # Taxa de varredura (Hz/s) - varre toda a banda em T
    
    # Fase do chirp linear: φ(t) = 2π(f0*t + k*t²/2)
    phase = 2*np.pi * (f0*t + k*t**2/2)
    
    signal = np.cos(phase)
    if return_time:
        return t, signal
    return signal


if __name__ == "__main__":
    N = DEFAULT_N
    T = DEFAULT_T
    t = np.linspace(0, T, N, endpoint=False)

    # Original
    signal = chirp_gen(470e6, 140e6)

    # FFT
    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)
    freqs = np.fft.fftfreq(N, T/N)

    # Espectrograma
    f, t_spec, Sxx = spectrogram(signal, fs=N/T, nperseg=128)

    # Imagens
    fig = plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.plot(t*1e6, signal)
    plt.title('Sinal no domínio do tempo')
    plt.xlabel('Tempo [µs]')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(freqs[:N//2]/1e6, magnitude[:N//2])
    plt.title('Domínio da frequência (Magnitude)')
    plt.xlim(0, 1000)
    plt.xlabel('Frequência [MHz]')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.pcolormesh(t_spec*1e6, f/1e6, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequência [MHz]')
    plt.xlabel('Tempo [µs]')
    plt.title('Espectrograma')
    # plt.ylim(800, 1000)
    plt.colorbar(label='Potência [dB]')

    plt.tight_layout()
    plt.show()


