import numpy as np

class IQToAbsAugmentation():

    def __init__(self, keep_iq: bool = False):
        self.keep_iq = keep_iq

    def __call__(self, item: dict) -> dict:
        for key in list(item.keys()):
            # Process all keys starting with 'inputs'
            if not key.startswith('inputs'):
                continue
            
            suffix = key[6:]  # Remove 'inputs' prefix
            cir = item[key]
            
            # cirs are currently unbatched, so the shape is channel, antennas, taps
            cir_abs = np.linalg.norm(cir, axis=0, keepdims=True)
            
            if self.keep_iq:
                item[key] = np.concatenate([cir, cir_abs], axis=0)
            else:
                item[key] = cir_abs
        
        return item


class TimeToFrequencyAugmentation():
    """
    Konvertiert Kanal-Daten vom Zeitbereich (CIR) in den Frequenzbereich (CFR).
    
    Input: Channel Impulse Response (CIR) im Zeitbereich
           Shape: (2, antenna, taps) wobei 2 für Real/Imag steht
    
    Output: Channel Frequency Response (CFR) im Frequenzbereich
            Shape: (2, antenna, subcarrier)
    """

    def __init__(self, n_fft: int = None, normalize: bool = True):
        """
        Args:
            n_fft: Anzahl der FFT-Punkte (Subcarrier)
                   None = verwende Anzahl der Taps
            normalize: Normalisiere FFT-Output (teile durch sqrt(n_fft))
        """
        self.n_fft = n_fft
        self.normalize = normalize

    def __call__(self, item: dict) -> dict:
        """
        Konvertiert alle 'inputs*' keys vom Zeit- in den Frequenzbereich.
        
        Args:
            item: Dictionary mit Kanal-Daten
            
        Returns:
            Dictionary mit transformierten Kanal-Daten
        """
        for key in list(item.keys()):
            # Verarbeite nur keys die mit 'inputs' beginnen
            if not key.startswith('inputs'):
                continue
            
            cir = item[key]
            
            # Speichere den ursprünglichen dtype
            original_dtype = cir.dtype
            
            # cirs sind unbatched: shape (2, antennas, taps)
            # wobei dim 0: 0=Real, 1=Imag
            n_taps = cir.shape[2]
            
            # Bestimme FFT-Größe
            n_fft = self.n_fft if self.n_fft is not None else n_taps
            
            # Rekonstruiere komplexe Zahlen: (antennas, taps)
            cir_complex = cir[0] + 1j * cir[1]
            
            # FFT entlang der Zeit-Achse (letzte Achse)
            # (antennas, taps) -> (antennas, n_fft)
            cfr_complex = np.fft.fft(cir_complex, n=n_fft, axis=-1)
            
            # Normalisierung
            if self.normalize:
                cfr_complex = cfr_complex / np.sqrt(n_fft)
            
            # Trenne wieder in Real- und Imaginärteile: (2, antennas, n_fft)
            cfr = np.stack([
                np.real(cfr_complex),
                np.imag(cfr_complex)
            ], axis=0)
            
            # Stelle den ursprünglichen dtype wieder her
            cfr = cfr.astype(original_dtype)
            
            item[key] = cfr
        
        return item