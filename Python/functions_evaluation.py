import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v2 as imageio
import os
import cv2
from numpy.lib.scimath import arcsin as asin

class functions_evaluation:
    """
    Clase para la evaluación y procesamiento de hologramas en microscopía holográfica digital.
    """

    def __init__(self, holo_path):
        self.holo, self.n, self.m, self.N, self.M = self.holo_read(holo_path)
        self.lambda_, self.dxy, self.filename = functions_evaluation.get_dhm_parameters(holo_path)
        self.k = 2 * np.pi / self.lambda_
        self.fx_0 = self.n // 2
        self.fy_0 = self.m // 2

    @staticmethod
    def distancia(x1, y1, x2, y2):
        """
        Cálcula la distancia euclidea entre pares ordenados. Esto no solo se restringue
        al caso entre un punto P y otro punto Q, sino que también se puede calcular
        la distancia de una malla con componentes x1 y y1 respecto a un simple punto (x,y).

        Parameters
        ----------
            x1 ndarray
                componente x del 1° par ordenado.
            y1 ndarray
                componente y del 2° par ordenado.
            x2 float
                componente x del 1° par ordenado.
            y2 float
                componente y del 2° par ordenado.

        Returns
        -------
            puede devolver un escalar (float) o una matriz (ndarray).
        """
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def graficar(self, complex_matrix, name, is_complex=True):
        """
        Grafica el plano complejo de una imagen que está contenida en una matriz.
        Puede ser la transformada de Fourier o la fase del holograma.

        Parameters
        ----------
            complex_matrix (ndarray)
                matriz de valores complejos.
            name (str)
                título del gráfico.

        Returns
        -------
            None.
        """
        if is_complex == True:
            magnitude = np.abs(complex_matrix)**2
            with np.errstate(divide='ignore'):
                objeto = np.log(magnitude)
        else:
            objeto = complex_matrix
        plt.figure()
        plt.imshow(objeto, cmap='gray')
        plt.title(f'{self.filename} - {name}')
        plt.gca().set_aspect('equal')

    def get_dhm_parameters(holo_path):
        """
        Extrae los parámetros de la muestra, es decir,
        la longitud de onda (λ) y el tamaño de pixel (Δxy)
        utilizados en el registro del holograma.

        Parameters
        ----------
            holo_path (ndarray)
                ruta absoluta del holograma,
                donde el nombre del archivo tiene la forma
                "nombre_lambda_dxy.extension"

        Returns
        -------
            lambda (float)
                número de longitud de onda del láser empleado.
            dxy (float)
                tamaño del pixel de la cámara usada
        """

        filename = os.path.basename(holo_path)
        name_no_ext = os.path.splitext(filename)[0].lower()
        cadena = name_no_ext.split('_')
        wavelength = float(cadena[-2])
        pixel_size = float(cadena[-1])

        return wavelength, pixel_size, filename

    @staticmethod
    def holo_read(pathfile):
        """
        Lee la imagen holográfica y genera los ejes de coordenadas.

        Parameters
        ----------
            pathfile (str)
                ruta al archivo de imagen.

        Returns
        -------
            holo (ndarray)
                Imagen en punto flotante n x m (se usa solo el primer canal si es RGB).
            n (int)
                Número de columnas (x).
            m (int)
                Número de filas (y).
            N (ndarray)
                Coordenada horizontal (similar a 'x').
            M (ndarray)
                Coordenada vertical (similar a 'y').
        """
        holo = imageio.imread(pathfile).astype(np.float64)
        if holo.ndim == 3:
            holo = holo[:, :, 0]
        m, n = holo.shape

        # Se crean los vectores de coordenadas. Se asume que n y m son pares.
        N, M = np.meshgrid(np.arange(-n//2, n//2), np.arange(-m//2, m//2))

        return holo, n, m, N, M

    def circular_filter(self, visual = True, scale = 1):
        """
        Aplica un filtro circular a un holograma.

        Parameters
        ----------
            visual (str)
                Si es 'Yes', se muestran figuras intermedias.
            factor (float)
                Factor para determinar el radio de la máscara circular.

        Returns
        -------
            holo_filt_circ (ndarray)
                Holograma filtrado (en el dominio espacial).
            fx_max (int)
                Coordenada x (columna) del pico detectado.
            fy_max (int)
                Coordenada y (fila) del pico detectado.
        """
        mitad_n = self.fx_0 - int(self.n * 0.05)

        # Filtrar la mitad derecha del rectángulo (DC y término -1 o +1 de difracción)
        ft_holo = fftshift(fft2(fftshift(self.holo)))
        mask_rectangle = np.ones((self.m, self.n))
        mask_rectangle[:,mitad_n:] = 0
        ft_holo_filt = ft_holo * mask_rectangle

        # Encontrar los picos máximos umax y vmax de forma discreta
        linear_index = np.argmax(np.abs(ft_holo_filt))
        fy_max, fx_max = np.unravel_index(linear_index, ft_holo_filt.shape)

        # Máscara circular
        rmax = min([fx_max, abs(self.fx_0 - fx_max), fy_max, abs(mitad_n - fy_max)])*scale
        R, P = np.meshgrid(np.arange(self.n), np.arange(self.m))
        circ_mask = (functions_evaluation.distancia(R, P, fx_max, fy_max) <= rmax)
        ft_holo_filt_circ = ft_holo_filt * circ_mask
        holo_filt_circ = fftshift(ifft2(ifftshift(ft_holo_filt_circ)))

        if visual == True:
            self.graficar(ft_holo_filt_circ, 'FT circular filter')
            self.graficar(holo_filt_circ, 'FT Filtered Hologram')

        return holo_filt_circ, fx_max, fy_max

    def reference_wave(self, fx_max, fy_max):
        """
        Genera la onda de referencia.

        Parameters
        ----------
            fx_max, fy_max (float)
                Coordenadas del punto máximo.

        Returns
        -------
            ref_wave (ndarray)
                Onda de referencia compleja.
        """
        u = (self.fx_0 - fx_max) * self.lambda_ / (self.n * self.dxy)
        v = (self.fy_0 - fy_max) * self.lambda_ / (self.m * self.dxy)
        coef = 1j * self.k * self.dxy
        phase = coef * (u * self.N + v * self.M)

        return np.exp(phase)

    def ang_spectrum(self, fx_max, fy_max, holo_filtered, visual = False, name = 'FT phase reconstruction'):
        """
        Calcula la fase del holograma reconstruido.

        Parameters
        ----------
            fx_max, fy_max (float)
                Coordenadas del punto máximo.
            holo_filtered (ndarray)
                Holograma filtrado.
            visual (bool)
                Si es True, se muestra la imagen.
            name (str)
                Nombre de la imagen.

        Returns:
            Matriz compleja que contiene la fase del holograma reconstruido
        """
        ref_wave = self.reference_wave(fx_max, fy_max)
        holo_rec = holo_filtered * ref_wave
        angle = np.angle(holo_rec)

        if visual == True:
            self.graficar(angle, name, False)
            plt.savefig(f'{name}.png', bbox_inches='tight')

        return angle

    def cost_function(self, x, holo_filtered):
        """
        Función de costo basada en la binarización de la fase del holograma reconstruido.

        Args:
            x: Vector [fx_max, fy_max] con las frecuencias espaciales
            holo_filtered: Holograma filtrado

        Returns:
            float: Valor de la función de costo
        """
        fx_max = x[0]
        fy_max = x[1]

        # Obtener la fase del holograma
        phase = self.ang_spectrum(fx_max, fy_max, holo_filtered)

        # Normalizar y binarizar usando las funciones equivalentes a MATLAB
        # Normaliza a [0,255] en uint8
        phase_u8 = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Binarizacion a partir del umbral 0.1*255
        _, mask = cv2.threshold(phase_u8, int(255*0.1), 255, cv2.THRESH_BINARY)
        cf = self.n * self.m - cv2.countNonZero(mask)

        return cf