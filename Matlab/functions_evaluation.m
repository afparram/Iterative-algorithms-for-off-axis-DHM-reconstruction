%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title: functions_evaluation                                                   %
%                                                                              %
% The script contains all implemented function for the evaluation_main.pyget   %
%                                                                              %
% Authors: Raul Castaneda and Ana Doblas                                       %
% Applied Optics Group EAFIT univeristy                                        %
%                                                                              %
% Email: racastaneq@eafit.edu.co; adoblas@umassd.edu                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef functions_evaluation
    % Clase para la evaluación y procesamiento de hologramas en microscopía holográfica digital.

    properties
        holo        % Holograma
        n           % Número de columnas
        m           % Número de filas
        N           % Coordenada horizontal
        M           % Coordenada vertical
        lambda_     % Longitud de onda
        dxy         % Tamaño del pixel
        k           % Número de onda
        fx_0        % Centro del holograma en x
        fy_0        % Centro del holograma en y
    end

    methods
        function obj = functions_evaluation(holo_path)
            % Constructor de la clase
            [obj.holo, obj.n, obj.m, obj.N, obj.M] = obj.holo_read(holo_path);
            [obj.lambda_, obj.dxy] = functions_evaluation.get_dhm_parameters(holo_path);
            obj.k = 2 * pi / obj.lambda_;
            obj.fx_0 = obj.m/2;
            obj.fy_0 = obj.n/2;
        end

        function [holo, n, m, N, M] = holo_read(obj, pathfile)
            % Lee la imagen holográfica y genera los ejes de coordenadas
            holo = im2double(imread(pathfile));
            if size(holo, 3) > 1
                holo = holo(:,:,1);
            end
            [n, m] = size(holo);
            [N, M] = meshgrid(-m/2:m/2-1, -n/2:n/2-1);
        end

        function [holo_filt_circ, fx_max, fy_max] = circular_filter(obj, visual, scale)
            % Aplica un filtro circular al holograma
            if nargin < 2
                visual = true;
            end
            if nargin < 3
                scale = 1;
            end

            mitad_m = obj.fx_0 - int64(obj.m * 0.05);

            % Filtrar la mitad derecha
            ft_holo = fftshift(fft2(fftshift(obj.holo)));
            mask_rectangle = ones(obj.n, obj.m);
            mask_rectangle(:,mitad_m:end) = 0;
            ft_holo_filt = ft_holo .* mask_rectangle;

            % Encontrar picos máximos
            [~, linear_index] = max(abs(ft_holo_filt(:)));
            [fy_max, fx_max] = ind2sub(size(ft_holo_filt), linear_index);

            % Máscara circular
            rmax = min([fx_max, abs(obj.fx_0 - fx_max), fy_max, abs(mitad_m - fy_max)]) * scale;
            [R, P] = meshgrid(1:obj.m, 1:obj.n);
            circ_mask = (obj.distancia(R, P, fx_max, fy_max) <= rmax);
            ft_holo_filt_circ = ft_holo_filt .* circ_mask;
            holo_filt_circ = fftshift(ifft2(ifftshift(ft_holo_filt_circ)));

            if visual
                figure;
                imagesc(log(abs(ft_holo_filt_circ).^2));
                title('FT circular filter');
                colormap gray;
                daspect([1 1 1]);

                figure;
                imagesc(log((abs(holo_filt_circ)).^2));
                title('FT Filtered Hologram');
                colormap gray;
                daspect([1 1 1]);
            end
        end

        function ref_wave = reference_wave(obj, fx_max, fy_max)
            % Genera la onda de referencia
            u = (obj.fx_0 - fx_max) * obj.lambda_ / (obj.m * obj.dxy);
            v = (obj.fy_0 - fy_max) * obj.lambda_ / (obj.n * obj.dxy);
            coef = 1j * obj.k * obj.dxy;
            phase = coef * (u * obj.N + v * obj.M);
            ref_wave = exp(phase);
        end

        function phase = ang_spectrum(obj, fx_max, fy_max, holo_filtered, visual, name)
            % Calcula la fase del holograma reconstruido
            if nargin < 5
                visual = false;
            end
            if nargin < 6
                name = 'FT phase reconstruction';
            end

            ref_wave = obj.reference_wave(fx_max, fy_max);
            holo_rec = holo_filtered .* ref_wave;
            phase = angle(holo_rec);

            if visual
                figure;
                imagesc(phase);
                title(name);
                colormap gray;
                daspect([1 1 1]);
                % saveas(gcf, [name '.png']);
            end
        end

        function cf = cost_function(obj, x, holo_filtered)
            % Función de costo basada en la binarización de la fase
            fx_max = x(1);
            fy_max = x(2);

            % Obtener fase y normalizar
            phase = obj.ang_spectrum(fx_max, fy_max, holo_filtered);
            phase_norm = uint8(255 * mat2gray(phase));

            % Binarizar
            phase_bin = imbinarize(phase_norm, 0.1);

            % Calcular costo
            cf = obj.n * obj.m - sum(phase_bin(:));
        end
    end

    methods(Static)
        function d = distancia(x1, y1, x2, y2)
            % Calcula la distancia euclidiana
            d = sqrt((x1 - x2).^2 + (y1 - y2).^2);
        end

        function [wavelength, pixel_size] = get_dhm_parameters(holo_path)
            % Extrae parámetros del nombre del archivo
            [~, name_no_ext] = fileparts(holo_path);
            cadena = strsplit(lower(name_no_ext), '_');
            wavelength = str2double(cadena{end-1});
            pixel_size = str2double(cadena{end});
        end
    end
end
