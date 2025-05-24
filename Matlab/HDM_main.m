% Script principal para la evaluación de algoritmos de optimización en DHM
clear;
close all;
clc;

% Ruta de archivo de los hologramas
path = fileparts(mfilename('fullpath'));
folder_path = fullfile(path, '..', 'PruebasMuestras');

% Buscar todos los archivos de imagen
extensions = {'.tif', '.jpg', '.jpeg', '.png', '.bmp'};
hologramas = [];
for i = 1:length(extensions)
    files = dir(fullfile(folder_path, ['*' extensions{i}]));
    hologramas = [hologramas; files];
end

% Crear archivo Excel
filename = 'resultados_Matlab_2.xlsx';
if exist(filename, 'file')
    delete(filename);
end

% Inicializar lista de hojas existentes (vacía porque borraste el archivo)
existing_sheets = {};

% Procesar cada holograma
for i = 1:length(hologramas)
    filename_holo = hologramas(i).name;
    fprintf('Procesando holograma: %s\n', filename_holo);

    % Ruta absoluta del archivo del holograma
    holo_path = fullfile(folder_path, filename_holo);

    % Crear objeto que permita leer el holograma
    mi_holograma = functions_evaluation(holo_path);

    % Valor inicial de la reconstrucción de fase
    [holo_filtered, fx_max, fy_max] = mi_holograma.circular_filter(false, 1);
    mi_holograma.ang_spectrum(fx_max, fy_max, holo_filtered, false, 'Valor inicial');

    % Configuración de límites
    bounds = [fx_max - 1, fy_max - 1; fx_max + 1, fy_max + 1];

    % Crear instancia y ejecutar optimización
    algoOpt = AlgoritmosOptimizacion(mi_holograma, holo_filtered, [fx_max, fy_max], bounds);

    % Ejecutar todos los algoritmos
    resultados = algoOpt.ejecutar_todos();

    % Mostrar resultados
    algoOpt.mostrar_resultados(resultados);
    fprintf('\n');

    % Preparar datos para Excel
    metodos = fieldnames(resultados);
    n_metodos = length(metodos);

    % Crear tabla de resultados
    tabla = cell(n_metodos, 5);
    for j = 1:n_metodos
        metodo = metodos{j};
        info = resultados.(metodo);
        tabla{j,1} = metodo;
        tabla{j,2} = info.valor(1);
        tabla{j,3} = info.valor(2);
        tabla{j,4} = info.costo;
        tabla{j,5} = info.tiempo;
    end

    cadena = strsplit(filename_holo, '_');
    number = cadena{1};

    % Genera base de nombre
    base = sprintf('%dx%d_%s_%.3f_%.2f', mi_holograma.n, mi_holograma.m, number, ...
                   mi_holograma.lambda_, mi_holograma.dxy);
    name = base;
    counter = 1;

    % Asegura unicidad
    while any(strcmp(name, existing_sheets))
        name = sprintf('%s_%d', base, counter);
        counter = counter + 1;
    end
    sheet_name = name;
    existing_sheets{end+1} = sheet_name;

    % Guardar en Excel  
    writecell({'Método', 'fx', 'fy', 'Costo', 'Tiempo (s)'}, filename, 'Sheet', sheet_name, 'Range', 'A1:E1');
    writecell(tabla, filename, 'Sheet', sheet_name, 'Range', 'A2');
end