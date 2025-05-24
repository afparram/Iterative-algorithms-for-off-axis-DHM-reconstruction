classdef AlgoritmosOptimizacion
    % Clase que implementa diferentes algoritmos de optimización para la reconstrucción de hologramas.

    properties
        holograma          % Objeto que contiene las funciones para la reconstrucción
        holograma_filtrado % Holograma tras aplicar filtrado
        x0                % Valor inicial para la optimización [fx_max, fy_max]
        tol               % Tolerancia para la condición de parada
        max_iter          % Número máximo de iteraciones
        bounds            % Límites de búsqueda para los algoritmos
        cost_function     % Función de costo para optimización
    end

    methods
        function obj = AlgoritmosOptimizacion(mi_holograma, holograma_filtrado, x0, bounds, tol, max_iter)
            % Constructor de la clase
            if nargin < 4
                bounds = [0 0; mi_holograma.m mi_holograma.n];
            end
            if nargin < 5
                tol = 1e-6;
            end
            if nargin < 6
                max_iter = 100;
            end

            obj.holograma = mi_holograma;
            obj.holograma_filtrado = holograma_filtrado;
            obj.x0 = x0;
            obj.tol = tol;
            obj.max_iter = max_iter;
            obj.bounds = bounds;
            obj.cost_function = @(x)mi_holograma.cost_function(x, holograma_filtrado);
        end

        function [resultado, tiempo] = algoritmo_bfgs(obj)
            % Optimización con BFGS (fminunc)
            options = optimset('Display', 'off', ...
                'MaxIter', obj.max_iter, ...
                'TolFun', obj.tol, 'TolX', obj.tol);
            tic;
            [x, fval] = fminunc(obj.cost_function, obj.x0, options);
            tiempo = toc;
            resultado = struct('valor', x, 'costo', fval, 'tiempo', tiempo);
        end

        function [resultado, tiempo] = algoritmo_lsq(obj)
            % Optimización con Trust-Region-Reflective (lsqnonlin)
            options = optimoptions('lsqnonlin', ...
                    'Algorithm', 'trust-region-reflective', ...
                    'FunctionTolerance',      obj.tol, ...
                    'MaxIterations', obj.max_iter, ...
                    'Display',   'off');
            tic;
            [x, ~] = lsqnonlin(obj.cost_function, obj.x0, obj.bounds(1,:), obj.bounds(2,:), options);
            tiempo = toc;
            resultado = struct('valor', x, 'costo', obj.holograma.cost_function(x, obj.holograma_filtrado), 'tiempo', tiempo);
        end

        function [resultado, tiempo] = algoritmo_pso(obj)
            % Optimización con Particle Swarm Optimization (PSO)
            options = optimoptions('particleswarm', ...
                'Display', 'off', ...
                'MaxIterations', obj.max_iter, ...
                'FunctionTolerance', obj.tol, ...
                'SwarmSize', 3);
            tic;
            [x, fval] = particleswarm(obj.cost_function, 2, obj.bounds(1,:), obj.bounds(2,:), options);
            tiempo = toc;
            resultado = struct('valor', x, 'costo', fval, 'tiempo', tiempo);
        end

        function [resultado, tiempo] = algoritmo_nelder_mead(obj)
            % Optimización con Nelder-Mead (fminsearch)
            options = optimset('Display', 'off', ...
                'MaxIter', obj.max_iter, ...
                'TolFun', obj.tol, 'TolX', obj.tol);
            tic;
            [x, fval] = fminsearch(obj.cost_function, obj.x0, options);
            tiempo = toc;
            resultado = struct('valor', x, 'costo', fval, 'tiempo', tiempo);
        end

        function [resultado, tiempo] = algoritmo_slsqp(obj)
            % Optimización con SLSQP (fmincon)
            options = optimset('Display', 'off', ...
                'MaxIter', obj.max_iter, ...
                'TolFun', obj.tol, 'TolX', obj.tol);
            tic;
            [x, fval] = fmincon(obj.cost_function, obj.x0, [], [], [], [], obj.bounds(1,:), obj.bounds(2,:), [], options);
            tiempo = toc;
            resultado = struct('valor', x, 'costo', fval, 'tiempo', tiempo);
        end

        function resultados = ejecutar_todos(obj)
            % Ejecuta todos los algoritmos de optimización
            algoritmos = {'BFGS', 'LSQ', 'PSO', 'Nelder-Mead', 'SLSQP'};
            resultados = struct();

            for i = 1:length(algoritmos)
                switch algoritmos{i}
                    case 'BFGS'
                        [resultados.BFGS, ~] = obj.algoritmo_bfgs();
                    case 'LSQ'
                        [resultados.LSQ, ~] = obj.algoritmo_lsq();
                    case 'PSO'
                        lista(15) = struct('valor', [], 'costo', [], 'tiempo', []);
                        for i = 1:15
                            [lista(i), ~] = obj.algoritmo_pso();
                        end
                        
                        costos  = [lista.costo]; 
                        tiempos = [lista.tiempo];
                        
                        % Calculo medianas
                        medCosto  = median(costos);
                        medTiempo = median(tiempos);
                        
                        % Localizar la iteración cuya 'costo' está en la posición mediana
                        [~, orden]     = sort(costos);          
                        idxMedCosto    = orden( ceil(15/2) );
                     
                        % Obtener fx, fy de esa misma iteración
                        fx_med = lista(idxMedCosto).valor(1);
                        fy_med = lista(idxMedCosto).valor(2);

                        resultados.PSO = struct('valor', [fx_med, fy_med], 'costo', medCosto, 'tiempo', medTiempo);

                    case 'Nelder-Mead'
                        [resultados.Nelder_Mead, ~] = obj.algoritmo_nelder_mead();
                    case 'SLSQP'
                        [resultados.SLSQP, ~] = obj.algoritmo_slsqp();
                end
            end
        end

        function mostrar_resultados(obj, resultados)
            % Muestra los resultados en formato de tabla
            algoritmos = fieldnames(resultados);
            fprintf('\nResultados de la optimización:\n');
            fprintf('%-12s %-8s %-8s %-10s %-10s\n', ...
                'Método', 'fx', 'fy', 'Costo', 'Tiempo (s)');
            fprintf('--------------------------------------------------------\n');

            for i = 1:length(algoritmos)
                metodo = algoritmos{i};
                info = resultados.(metodo);
                fprintf('%-12s %8.4f %8.4f %10.4f %10.4f\n', ...
                    metodo, info.valor(1), info.valor(2), info.costo, info.tiempo);
            end
        end
    end
end