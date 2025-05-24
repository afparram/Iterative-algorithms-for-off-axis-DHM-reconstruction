import time
import numpy as np
import scipy.optimize as so
import os
import pyswarms as ps
import pygad
from multiprocessing import Pool, cpu_count
from itertools import product

from tabulate import tabulate
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from functools import wraps
import logging
import warnings

@dataclass
class ResultadoOptimizacion:
    valor: np.ndarray
    costo: float
    tiempo: float

# Decorador para medir tiempo
def medir_tiempo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        out['tiempo'] = time.perf_counter() - start
        return out
    return wrapper

def format_bounds(bounds: Tuple[np.ndarray, np.ndarray]) -> List[Tuple[float, float]]:
    lb, ub = bounds
    return [(float(lb[i]), float(ub[i])) for i in range(len(lb))]

class AlgoritmosOptimizacion:
    def __init__(
        self,
        mi_holograma: Any,
        holograma_filtrado: np.ndarray,
        x0: List[float],
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        tol: float = 1e-6,
        max_iter: int = 100
    ):
        self.holograma = mi_holograma
        self.holo = holograma_filtrado
        self.x0 = np.array(x0, dtype=float)
        self.bounds = bounds or [[0, 0], [self.holograma.m, self.holograma.n]]
        self.tol = tol
        self.max_iter = max_iter

    def f_obj(self, x: np.ndarray) -> float:
        return self.holograma.cost_function(x, self.holo)

    def f_obj_vectorized(self, X: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray de forma (n_particles, dim)
        Devuelve np.ndarray con los costes de cada fila de X, paralelizado.
        """
        # pool.map requiere lista de tuplas/arrays, así que convertimos
        return np.array([self.f_obj(xi) for xi in X])

    def f_obj_ga(self, ga_instance, solution, solution_idx):
        cost = self.f_obj(solution)
        return -cost

    # ---------- Métodos de optimización local (scipy.minimize) ----------
    @medir_tiempo
    def algoritmo_nelder_mead(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='Nelder-Mead',
                          bounds=format_bounds(self.bounds),
                          options={'fatol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_powell(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='Powell',
                          bounds=format_bounds(self.bounds),
                          options={'ftol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_cg(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='CG',
                          options={'gtol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_lsq(self, **kwargs) -> Dict[str, Any]:
        res = so.least_squares(fun=self.f_obj, x0=self.x0, bounds=self.bounds,
                               ftol=self.tol, verbose=0, max_nfev=self.max_iter)
        return {'valor' : res.x, 'costo' : self.f_obj(res.x)}

    @medir_tiempo
    def algoritmo_bfgs(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='BFGS',
                          options={'gtol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_l_bfgs_b(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='L-BFGS-B',
                          bounds=format_bounds(self.bounds),
                          options={'ftol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_tnc(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='TNC',
                          bounds=format_bounds(self.bounds),
                          options={'ftol': self.tol, 'maxfun': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_cobyla(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='COBYLA',
                          options={'tol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    @medir_tiempo
    def algoritmo_slsqp(self, **kwargs) -> Dict[str, Any]:
        res = so.minimize(self.f_obj, self.x0, method='SLSQP',
                          bounds=format_bounds(self.bounds),
                          options={'ftol': self.tol, 'maxiter': self.max_iter, 'disp': False})
        return {'valor': res.x, 'costo': res.fun}

    # --------------- Objetivos globales ----------------
    @medir_tiempo
    def algoritmo_basin_hopping(self, **kwargs) -> Dict[str, Any]:
        ret = so.basinhopping(self.f_obj, self.x0, **{'niter_success': 2, 'disp': False})
        return {'valor': ret.x, 'costo': ret.fun}

    @medir_tiempo
    def algoritmo_brute(self, **kwargs):
        res = so.brute(
            self.f_obj,
            ranges=format_bounds(self.bounds),
            full_output=True,
            finish=None,
            workers=-1
        )
        return {'valor': res[0], 'costo': res[1]}

    @medir_tiempo
    def algoritmo_differential_evolution(self) -> Dict[str, Any]:
        ret = so.differential_evolution(self.f_obj, bounds = format_bounds(self.bounds), popsize=5,
                                        tol = self.tol, maxiter = self.max_iter, updating='deferred',
                                        disp = False, workers=-1, x0=self.x0)
        return {'valor': ret.x, 'costo': ret.fun}

    @medir_tiempo
    def algoritmo_shgo(self) -> Dict[str, Any]:
        # 1) Guarda el nivel actual del root logger
        root_logger = logging.getLogger()
        old_level   = root_logger.level
        # 2) Sube el nivel para ignorar INFO (y DEBUG)
        root_logger.setLevel(logging.WARNING)
        try:
            ret = so.shgo(self.f_obj, bounds = format_bounds(self.bounds), iters=2,
                        options={'f_tol': self.tol, 'maxiter': self.max_iter, 'disp': False}, workers=1)
        finally:
            root_logger.setLevel(old_level)
        return {'valor': ret.x, 'costo': ret.fun}

    @medir_tiempo
    def algoritmo_dual_annealing(self, initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=-5.0) -> Dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ret = so.dual_annealing(self.f_obj, bounds = format_bounds(self.bounds),
                                    maxiter=self.max_iter, maxfun=self.max_iter, x0=self.x0,
                                    initial_temp=initial_temp, restart_temp_ratio=restart_temp_ratio,
                                    visit=visit, accept=accept)
        return {'valor': ret.x, 'costo': ret.fun}

    @medir_tiempo
    def algoritmo_direct(self, **kwargs) -> Dict[str, Any]:
        ret = so.direct(self.f_obj, bounds = format_bounds(self.bounds), maxiter=self.max_iter, f_min_rtol=self.tol)
        return {'valor': ret.x, 'costo': ret.fun}

    @medir_tiempo
    def algoritmo_pso(self, **kwargs) -> Dict[str, Any]:
        optimizer = ps.single.GlobalBestPSO(
            n_particles=3, dimensions=2, options= {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'ftol': self.tol}, bounds=self.bounds
        )
        cost, pos = optimizer.optimize(self.f_obj_vectorized, iters=self.max_iter, verbose=False, n_processes = os.cpu_count())
        return {'valor': pos, 'costo': cost}

    @medir_tiempo
    def algoritmo_pygad(self,
                        sol_per_pop: int = 15,
                        **kwargs
                        ) -> Dict[str, Any]:

        lb, ub = self.bounds
        gene_space = [ {'low': lb[i], 'high': ub[i]} for i in range(len(self.x0)) ]

        ga = pygad.GA(num_generations=self.max_iter,
                       gene_space=gene_space,
                       fitness_func=self.f_obj_ga,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=sol_per_pop//4,
                       num_genes=len(self.x0),
                       stop_criteria="saturate_2",
                       parallel_processing=["process", os.cpu_count()],
                       on_generation=None)

        ga.run()
        solution, best_cost, _ = ga.best_solution()
        return {'valor': np.array(solution), 'costo': -best_cost}

    def ejecutar_todos(self) -> Dict[str, Dict[str, Any]]:
        algoritmos = {
            'BFGS' : self.algoritmo_bfgs,
            'LSQ' : self.algoritmo_lsq,
            'SLSQP' : self.algoritmo_slsqp,
            'CG' : self.algoritmo_cg,
            'POWELL' : self.algoritmo_powell,
            'TNC' : self.algoritmo_tnc,
            'PYGAD' : self.algoritmo_pygad,
            'L-BFGS-B': self.algoritmo_l_bfgs_b,
            'COBYLA': self.algoritmo_cobyla,
            'Brute' : self.algoritmo.brute,
            'Basin-Hopping': self.algoritmo_basin_hopping,
            'SHGO': self.algoritmo_shgo,
            'Dual-Annealing': self.algoritmo_dual_annealing,
            'DIRECT': self.algoritmo_direct,
            'PSO': self.algoritmo_pso,
            'GA': self.algoritmo_pygad,
            'SHGO' : self.algoritmo_shgo
        }

        stochastic = {'Basin-Hopping', 'SHGO', 'Dual-Annealing', 'DIRECT', 'PSO', 'GA'}

        resultados = {}

        for name, func in algoritmos.items():
            if name in stochastic:
                costos = np.zeros(15)
                tiempos = np.zeros(15)
                vals = np.zeros((15, 2))
                for i in range(15):
                    out = func()
                    costos[i] = out['costo']
                    tiempos[i] = out['tiempo']
                    vals[i, :] = out['valor']
                # calcular medianas
                median_cost  = np.median(costos)
                median_time  = np.median(tiempos)

                median_idx   = np.argsort(costos)[15 // 2]

                median_fx, median_fy = vals[median_idx]

                resultados[name] = {
                    'valor': np.array([median_fx, median_fy]),
                    'costo': median_cost,
                    'tiempo': median_time,
                    'MAD_costo' : np.median(np.abs(costos - median_cost)),
                    'MAD_tiempo' : np.median(np.abs(tiempos - median_time)),
                }
            else:
                single = func()
                resultados[name] = {
                    'valor':             single['valor'],
                    'costo':             single['costo'],
                    'tiempo':            single['tiempo'],
                    'MAD_costo':   0.0,
                    'MAD_tiempo':  0.0
                }

        return resultados

    def mostrar_resultados(self, resultados: Dict[str, Dict[str, Any]], visual = True) -> None:
        """
        Muestra los resultados de la optimización en formato de tabla.

        Args:
            resultados: Diccionario con los resultados de la optimización
        """
        tabla = []
        for metodo, info in resultados.items():
            fx, fy = info['valor']
            tabla.append([
                metodo,
                np.round(fx, 4),
                np.round(fy, 4),
                info['costo'],
                np.round(info['tiempo'], 4)
            ])

        if visual:
            print(tabulate(
                tabla,
                headers=["Método", "fx", "fy", "Costo", "Tiempo (s)"],
                tablefmt="pretty"
            ))

        return tabla

    def evaluar_combinacion(self, params):
        """Función que evalúa una combinación específica de hiperparámetros"""
        it, rt, v, a = params
        key = f"DA_it{it}_rt{rt}_v{v}_a{a}"
        costos = np.zeros(15)
        tiempos = np.zeros(15)
        vals = np.zeros((15, 2))

        for i in range(15):
            out = self.algoritmo_dual_annealing(
                initial_temp=it,
                restart_temp_ratio=rt,
                visit=v,
                accept=a
            )
            costos[i] = out['costo']
            tiempos[i] = out['tiempo']
            vals[i] = out['valor']

        med_cost = np.median(costos)
        med_time = np.median(tiempos)
        mad_cost = np.median(np.abs(costos - med_cost))
        mad_time = np.median(np.abs(tiempos - med_time))
        idx_med = np.argsort(costos)[len(costos)//2]
        fx_med, fy_med = vals[idx_med]

        return key, {
            'valor': np.array([fx_med, fy_med]),
            'costo_mediana': med_cost,
            'tiempo_mediana': med_time,
            'MAD_costo': mad_cost,
            'MAD_tiempo': mad_time
        }

# # Ejemplo de uso
def main():
    # Ejemplo de una simulación
    from functions_evaluation import functions_evaluation
    import pandas as pd
    import matplotlib.pyplot as plt

    # Configuración de rutas
    path = os.path.dirname(os.path.abspath(__file__))
    filename_holo = 'holo-RBC-20p205-2-3_0.532_2.4.png'
    holo_path = os.path.join(path, "..", "PruebasMuestras", filename_holo)

    # Inicialización
    mi_holograma = functions_evaluation(holo_path)
    holo_filtered, fx_max, fy_max = mi_holograma.circular_filter(visual=False, scale=1)
    mi_holograma.ang_spectrum(fx_max, fy_max, holo_filtered, True, 'phase reconstruction with integer pixel')

    # Configuración de límites
    bounds = (np.array([fx_max - 1, fy_max - 1]), np.array([fx_max + 1, fy_max + 1]))

    # Crear instancia y ejecutar optimización
    algoOpt = AlgoritmosOptimizacion(
        mi_holograma,
        holo_filtered,
        [fx_max, fy_max],
        bounds=bounds,
        tol=1e-3,
        max_iter=15
    )

    result = algoOpt.algoritmo_dual_annealing(7500, 2e-7, 1.5, -0.5)
    fx_max = result['valor'][0]
    fy_max = result['valor'][1]
    mi_holograma.ang_spectrum(fx_max, fy_max, holo_filtered, True, 'phase reconstruction with non-integer pixel')


    # Hiperparámetros a explorar
    # hyperparams = {
    #     'initial_temp': [5230, 2500, 7500],
    #     'restart_temp_ratio': [2e-05, 2e-3, 2e-7],
    #     'visit': [2.62, 1.5, 2.0],
    #     'accept': [-5, -5e-1, -5e-2]
    # }

    # # Usar multiprocessing para evaluar las combinaciones en paralelo
    # n_cores = max(1, cpu_count())

    # # Generar todas las combinaciones de hiperparámetros
    # param_combinations = list(product(
    #     hyperparams['initial_temp'],
    #     hyperparams['restart_temp_ratio'],
    #     hyperparams['visit'],
    #     hyperparams['accept']
    # ))

    # with Pool(n_cores) as pool:
    #     resultados_hyper = dict(pool.map(algoOpt.evaluar_combinacion, param_combinations))

    # Crear DataFrame y guardar resultados
    # df = pd.DataFrame.from_dict(resultados_hyper, orient='index')
    # output_filename = f"{mi_holograma.n}x{mi_holograma.m}_{mi_holograma.lambda_}_{mi_holograma.dxy}.xlsx"
    # df.to_excel(output_filename)
    # print(f"Resultados guardados en {output_filename}")