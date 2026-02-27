import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.optimize import newton, ridder, brentq, minimize, root
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from scipy import integrate, optimize
import plotly.subplots as sp
from tqdm import tqdm
from scipy.integrate import quad
from scipy.interpolate import griddata
from scipy import integrate
from scipy.optimize import root
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
log = np.log
exp = np.exp

class HestonModel:
    """Класс для реализации модели Хестона"""
    def __init__(self, S0: float, v0: float, kappa: float, theta: float, sigma: float, rho: float, r: float = 0.0, q: float = 0.0):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.q = q

    def heston_price_mc(self, K: float, T: float, num_paths: int = 20000, num_steps: int = 200, option_type: str = 'call') -> float:
        paths, _ = self.simulate_paths(T, num_paths, num_steps)
        ST = paths[:, -1]
        ST_antithetic = self.S0**2 / ST
        
        if option_type == 'call':
            payoff = np.maximum(ST - K, 0)
            payoff_antithetic = np.maximum(ST_antithetic - K, 0)
        elif option_type == 'put':
            payoff = np.maximum(K - ST, 0)
            payoff_antithetic = np.maximum(K - ST_antithetic, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
        avg_payoff = 0.5 * (np.mean(payoff) + np.mean(payoff_antithetic))
        return np.exp(-self.r * T) * avg_payoff
    
    def heston_price(self, K: float, T: float, option_type: str = 'call') -> float:
        """
        Аналитический расчет цены опциона по модели Хестона
        с использованием характеристической функции
        """
        def characteristic_function(phi: float, j: int) -> complex:
            if j == 1:
                u = 0.5
                b = self.kappa - self.rho * self.sigma
            else:
                u = -0.5
                b = self.kappa
            d = np.sqrt((self.rho * self.sigma * 1j * phi - b)**2 - 
                        self.sigma**2 * (2 * u * 1j * phi - phi**2))
            if j == 1:
                d = -d
                
            g = (b - self.rho * self.sigma * 1j * phi + d) / \
                (b - self.rho * self.sigma * 1j * phi - d)
                
            C = (self.r - self.q) * 1j * phi * T + \
                (self.kappa * self.theta / self.sigma**2) * \
                ((b - self.rho * self.sigma * 1j * phi + d) * T - 
                 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
                
            D = (b - self.rho * self.sigma * 1j * phi + d) / self.sigma**2 * \
                ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
                
            return np.exp(C + D * self.v0 + 1j * phi * np.log(self.S0))

        def integral(j: int) -> float:
            """Вычисление интеграла для вероятности Pj"""
            def integrand(phi: float) -> float:
                return np.real(np.exp(-1j * phi * np.log(K)) * 
                              characteristic_function(phi, j) / (1j * phi))
            
            integral_val, _ = quad(integrand, 1e-10, 100, limit=1000)
            return 0.5 + (1 / np.pi) * integral_val

        P1 = integral(1)
        P2 = integral(2)

        if option_type == 'call':
            price = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        else:
            price = K * np.exp(-self.r * T) * (1 - P2) - self.S0 * np.exp(-self.q * T) * (1 - P1)
            
        return price
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = 'call') -> float:
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        return price

    def calculate_implied_volatility(self, K: float, T: float, num_paths: int = 10000, num_steps: int = 100, option_type: str = 'call') -> float:
        price = self.heston_price(K, T, option_type)  # Добавлен option_type

        if option_type == 'call':
            intrinsic = max(self.S0 * np.exp(-self.q * T) - K * np.exp(-self.r * T), 0)  # Исправлено
        else:
            intrinsic = max(K * np.exp(-self.r * T) - self.S0 * np.exp(-self.q * T), 0)  # Исправлено

        if price < intrinsic - 1e-5:
            return np.nan
        
        if option_type == 'call' and price > self.S0 * np.exp(-self.q * T):  # Исправлено
            return np.nan
            
        if option_type == 'put' and price > K * np.exp(-self.r * T):
            return np.nan
        
        if price <= 1e-8:
            return 0.0

        def iv_obj(sigma):
            bs_price = HestonModel.black_scholes_price(self.S0, K, T, self.r, sigma, self.q, option_type)  # Добавлен q
            return bs_price - price

        try:
            low_vol, high_vol = 0.001, 5.0
            max_tries = 20
            
            while iv_obj(low_vol) * iv_obj(high_vol) > 0 and max_tries > 0:
                low_vol /= 2
                high_vol *= 2
                max_tries -= 1
            
            if iv_obj(low_vol) * iv_obj(high_vol) > 0:
                if option_type == 'call':
                    approx_vol = np.sqrt(2*np.pi/T) * price / (self.S0 * np.exp(-self.q * T))  # Исправлено
                else:
                    approx_vol = np.sqrt(2*np.pi/T) * price / (K * np.exp(-self.r * T))
                return max(0.01, min(approx_vol, 2.0))
                
            iv = brentq(iv_obj, low_vol, high_vol, xtol=1e-6, maxiter=50)
            return max(iv, 0)
        
        except (RuntimeError, ValueError):
            if option_type == 'call':
                approx_vol = np.sqrt(2*np.pi/T) * price / (self.S0 * np.exp(-self.q * T))  # Исправлено
            else:
                approx_vol = np.sqrt(2*np.pi/T) * price / (K * np.exp(-self.r * T))
            return max(0.01, min(approx_vol, 2.0))
    
    def simulate_paths(self, T: float, num_paths=10000, num_steps=100) -> tuple:  # Исправлен тип возврата
        dt = T / num_steps
        print(num_paths ,num_steps )
        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        cov = np.array([[1, self.rho], [self.rho, 1]])
        for t in range(1, num_steps + 1):
            dw = multivariate_normal.rvs(mean=[0, 0], cov=cov, size=num_paths)
            dw1 = dw[:, 0] * np.sqrt(dt)
            dw2 = dw[:, 1] * np.sqrt(dt)

            v[:, t] = np.maximum(v[:, t-1] + self.kappa * (self.theta - v[:, t-1]) * dt + self.sigma * np.sqrt(np.abs(v[:, t-1])) * dw2, 0)
            S[:, t] = S[:, t-1] * np.exp((self.r - self.q - 0.5 * v[:, t-1]) * dt + np.sqrt(np.abs(v[:, t-1])) * dw1)  # Добавлен q

        return S, v
    

class VolatilitySurface:
    """Класс для построения и интерполяции поверхности волатильности"""
    def __init__(self, strikes: np.ndarray, maturities: np.ndarray):
        """
        :param strikes: страйки
        :param maturities: сроки до экспирации (в годах)
        """
        self.strikes = strikes
        self.maturities = maturities
        self.surface = np.zeros((len(strikes), len(maturities)))
        self.interpolator = None
    
    def build_from_model(self, model: HestonModel, option_type: str = 'call', verbose: bool = True):
        """Построение поверхности из модели, по умолчанию - модели хестона"""
        total_points = len(self.strikes) * len(self.maturities)
        
        pbar = tqdm(total=total_points, desc="Building volatility surface", disable=not verbose)
        
        for i, K in enumerate(self.strikes):
            for j, T in enumerate(self.maturities):
                self.surface[i, j] = model.calculate_implied_volatility(K, T, option_type=option_type)
                pbar.update(1)
                pbar.set_postfix({
                    "strike": f"{K[0]:.1f}", 
                    "maturity": f"{T[0]:.2f}y",
                    "vol": f"{self.surface[i, j]:.4f}"
                })
        pbar.close()
        self._build_interpolator()
    
    def build_from_market_data(self, market_vols: np.ndarray):
        """Функция просто для визуализации если матрица страйков-матьюритиз задана"""
        if market_vols.shape != (len(self.strikes), len(self.maturities)):
            raise ValueError("Invalid market_vols shape")
        self.surface = market_vols
        self._build_interpolator()
    
    def _build_interpolator(self):
        """интерполяция значений"""
        # nan_mask = np.isnan(self.surface)
        # if np.any(nan_mask):
        #     avg_vol = np.nanmean(self.surface)
        #     self.surface[nan_mask] = avg_vol
        self.strikes = np.asarray(self.strikes).astype(float).flatten()
        self.maturities = np.asarray(self.maturities).astype(float).flatten()
        self.interpolator = RectBivariateSpline(self.strikes, self.maturities, self.surface, kx=3, ky=1)
    
    def get_vol(self, strike: float, maturity: float) -> float:
        """тестировачная функция для возврата волатильности в точке"""
        if self.interpolator is None:
            raise RuntimeError('Surface is not built')
        return float(self.interpolator(strike, maturity))
    
    def plot_surface(self, title="Volatility Surface"):
        """визуализация поверхности волатильности"""
        if self.interpolator is None:
            raise RuntimeError("Surface not built")
            
        K_min, K_max = min(self.strikes), max(self.strikes)
        T_min, T_max = min(self.maturities), max(self.maturities)
        
        K_grid = np.linspace(K_min, K_max, 100)
        T_grid = np.linspace(T_min, T_max, 50)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid, indexing='ij')
        
        vol_grid = self.interpolator(K_grid, T_grid)
        
        fig = go.Figure(data=[go.Surface(z=vol_grid, x=K_mesh, y=T_mesh, colorscale='Viridis')])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike',
                yaxis_title='Time to Maturity',
                zaxis_title='Volatility'
            ),
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        fig.show()


class HestonCalibrator:
    """Калибровщик модели Хестона по рыночным данным"""
    def __init__(self, S0: float, r: float, q: float = 0.0, 
                 max_iter: int = 200, tol: float = 1e-4, verbose: bool = True,
                 params_bounds: dict = None):
        self.S0 = S0
        self.r = r
        self.q = q
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.params_bounds = params_bounds or {
            'v0': (0.01, 0.5),
            'kappa': (0.1, 5.0),
            'theta': (0.01, 0.3),
            'sigma': (0.05, 0.8),
            'rho': (-0.9, 0.9)
        }
        self.best_model = None
        self.history = []
        self.loss_values = []
    
    def calibrate(self, strikes: np.ndarray, maturities: np.ndarray, 
                 market_prices: np.ndarray, option_type: str = 'call') -> HestonModel:
        """
        Калибровка модели Хестона по рыночным ценам опционов
        
        :param strikes: страйки
        :param maturities: сроки экспирации (в годах)
        :param market_prices: матрица рыночных цен опционов
        :param option_type: 'call' или 'put'
        :return: калиброванная модель Хестона
        """
        if market_prices.shape != (len(strikes), len(maturities)):
            raise ValueError("Invalid market_prices shape")
        
        x0 = np.array([0.2, 1.5, 0.2, 0.3, -0.5]) 
        bounds = [self.params_bounds['v0'],
                  self.params_bounds['kappa'],
                  self.params_bounds['theta'],
                  self.params_bounds['sigma'],
                  self.params_bounds['rho']]
        

        res = minimize(
            fun=self._loss_function,
            x0=x0,
            args=(strikes, maturities, market_prices, option_type),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'disp': self.verbose,
                'gtol': 1e-6,
                'maxls': 50 
            }
        )
        
        v0, kappa, theta, sigma, rho = res.x
        self.best_model = HestonModel(
            S0=self.S0, v0=v0, kappa=kappa, theta=theta, 
            sigma=sigma, rho=rho, r=self.r, q=self.q
        )
        
        if self.verbose:
            print("\nCalibration results:")
            print(f"Final loss: {res.fun:.6f}")
            print(f"v0 = {v0:.6f}, kappa = {kappa:.6f}, theta = {theta:.6f}")
            print(f"sigma = {sigma:.6f}, rho = {rho:.6f}")
            feller_value = 2*kappa*theta
            feller_condition = feller_value > sigma**2
            print(f"Feller condition: {feller_value:.4f} > {sigma**2:.4f} = {'Satisfied' if feller_condition else 'Violated'}")
            print(f"Optimization status: {res.message}")
            print(f"Number of iterations: {res.nit}")
            print(f"Number of function evaluations: {res.nfev}")

        return self.best_model
    def _loss_function(self, params, strikes, maturities, market_prices, option_type):
        v0, kappa, theta, sigma, rho = params
        
        if 2 * kappa * theta <= sigma**2:
            return 1e10

        try:
            model = HestonModel(
                S0=self.S0, v0=v0, kappa=kappa, theta=theta, 
                sigma=sigma, rho=rho, r=self.r, q=self.q
            )
        except Exception as e:
            if self.verbose:
                print(f"Model initialization error: {e}")
            return 1e10
        

        total_loss = 0.0
        valid_points = 0
        n_strikes = len(strikes)
        n_maturities = len(maturities)
        
        if self.verbose and len(self.history) % 5 == 0:
            pbar = tqdm(total=n_strikes * n_maturities, desc="Calculating loss", leave=False)
        
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                market_price = float(market_prices[i, j])
                if market_price <= 1e-8: 
                    if self.verbose and 'pbar' in locals():
                        pbar.update(1)
                    continue
                try:
                    model_price = model.heston_price(K, T)

                    
                    if np.isnan(model_price) or model_price <= 0:
                        if self.verbose and 'pbar' in locals():
                            pbar.update(1)
                        continue
                        
                    intrinsic = max(0, self.S0 - K) if option_type == 'call' else max(0, K - self.S0)
                    if model_price < intrinsic - 1e-5:
                        if self.verbose and 'pbar' in locals():
                            pbar.update(1)
                        continue
                        
                    if option_type == 'call' and model_price > self.S0:
                        if self.verbose and 'pbar' in locals():
                            pbar.update(1)
                        continue
                        
                    moneyness = abs(np.log(K/self.S0))
                    weight = np.exp(-moneyness**2 / 0.1)
                    
                    price_diff = model_price - market_price
                    rel_error = price_diff / market_price
                    combined_error = 0.7 * abs(price_diff) + 0.3 * abs(rel_error)
                    
                    total_loss += weight * combined_error
                    valid_points += 1
                    
                except Exception as e:
                    if self.verbose and len(self.history) < 10:
                        print(f"Error calculating price: K={K}, T={T}, Error={str(e)}")
                
                if self.verbose and 'pbar' in locals():
                    pbar.update(1)
        
        if self.verbose and 'pbar' in locals():
            pbar.close()
        
        if valid_points == 0:
            return 1e10
            
        normalized_loss = total_loss / valid_points
        
        if self.verbose and len(self.history):
            print(f"Iter {len(self.history)}: Loss = {normalized_loss.item():.6f}, Valid points: {valid_points}/{n_strikes*n_maturities}")


        self.history.append(params)
        self.loss_values.append(normalized_loss)
        return normalized_loss
