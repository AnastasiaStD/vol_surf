import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pygam import LinearGAM, s
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from py_vollib.black_scholes.implied_volatility import implied_volatility
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Подгружает данные, считает вложенную волатильность и фильтрует данные"""
    
    def __init__(self, file_path='final_df_2.xlsx', save_plots=False, plot_dir='plots'):
        self.file_path = file_path
        self.data = None
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
    def load_and_preprocess(self):
        self.data = pd.read_excel(self.file_path)
        self._calculate_iv()
        self._filter_data()
        return self.data
    
    def _calculate_iv(self, risk_free_rate=0.21):
        
        def iv_calculator(row):
            """Расчет подразмеваемой волатильности"""
            try:
                price = row['option_price']
                S = row['asset_price']
                K = row['strike']
                tau = row['tau']
                option_type = 'c' if row['option_type'] == 'C' else 'p'
                return implied_volatility(price, S, K, tau, risk_free_rate, option_type)
            except:
                return np.nan
                
        self.data['IV'] = self.data.apply(iv_calculator, axis=1)
        self.data = self.data.dropna(subset=['IV'])
        
    def _filter_data(self):
        
        self.data = self.data[
            (self.data['option_type'] == 'C') &
            (self.data['moneyness'].between(0.9, 1.1)) & # текущая рыночная цена / страйк в таком дипазоне => опционы около денег => больше лик-ть
            (self.data['tau'].between(14/365, 1))
        ]
        
    def save_filtered_data(self, output_path='final_df_gb.xlsx'):
        
        self.data.to_excel(output_path, index=False)
        return self.data

    def plot_initial_distributions(self):

        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.hist(self.data['IV'], bins=50)
        plt.title('IV Distribution')
        plt.subplot(132)
        plt.scatter(self.data['moneyness'], self.data['IV'])
        plt.title('IV vs Moneyness')
        plt.subplot(133)
        plt.scatter(self.data['tau'], self.data['IV'])
        plt.title('IV vs Tau')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'initial_distributions.png'))
        plt.show()
        plt.close()

class PointWiseForecaster:
    """Осуществляет точечное прогнозирование по сетке"""
    
    def __init__(self, n_moneyness_bins=8, n_tau_bins=6, n_lags=5, save_plots=False, plot_dir='plots'):
        self.n_moneyness_bins = n_moneyness_bins # кол-во интервалов для дискретизации манинес
        self.n_tau_bins = n_tau_bins # кол-во интервалов для дискретизации времени до экспирации
        self.n_lags = n_lags # кол-во предыдущих значений для временного ряда
        self.model = None
        self.grid_info = {}
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
    def prepare_3d_dataset(self, df):
        """Преобразует в формат 3D сетки"""
        moneyness_bins = np.linspace(0.9, 1.1, self.n_moneyness_bins) # 8 равномерно распред на отрезке 0.9-1.1 чиселок
        tau_bins = np.linspace(0.03, 0.9, self.n_tau_bins)

        df['moneyness_bin'] = pd.cut(df['moneyness'], bins=moneyness_bins, labels=False)
        df['tau_bin'] = pd.cut(df['tau'], bins=tau_bins, labels=False)

        grid = df.pivot_table(index='trade_date',
                             columns=['moneyness_bin', 'tau_bin'], # все комбинации бинов денежности и тау
                             values='IV',
                             aggfunc='mean') # усредням значение волатильности для опционов с одинвковой комб date+moneyness+tau

        grid = grid.ffill(axis=0).bfill(axis=0) # заполняем пропуски пред значение (по времени), для краевых - затем следующим
        
        self.grid_info = {
            'moneyness_bins': moneyness_bins,
            'tau_bins': tau_bins,
            'grid_points': grid.columns.tolist()
        }
        
        return grid

    def create_lagged_features(self, grid):
        """Для каждой точки создает объекты временного ряда"""
        features, targets, dates = [], [], []
        grid_points = grid.columns.tolist()

        for i in range(self.n_lags, len(grid)):
            current_date = grid.index[i]
            for point in grid_points:
                X = grid[point].iloc[i-self.n_lags:i].values # несколько пред (по времени) значений IV для этой точки (манинес, тау) = признаки
                y = grid[point].iloc[i] # реальное значение
                if not np.isnan(X).any() and not np.isnan(y):
                    features.append(X)
                    targets.append(y)
                    dates.append(current_date)

        return np.array(features), np.array(targets), dates

    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"\n{model_name} Evaluation:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        return mae, rmse, r2

    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual"):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.3) # облако соотношения таргета и предсказания
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--') # линия идеального предсказания
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        
        if self.save_plots:
            filename = title.lower().replace(' ', '_').replace(':', '') + '.png'
            plt.savefig(os.path.join(self.plot_dir, filename))
        plt.show()
        plt.close()

    def train_model(self, grid):
        X, y, dates = self.create_lagged_features(grid)

        # используем град бустинг
        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            early_stopping=True
        )

        tscv = TimeSeriesSplit(n_splits=3) # разделитель для кросс-валидации (с учетом временной структуры)
        # сначала первые 25% данных в трейн, следующие 25% в тест; потом первые 50% в трейн, 25% в тест, далее 75% и 25% (три итерации)
        r2_scores, mae_scores, rmse_scores = [], [], []

        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])

            r2_scores.append(r2_score(y[test_idx], pred)) # получится в итоге список из трех значений
            mae_scores.append(mean_absolute_error(y[test_idx], pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y[test_idx], pred)))

        print("\nCross-Validation Results:")
        print(f"Average R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
        print(f"Average MAE: {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")

        model.fit(X, y)
        full_pred = model.predict(X)
        self.evaluate_model(y, full_pred, "Final Model")
        self.plot_predictions(y, full_pred, "Final Model Predictions vs Actual")
        
        self.model = model
        return model, (r2_scores, mae_scores, rmse_scores)

    def predict_next_day(self, grid): # используем обученную модель, исторические данные и инфу о сетке
        """Точечный прогноз поверхности (поточечно!) на следующий день (в данных этого дня нет)"""
        predictions = {}
        for point in self.grid_info['grid_points']:
            X_pred = grid[point].iloc[-self.n_lags:].values.reshape(1, -1) # берем точку, берем последнии (по времени) значения IV в ней
            predictions[point] = self.model.predict(X_pred)[0] # делаем предсказание для этой точки 

        # создаем структуру поверхности
        surface = pd.DataFrame(
            index=[f"{self.grid_info['moneyness_bins'][i]:.3f}-{self.grid_info['moneyness_bins'][i+1]:.3f}"
                   for i in range(len(self.grid_info['moneyness_bins'])-1)],
            columns=[f"{self.grid_info['tau_bins'][i]:.3f}-{self.grid_info['tau_bins'][i+1]:.3f}"
                    for i in range(len(self.grid_info['tau_bins'])-1)]
        )

        # заполняем предсказаниями
        for (m_bin, t_bin), value in predictions.items():
            surface.iloc[m_bin, t_bin] = value

        return surface

    def run_full_analysis(self, df):
        grid = self.prepare_3d_dataset(df)
        model, scores = self.train_model(grid)
        next_day_pred = self.predict_next_day(grid)
        
        print("\nPredicted IV Surface for Next Day:")
        print(next_day_pred)
        
        return {
            'model': model,
            'scores': scores,
            'prediction': next_day_pred,
            'grid_info': self.grid_info
        }

class HybridSurfaceForecaster:
    """
    Сочетает точечное прогнозирование c GAM сглаживанием
    
    Для каждого предикта (у нас это тау и манинесс) строиться своя гладкая функция.
    Она описывает их связь с IV. Функции строятся с помощью  cглаживающих сплайнов (s_0 и s_1).
    Далее мы исходим из предположения, что эффекты moneyness и tau на IV не взаимодействуют друг с другом.
    Они аддитивны (общий эффект - сумма двух компонент). То есть мы можем их просто сложить.
    """
    
    def __init__(self, n_moneyness_bins=8, n_tau_bins=5, n_splines=5, n_lags=5, save_plots=False, plot_dir='plots'):
        self.n_moneyness_bins = n_moneyness_bins 
        self.n_tau_bins = n_tau_bins 
        self.n_splines = n_splines
        self.n_lags = n_lags 
        self.point_models = {}
        self.scalers = {}
        self.gam_models = {}
        self.grid_info = {}
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    def prepare_data(self, df):
        moneyness_bins = np.linspace(0.9, 1.1, self.n_moneyness_bins)
        tau_bins = np.linspace(0.03, 0.9, self.n_tau_bins)

        df['moneyness_bin'] = np.digitize(df['moneyness'], moneyness_bins) - 1
        df['tau_bin'] = np.digitize(df['tau'], tau_bins) - 1

        grid = df.pivot_table(index='trade_date',
                             columns=['moneyness_bin', 'tau_bin'],
                             values='IV',
                             aggfunc='mean')

        self.grid_info = {
            'moneyness_bins': moneyness_bins,
            'tau_bins': tau_bins,
            'grid_points': grid.columns.tolist()
        }

        return grid.ffill().bfill()

    def train_models(self, grid):
        """Обучение точечных моделей (для каждой точки сетки создается свой временной ряд IV)"""
        for point in tqdm(self.grid_info['grid_points'], desc="Training point models"):
            series = grid[point].values.reshape(-1, 1)
            X, y = [], []
            
            for i in range(self.n_lags, len(series)):
                X.append(series[i-self.n_lags:i].flatten())
                y.append(series[i][0])
                
            X, y = np.array(X), np.array(y)

            if len(X) < 10:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[point] = scaler

            model = HistGradientBoostingRegressor(
                max_iter=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(X_scaled, y)
            self.point_models[point] = model

    def fit_gam_surfaces(self, df):
        """Для каждого дня моделируем поверхность с помощью GAM"""
        for date, group in df.groupby('trade_date'):
            if len(group) < 10:
                continue

            gam = LinearGAM(s(0, n_splines=self.n_splines) + s(1, n_splines=self.n_splines))
            
            try: 
                gam.gridsearch(group[['moneyness', 'tau']].values, 
                             group['IV'].values,
                             lam=np.logspace(-3, 3, 5))
                self.gam_models[date] = gam
            except:
                continue

    def predict_surface(self, grid): # используем grid в которм содержится историческая инфа о волатильности
        """Строит поверхность для предсказаний с помощью GAM"""
        predictions = {}
        for point in self.grid_info['grid_points']:
            if point in self.point_models: # проверяем есть ли обученная модель для этой точки
                X = grid[point].iloc[-self.n_lags:].values.reshape(1, -1)
                X_scaled = self.scalers[point].transform(X)
                predictions[point] = self.point_models[point].predict(X_scaled)[0]

        XX, YY, ZZ = [], [], []
        for (m_bin, t_bin), iv in predictions.items():
            moneyness = np.mean(self.grid_info['moneyness_bins'][m_bin:m_bin+2])
            tau = np.mean(self.grid_info['tau_bins'][t_bin:t_bin+2])
            XX.append(moneyness)
            YY.append(tau)
            ZZ.append(iv)

        gam = LinearGAM(s(0, n_splines=self.n_splines) + s(1, n_splines=self.n_splines))
        gam.fit(np.column_stack([XX, YY]), ZZ)

        return gam, predictions

    def _plot_surface(self, date, df, title, points=None):
        gam = self.gam_models[date]
        moneyness = np.linspace(0.9, 1.1, 50)
        tau = np.linspace(0.03, 0.9, 50)
        XX, YY = np.meshgrid(moneyness, tau)
        ZZ = gam.predict(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

        fig = go.Figure()
        fig.add_trace(go.Surface(x=XX, y=YY, z=ZZ, colorscale='Viridis'))

        if points:
            xp, yp, zp = zip(*[
                (np.mean(self.grid_info['moneyness_bins'][m:m+2]),
                 np.mean(self.grid_info['tau_bins'][t:t+2]),
                 iv)
                for (m,t), iv in points.items()
            ])
            fig.add_trace(go.Scatter3d(
                x=xp, y=yp, z=zp,
                mode='markers',
                marker=dict(color='red', size=5)
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Moneyness',
                yaxis_title='Tau',
                zaxis_title='Implied Volatility'
            ),
            width=800,
            height=600
        )
        
        if self.save_plots:
            filename = title.lower().replace(' ', '_').replace(':', '') + '.html'
            fig.write_html(os.path.join(self.plot_dir, filename))
        fig.show()

    def generate_metrics(self):
        historical_r2 = [
            self.gam_models[d].statistics_['pseudo_r2']['explained_deviance']
            for d in self.gam_models
            if d != "predicted"
        ]

        metrics = pd.DataFrame({
            'R²': historical_r2,
            'Type': 'Historical'
        })

        summary = pd.DataFrame({
            'Metric': ['Best', 'Worst', 'Mean', 'Median'],
            'R²': [
                max(historical_r2),
                min(historical_r2),
                np.mean(historical_r2),
                np.median(historical_r2)
            ]
        })

        return metrics, summary

    def visualize_results(self, df):
        best_date = max(self.gam_models,
                       key=lambda d: self.gam_models[d].statistics_['pseudo_r2']['explained_deviance'])
        self._plot_surface(best_date, df, "Best Historical Fit")

        grid = self.prepare_data(df)
        pred_gam, pred_points = self.predict_surface(grid)
        self.gam_models["predicted"] = pred_gam
        self._plot_surface("predicted", df, "Predicted Surface", pred_points)

    def evaluate_forecast_quality(self, df):
        df = df.sort_values('trade_date')
        
        # Train/test split (80%/20%)
        train_size = int(len(df) * 0.8)
        train = df.iloc[:train_size]
        test = df.iloc[train_size:]
        
        metrics = []
        print(f"\nStarting walk-forward validation on {len(test['trade_date'].unique())} test days...")
        

        # для каждой даты в тесте:
        for i, test_date in enumerate(tqdm(test['trade_date'].unique(), desc="Evaluating")):
            historical_data = df[df['trade_date'] < test_date]
            
            self.point_models = {}
            self.scalers = {}
            self.gam_models = {}
            
            grid = self.prepare_data(historical_data) 
            self.train_models(grid) # обучаем поточеченые модели на трейне
            self.fit_gam_surfaces(historical_data) # для каждого дня трейна создаем свою GAM
            
            predicted_gam, _ = self.predict_surface(grid) # получаем поверхность с помощтью GAM
            
            actual_data = test[test['trade_date'] == test_date]
            
            if len(actual_data) == 0:
                continue

            X_test = actual_data[['moneyness', 'tau']].values
            y_pred = predicted_gam.predict(X_test)
            y_true = actual_data['IV'].values

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            metrics.append({
                'date': test_date,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'n_observations': len(actual_data)
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.plot(metrics_df['date'], metrics_df['R2'])
        plt.title('R² over Time')
        plt.xticks(rotation=45)
        
        plt.subplot(132)
        plt.plot(metrics_df['date'], metrics_df['RMSE'])
        plt.title('RMSE over Time')
        plt.xticks(rotation=45)
        
        plt.subplot(133)
        plt.plot(metrics_df['date'], metrics_df['MAE'])
        plt.title('MAE over Time')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'forecast_metrics_over_time.png'))
        plt.show()
        plt.close()
        
        return metrics_df

    def run_full_analysis(self, df):
        """Complete analysis pipeline"""
        grid = self.prepare_data(df)
        self.train_models(grid)
        self.fit_gam_surfaces(df)
        
        metrics_df, summary_df = self.generate_metrics()
        print("\nHistorical Model Performance:")
        print(metrics_df.describe())
        print("\nKey Statistics:")
        print(summary_df)
        
        self.visualize_results(df)

        forecast_metrics = self.evaluate_forecast_quality(df)
        
        return {
            'point_models': self.point_models,
            'gam_models': self.gam_models,
            'historical_metrics': metrics_df,
            'forecast_metrics': forecast_metrics,
            'summary_stats': summary_df
        }

class GAMCoefficientForecaster:
    """Прогнозирует поверхность, предсказывая коэффициенты GAM"""
    
    def __init__(self, n_splines=5, n_lags=5, save_plots=False, plot_dir='plots'):
        self.n_splines = n_splines
        self.n_lags = n_lags
        self.gam_models = {}
        self.alpha_dict = {}
        self.gam_metrics = {}
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
    def fit_gam_for_day(self, group):
        """Подгоняет GAM под данные за конкретный день (лежат в group)"""
        try:
            X = group[['moneyness', 'tau']].values 
            y = group['log_IV'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            gam = LinearGAM(s(0, n_splines=self.n_splines) + s(1, n_splines=self.n_splines))
            gam.gridsearch(X_train, y_train, lam=np.logspace(-3, 3, 10))

            y_pred = gam.predict(X_test) # смотрим насколько хоршо по тау и манинес удается восстановить IV c помощью GAM
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pseudo_r2 = gam.statistics_['pseudo_r2']['explained_deviance']

            return gam, {'mse': mse, 'r2': r2, 'pseudo_r2': pseudo_r2}
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return None, None
            
    def prepare_lagged_features(self, alpha_df, lags=5):
        """Создаем обычаующую выборку (предыдущие значения параметров) и таргет (текущие значение параметров)"""
        X, y, dates = [], [], []
        for i in range(lags, len(alpha_df)): # range(5, 64) => для 59 дней можем сделать (если лаг=5)
            lag_data = alpha_df.iloc[i-lags:i].values.flatten() # то есть для 59 дней загоняем все предыдущие значения всех коэффов в строку 
            X.append(lag_data) #(59х55)
            y.append(alpha_df.iloc[i].values) #(59х11) - текущие значения каждого коэфа для каждого дня 
            dates.append(alpha_df.index[i])
        return np.array(X), np.array(y), dates

    def train_coefficient_models(self, alpha_df):
        X, y, dates = self.prepare_lagged_features(alpha_df, lags=self.n_lags)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        coef_metrics = []

        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            models = {}
            for i in range(y.shape[1]):
                gbrt = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_split=5,
                    random_state=42
                )
                gbrt.fit(X_train, y_train[:, i])
                models[i] = gbrt

                y_pred = gbrt.predict(X_test)
                mse = mean_squared_error(y_test[:, i], y_pred)
                r2 = r2_score(y_test[:, i], y_pred)
                coef_metrics.append({'coef': i, 'mse': mse, 'r2': r2})

        coef_metrics_df = pd.DataFrame(coef_metrics)
        print("\nCoefficient Prediction Quality:")
        print(coef_metrics_df.groupby('coef').mean())
        
        return models, coef_metrics_df

    def plot_approximation_quality(self, metrics_df):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.hist(metrics_df['r2'], bins=20)
        plt.title('R² Distribution')
        plt.subplot(132)
        plt.hist(metrics_df['pseudo_r2'], bins=20)
        plt.title('Pseudo-R² Distribution')
        plt.subplot(133)
        plt.scatter(metrics_df['r2'], metrics_df['pseudo_r2'])
        plt.xlabel('R²')
        plt.ylabel('Pseudo-R²')
        plt.title('Metrics Comparison')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'gam_approximation_quality.png'))
        plt.show()
        plt.close()

    def _create_merged_data_2(self, df, metrics_df):
        return df[df.trade_date.isin(metrics_df[metrics_df['r2'] > 0].index)]

    def run_full_analysis(self, df, use_filtered_data=True):
        df['log_IV'] = np.log(df['IV'])
        df = df.dropna(subset=['moneyness', 'tau', 'log_IV'])
        
        print("\nInitial GAM fitting...")
        for date, group in df.groupby('trade_date'): # для каждой даты подгоняем свою аппроксимацию 
            if len(group) < 10:
                continue
            gam, metrics = self.fit_gam_for_day(group)
            if gam is not None and metrics is not None:
                self.gam_models[date] = gam
                self.alpha_dict[date] = np.hstack([gam.coef_]) # получим свободный член + 5 коэф для манинес + 5 коэф для тау = 11
                self.gam_metrics[date] = metrics

        metrics_df = pd.DataFrame.from_dict(self.gam_metrics, orient='index')
        print("\nInitial Surface Approximation Quality:")
        print(metrics_df.describe()) # смотрим среднее качество подгонки для всех дат

        if use_filtered_data: # удаляем даты для которых не удалось сделать хорошую подгонку
            merged_data_2 = self._create_merged_data_2(df, metrics_df)
            print(f"\nFiltered dataset contains {len(merged_data_2)} records "
                  f"(original had {len(df)})")
            
            print("\nRe-running with filtered data...")
            self.gam_models = {}
            self.alpha_dict = {}
            self.gam_metrics = {}
            
            for date, group in merged_data_2.groupby('trade_date'):
                if len(group) < 10:
                    continue
                gam, metrics = self.fit_gam_for_day(group)
                if gam is not None and metrics is not None:
                    self.gam_models[date] = gam
                    self.alpha_dict[date] = np.hstack([gam.coef_])
                    self.gam_metrics[date] = metrics

            metrics_df = pd.DataFrame.from_dict(self.gam_metrics, orient='index')
            print("\nFiltered Surface Approximation Quality:")
            print(metrics_df.describe())
            df = merged_data_2

        alpha_df = pd.DataFrame.from_dict(self.alpha_dict, orient='index') # сохраняем коэффы, они описывают поверхность
        alpha_df = alpha_df.sort_index()
        alpha_df.to_excel('ALPHA.xlsx', index=False)

        models, coef_metrics_df = self.train_coefficient_models(alpha_df)
        
        self.plot_approximation_quality(metrics_df)
        
        return {
            'gam_models': self.gam_models,
            'coefficient_models': models,
            'approximation_metrics': metrics_df,
            'coefficient_metrics': coef_metrics_df,
            'filtered_data': df if use_filtered_data else None
        }