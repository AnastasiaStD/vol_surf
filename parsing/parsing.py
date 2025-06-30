import apimoex
import requests

import string

import pandas as pd
import numpy as np

import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from scipy.optimize import minimize

from datetime import datetime
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def fetch_option_data(session, secid, start='1900-01-01', end='2100-01-01'):
    """
    Получает метаданные и исторические данные для одного опциона.
    """
    try:
        ticker = apimoex.find_security_description(session, secid)
        end_day = apimoex.get_board_history(session, secid, board='ROPD', engine='futures', market='options', 
                                           columns=None, start=start, end=end)
        return ticker, end_day, secid
    except Exception as e:
        print(f"Ошибка при обработке опциона {secid}: {e}")
        return None, None, secid


def option_data_parser(base_active, strikes_str, opt_type, months, years, weeks_flg=['', 'A', 'B', 'C', 'D', 'E']):

        
    '''
    
    Парсит данные об опционах с использованием API Московской биржи (MOEX).

    Функция собирает метаданные опционов, их исторические данные и свечные графики,
    а затем сохраняет эти данные в два отдельных DataFrame: `ticket_info` и `end_day_info`.

    Параметры:
    ----------
    base_active : str
        Базовый актив опциона (например, 'Si', 'RTS').
    strikes_str : list
        Список страйков опционов (например, ['75', '76', '77']).
    opt_type : str
        Тип опциона (например, 'C' - европейский).
    months : list
        Список месяцев экспирации опционов (call: A-L, put: M-X).
    years : list
        Список годов экспирации опционов (например, ['3', '4'] для 2023 и 2024 годов).
    weeks_flg : list, optional
        Список флагов недель для опционов (по умолчанию - все).

    Возвращает:
    -----------
    ticket_info : pd.DataFrame
        DataFrame с метаданными опционов (например, название, тип, дата исполнения и т.д.).
    end_day_info : pd.DataFrame
        DataFrame с историческими данными по опционам (например, цены, объемы и т.д.).

    Пример использования:
    ---------------------
    base_active = 'Si'
    strikes_str = ['75', '76']
    opt_type = 'C'
    months = ['1', '2']
    years = ['3', '4']

    ticket_info, end_day_info = option_data_parser(base_active, strikes_str, opt_type, months, years)
    
    '''
    
    ticket_title = ['Краткий код', 'Наименование серии инструмента', 'Краткое наименование контракта', 'Последний день обращения', 
                    'Тип опциона', 'Дата исполнения', 'Код базового актива', 'Начало обращения', 'Вид опциона пут или колл', 'Цена Страйк', 
                    'Исполнение', 'Способ исполнения опциона', 'Способ маржирования опциона', 'Лот', 'Наименование контракта', 
                    'Группа контрактов', 'Котировка', 'Базовый актив', 'Клиринг исполнения', 'Опционная серия', 'Вид контракта', 
                    'Код типа инструмента', 'Тип бумаги', 'Типа инструмента']
    
    columns_end_of_date = ['NAME', 'BOARDID', 'TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE',
                       'OPENPOSITIONVALUE', 'VALUE', 'VOLUME', 'OPENPOSITION', 'SETTLEPRICE',
                       'WAPRICE', 'SETTLEPRICEDAY', 'CHANGE', 'QTY', 'NUMTRADES']

    RETRY_DELAY = 5 # Задержка в секундах между попытками

    ticket_info = pd.DataFrame(columns=ticket_title)
    end_day_info = pd.DataFrame(columns=columns_end_of_date)

    for year in years:
        time.sleep(0.4)
        print(f'202{year} year is being parsed')
        for month in months:
            print(f'- month: {month}')
            for week in weeks_flg:
                for strike in tqdm(strikes_str, desc=f'Week: {"-" if week == "" else week}'):
                    try:
                        with requests.Session() as session:

                            secid = f'{base_active}{strike}C{month}{year}{week}'

                            ticker, end_day, _  = fetch_option_data(session, secid)
                            
                            if ticker != [] and end_day != []:
                                print(f'{base_active}{strike}C{month}{year}{week}')

                                columns = pd.DataFrame(ticker)['title'].to_list()
                                values = pd.DataFrame(ticker)['value'].to_list()
                                ticket_new = pd.DataFrame([values], columns=columns)
                                ticket_info = pd.concat([ticket_info, ticket_new], ignore_index=True)

                                end_day = pd.DataFrame(end_day)
                                end_day.insert(0, 'NAME', f'{base_active}{strike}C{month}{year}{week}') 
                                end_day_info = end_day_info._append(end_day)
    
                                print('Saved.')
                    except:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        print(f"Произошла ошибка типа: {exc_type.__name__}")

    return ticket_info, end_day_info


def fast_option_data_parser(base_active, strikes_str, opt_type, months, years, weeks_flg=['', 'A', 'B', 'C', 'D', 'E']):
    """
    Парсит данные об опционах с использованием API Московской биржи (MOEX).

    Аналогично option_data_parser, только с параллельным выполнением задач.
    """
    
    ticket_title = ['Краткий код', 'Наименование серии инструмента', 'Краткое наименование контракта', 'Последний день обращения', 
                    'Тип опциона', 'Дата исполнения', 'Код базового актива', 'Начало обращения', 'Вид опциона пут или колл', 'Цена Страйк', 
                    'Исполнение', 'Способ исполнения опциона', 'Способ маржирования опциона', 'Лот', 'Наименование контракта', 
                    'Группа контрактов', 'Котировка', 'Базовый актив', 'Клиринг исполнения', 'Опционная серия', 'Вид контракта', 
                    'Код типа инструмента', 'Тип бумаги', 'Типа инструмента']
    
    columns_end_of_date = ['NAME', 'BOARDID', 'TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE',
                       'OPENPOSITIONVALUE', 'VALUE', 'VOLUME', 'OPENPOSITION', 'SETTLEPRICE',
                       'WAPRICE', 'SETTLEPRICEDAY', 'CHANGE', 'QTY', 'NUMTRADES']

    ticket_info = pd.DataFrame(columns=ticket_title)
    end_day_info = pd.DataFrame(columns=columns_end_of_date)

    with requests.Session() as session:
        options_to_process = []
        for year in years:
            for month in months:
                for week in weeks_flg:
                    for strike in strikes_str:
                        secid = f'{base_active}{strike}{opt_type}{month}{year}{week}'
                        options_to_process.append(secid)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_option_data, session, secid) for secid in options_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing options"):
                ticker, end_day, secid = future.result()
                if ticker and end_day:
                    columns = pd.DataFrame(ticker)['title'].to_list()
                    values = pd.DataFrame(ticker)['value'].to_list()
                    ticket_new = pd.DataFrame([values], columns=columns)
                    ticket_info = pd.concat([ticket_info, ticket_new], ignore_index=True)
                    
                    end_day = pd.DataFrame(end_day)
                    end_day.insert(0, 'NAME', secid)
                    end_day_info = end_day_info._append(end_day)

                    print(f'Saved {secid}')

    return ticket_info, end_day_info

def optimizid_option_data_parser(base_active, strikes_str, opt_type, months, years, weeks_flg=['', 'A', 'B', 'C', 'D', 'E'], 
                                parallel=True):
    """
    Оптимизированный парсер данных об опционах с использованием API Московской биржи (MOEX). Уменьшенное число обращений к API засчет
    выгрузи данных о всех торгуемых опционах и их фильтрации.
    Можно выбрать параллелить задачи или нет (по умолчанию parallel=True)

    """
    
    ticket_title = ['Краткий код', 'Наименование серии инструмента', 'Краткое наименование контракта', 'Последний день обращения', 
                    'Тип опциона', 'Дата исполнения', 'Код базового актива', 'Начало обращения', 'Вид опциона пут или колл', 'Цена Страйк', 
                    'Исполнение', 'Способ исполнения опциона', 'Способ маржирования опциона', 'Лот', 'Наименование контракта', 
                    'Группа контрактов', 'Котировка', 'Базовый актив', 'Клиринг исполнения', 'Опционная серия', 'Вид контракта', 
                    'Код типа инструмента', 'Тип бумаги', 'Типа инструмента']
    
    columns_end_of_date = ['NAME', 'BOARDID', 'TRADEDATE', 'SECID', 'OPEN', 'LOW', 'HIGH', 'CLOSE',
                       'OPENPOSITIONVALUE', 'VALUE', 'VOLUME', 'OPENPOSITION', 'SETTLEPRICE',
                       'WAPRICE', 'SETTLEPRICEDAY', 'CHANGE', 'QTY', 'NUMTRADES']

    ticket_info = pd.DataFrame(columns=ticket_title)
    end_day_info = pd.DataFrame(columns=columns_end_of_date)

    with requests.Session() as session:
        # Получаем список всех опционов на базовый актив
        all_options = apimoex.get_board_securities(session, market='options', board='ROPD', engine='futures')
        all_options_df = pd.DataFrame(all_options)

        # Фильтруем опционы по базовому активу, месяцу, году и тому что опцион - европейский
        pattern = re.compile(rf'^{base_active}(\d+)({opt_type})')
        filtered_options = all_options_df[
            (all_options_df['SECID'].str.extract(pattern)[1].notna()) &
            (all_options_df['SECID'].str[-1].isin(years + weeks_flg)) &
            (all_options_df['SECID'].str[-2].isin(years + list(months)))
        ]

        if parallel:
            # Используем ThreadPoolExecutor для параллельной обработки
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_option_data, session, secid) for secid in filtered_options['SECID']]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing options"):
                    ticker, end_day, secid = future.result()
                    if ticker and end_day:
                        columns = pd.DataFrame(ticker)['title'].to_list()
                        values = pd.DataFrame(ticker)['value'].to_list()
                        ticket_new = pd.DataFrame([values], columns=columns)
                        ticket_info = pd.concat([ticket_info, ticket_new], ignore_index=True)

                        end_day = pd.DataFrame(end_day)
                        end_day.insert(0, 'NAME', secid)
                        end_day_info = end_day_info._append(end_day)

                        print(f'Saved {secid}')
        else:
            for secid in tqdm(filtered_options['SECID'], desc="Processing options"):
                ticker, end_day, secid = fetch_option_data(session, secid)
                if ticker and end_day:
                    columns = pd.DataFrame(ticker)['title'].to_list()
                    values = pd.DataFrame(ticker)['value'].to_list()
                    ticket_new = pd.DataFrame([values], columns=columns)
                    ticket_info = pd.concat([ticket_info, ticket_new], ignore_index=True)

                    end_day = pd.DataFrame(end_day)
                    end_day.insert(0, 'NAME', secid)
                    end_day_info = end_day_info._append(end_day)

                    print(f'Saved {secid}')
                    
    return ticket_info, end_day_info

def get_asset_price(asset_code, start_date, end_date=None, board='TQBR'):
    '''
    Получает стоимость актива на указанную дату или за период.

    Параметры:
    ----------
    asset_code : str
        Код ценной бумаги (например, 'SBER' для акций Сбербанка).
    start_date : str
        Начальная дата в формате 'YYYY-MM-DD'.
    end_date : str, optional
        Конечная дата в формате 'YYYY-MM-DD'. Если не указана, возвращает данные только на start_date.
    board : str, optional
        Режим торгов (по умолчанию 'TQBR' — основной режим торгов акциями).

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с данными по стоимости актива на указанную дату или за период.
    '''
    
    with requests.Session() as session:
        try:
            data = apimoex.get_board_history(session, 
                                             security=asset_code, 
                                             board=board, 
                                             start=start_date, 
                                             end=end_date if end_date else start_date, 
                                             columns=('TRADEDATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME'))
    
            df = pd.DataFrame(data)
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            df = df.sort_values(by='TRADEDATE')
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"Произошла ошибка типа: {exc_type.__name__}")

    return df


