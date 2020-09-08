#!/usr/bin/env python3

"""
Quickly written curve fitting script for covid data.
"""

import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit

BASE_DATE = datetime.datetime(2020, 8, 20)
Y_BASE = 5046
TODAY = datetime.datetime.now().strftime('%Y-%m-%d')


def parse_covid_data(filename):
    "Loads the date from file into two lists separate x and y list"
    with open(filename) as file:
        content = file.readlines()

    x_data = []
    y_data = []

    for line in content:
        fields = line.split()
        if len(fields) < 2:
            continue
        date = (datetime.datetime.strptime(fields[0], '%Y-%m-%d')
                - BASE_DATE).days
        number = int(fields[1])
        x_data.append(date)
        y_data.append(number)

    return x_data, y_data


def logistic_model(day, x_scale, peak, max_cases):
    "Logistic model formula"
    return max_cases/(1+np.exp(-(day-peak)/x_scale)) + Y_BASE


def fit_logistic_model(x_data, y_data):
    return None
    "Fits data into logistic curve"
    fit = curve_fit(logistic_model, x_data, y_data, p0=[2, 30, 1000000])
    errors = np.sqrt(np.diag(fit[1]))
    peak_date = (BASE_DATE + datetime.timedelta(days=fit[0][1]))
    peak_date_error = errors[1]
    max_inf = fit[0][2]
    max_inf_error = errors[2]

    if peak_date_error == float("inf") or max_inf_error == float("inf"):
        return None

    return {
        'peak': fit[0][1],
        'peak_date': peak_date,
        'peak_date_error': peak_date_error,
        'max_inf': max_inf,
        'max_inf_error': max_inf_error,
        'x_scale': fit[0][0],
        'x_scale_error': errors[0]
    }


def exponential_model(day, ln_daily_growth, x_shift):
    "Exponential model formula"
    return np.exp(ln_daily_growth*(day-x_shift)) + Y_BASE


def fit_exponential_model(x_data, y_data):
    "Fits exponential model to data"
    expfit = curve_fit(exponential_model, x_data, y_data)
    params = expfit[0]
    errors = np.sqrt(np.diag(expfit[1]))

    return {
        'ln_daily_growth': params[0],
        'ln_daily_growth_error': errors[0],
        'daily_growth': np.exp(params[0] + errors[0]**2 / 2),
        'raw_daily_growth': np.exp(params[0]),
        'daily_growth_error': np.sqrt((np.exp(errors[0]**2)-1)*np.exp(2*params[0]+errors[0]**2)),
        'x_shift': params[1],
        'x_shift_error': errors[1]
    }


def create_curve_data(x_data, y_data, log_result, exp_result):
    """
    Creates the curves to be used when plotting data based
    on the calculated results.
    """
    if log_result is None:
        days_to_simulate = 2*(x_data[-1] - x_data[0] + 1)
    else:
        days_to_simulate = max(2*(log_result['peak_date'] - BASE_DATE).days, x_data[-1] - x_data[0] + 1)
    i = 0
    day_base = x_data[0]
    out_date = []
    out_y = []
    out_log = []
    out_exp = []
    while i < days_to_simulate:
        day = day_base + i
        out_date.append(BASE_DATE + datetime.timedelta(days=day))
        out_y.append(y_data[i] if i < len(y_data) else float('nan'))
        if not log_result is None:
            out_log.append(logistic_model(
                day, log_result['x_scale'], log_result['peak'], log_result['max_inf']))
        else:
            out_log.append(0)
        out_exp.append(exponential_model(
            day, exp_result['ln_daily_growth'], exp_result['x_shift']))
        i = i + 1

    return {
        'date': out_date,
        'y': out_y,
        'logistic': out_log,
        'exponential': out_exp
    }


def main():
    "Entry point"

    x_data, y_data = parse_covid_data('covid_data.txt')

    log_result = fit_logistic_model(x_data, y_data)
    if not log_result is None:
        peak_date_str = "Tetőzés a szigmoid modell alapján: " \
            "{} ± {:.2f} nap".format(
                log_result['peak_date'].strftime('%Y-%m-%d'), log_result['peak_date_error'])
        print(peak_date_str)
        max_inf_str = "Maximum a szigmoid modell alapján: {:.2f} ± {:.2f} eset".format(
            log_result['max_inf'] + Y_BASE, log_result['max_inf_error'])
        print(max_inf_str)
    else:
        peak_date_str = "Szigmoid modell nem illeszkedik az adatokra."
        max_inf_str = ""
        print("Logistic curve is too bad fit for current data")

    exp_result = fit_exponential_model(x_data, y_data)
    print(exp_result)
    daily_growth_str = "Napi növekedés az exponenciális modell alapján: {:.2f}% ± {:.2}%".format(
        exp_result['daily_growth']*100-100, exp_result['daily_growth_error']*100)
    print(daily_growth_str)
    print("ln daily growth: {}, x_shift: {}".format(exp_result["ln_daily_growth"], exp_result["x_shift"]))

    still_exp_str = "Ha még mindig exponenciális a növekedés, "\
        "holnap kb. {:.0f} új esetet kellene jelenteniük legalább.".format(
            (exp_result['daily_growth']-1)*(y_data[-1]-Y_BASE))

    curve_data = create_curve_data(x_data, y_data, log_result, exp_result)

    print("Date\tActual\tPredicted log\tPredicted exp")
    for i in range(0, len(curve_data['date'])):
        print("{}\t{}\t{}\t{}".format(
            curve_data['date'][i],
            curve_data['y'][i],
            curve_data['logistic'][i],
            curve_data['exponential'][i]
        ))

    axes = plt.gca()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    axes.xaxis.set_minor_locator(mdates.DayLocator())

    plt.figure(figsize=[10.24, 7.68])
    plt.plot(curve_data['date'], curve_data['y'],
             'ro', label='Jelentett esetek')
    if not log_result is None:
        plt.plot(curve_data['date'], curve_data['logistic'],
                 'g-', label='Szigmoid modell')
    plt.plot(curve_data['date'], curve_data['exponential'],
             'b-', label='Exponenciális modell')
    plt.ylabel('Összes eset')
    plt.xlabel('Dátum')
    if log_result is None:
        max_y = 2*max(y_data)
    else:
        max_y = max(curve_data['logistic'] + y_data)
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   max_inf_str + "\n" +
                   peak_date_str + "\n" +
                   daily_growth_str + "\n" +
                   still_exp_str, va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(curve_data['date']), Y_BASE, max_y])
    plt.legend()
    plt.title("COVID-19 görbeillesztés {}".format(TODAY))
    plt.savefig('plot.png')


if __name__ == "__main__":
    main()
