#!/usr/bin/env python3

"""
Quickly written curve fitting script for covid data.
"""

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit

BASE_DATE = datetime.datetime(2020, 3, 1)
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
    return max_cases/(1+np.exp(-(day-peak)/x_scale))


def fit_logistic_model(x_data, y_data):
    "Fits data into logistic curve"
    fit = curve_fit(logistic_model, x_data, y_data, p0=[2, 50, 100000])
    errors = np.sqrt(np.diag(fit[1]))
    peak_date = (BASE_DATE + datetime.timedelta(days=fit[0][1]))
    peak_date_error = errors[1]
    max_inf = fit[0][2]
    max_inf_error = errors[2]

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
    return np.exp(ln_daily_growth*(day-x_shift))


def fit_exponential_model(x_data, y_data):
    "Fits exponential model to data"
    expfit = curve_fit(exponential_model, x_data, y_data)
    params = expfit[0]
    errors = np.sqrt(np.diag(expfit[1]))

    return {
        'ln_daily_growth': params[0],
        'ln_daily_growth_error': errors[0],
        'daily_growth': np.exp(params[0] + errors[0]**2 / 2),
        'daily_growth_error': np.sqrt((np.exp(errors[0]**2)-1)*np.exp(2*params[0]+errors[0]**2)),
        'x_shift': params[1],
        'x_shift_error': errors[1]
    }


def create_curve_data(x_data, y_data, log_result, exp_result):
    """
    Creates the curves to be used when plotting data based
    on the calculated results.
    """
    days_to_simulate = 2*(log_result['peak_date'] - BASE_DATE).days
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
        out_log.append(logistic_model(
            day, log_result['x_scale'], log_result['peak'], log_result['max_inf']))
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
    peak_date_str = "Tetőzés a szigmoid modell alapján: " \
        "{} ± {:.2f} nap".format(
            log_result['peak_date'].strftime('%Y-%m-%d'), log_result['peak_date_error'])
    print(peak_date_str)
    max_inf_str = "Maximum a szigmoid modell alapján: {:.2f} ± {:.2f}".format(
        log_result['max_inf'], log_result['max_inf_error'])
    print(max_inf_str)

    exp_result = fit_exponential_model(x_data, y_data)
    print(exp_result)
    daily_growth_str = "Napi növekedés az exponenciális modell alapján: {:.2f}% ± {:.2}%".format(
        exp_result['daily_growth']*100-100, exp_result['daily_growth_error']*100)
    print(daily_growth_str)

    still_exp_str = "Ha még mindig exponenciális a növekedés, "\
        "holnap kb. {:.0f} új esetet kellene jelenteniük.".format(
            (exp_result['daily_growth']-1)*y_data[-1])

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
    plt.plot(curve_data['date'], curve_data['logistic'],
             'g-', label='Szigmoid modell')
    plt.plot(curve_data['date'], curve_data['exponential'],
             'b-', label='Exponenciális modell')
    plt.ylabel('Esetek')
    plt.xlabel('Dátum')
    max_log = max(curve_data['logistic'])
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   max_inf_str + "\n" +
                   peak_date_str + "\n" +
                   daily_growth_str + "\n" +
                   still_exp_str, va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(curve_data['date']), 0, max_log])
    plt.legend()
    plt.title("COVID-19 görbeillesztés {}".format(TODAY))
    plt.savefig('plot.png')


if __name__ == "__main__":
    main()
