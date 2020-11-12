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
import sys
import math

BASE_DATE = ''
Y_BASE = ''

TODAY = datetime.datetime.now().strftime('%Y-%m-%d')
LAST_DATE = ''

def parse_covid_data(filename):
    "Loads the date from file into two lists separate x and y list"
    with open(filename) as file:
        content = file.readlines()

    x_data = []
    y_data = []
    global LAST_DATE
    global BASE_DATE

    for line in content:
        fields = line.split()
        if len(fields) < 2:
            continue
        parsed_date = datetime.datetime.strptime(fields[0], '%Y-%m-%d')
        if BASE_DATE == '':
            BASE_DATE = parsed_date
        LAST_DATE = fields[0]
        date = (parsed_date - BASE_DATE).days
        number = float(fields[1])
        x_data.append(date)
        y_data.append(number)

    return x_data, y_data


def logistic_model(day, x_scale, peak, max_cases):
    "Logistic model formula"
    return max_cases/(1+np.exp(-(day-peak)/x_scale)) + Y_BASE


def fit_logistic_model(x_data, y_data):
    "Fits data into logistic curve"
    try:
        sigma = [1] * len(y_data)
        # sigma[-1] = 0.1
        popt, pcov = curve_fit(logistic_model, x_data, y_data, p0=[2, 60, 100000], sigma = sigma)
        errors = np.sqrt(np.diag(pcov))
        peak_date = (BASE_DATE + datetime.timedelta(days=popt[1]))
        peak_date_error = errors[1]
        max_inf = popt[2]
        max_inf_error = errors[2]

        if peak_date_error > 1e7 or max_inf_error > 1e7:
            print("No sigmoid fit due to too large covariance.")
            return None

        if max_inf_error > max_inf:
            print("No sigmoid fit because the uncertainty of the maximum is larger than the maximum itself.")
            return None

        return {
            'peak': popt[1],
            'peak_date': peak_date,
            'peak_date_error': peak_date_error,
            'peak_growth': logistic_model(popt[1]+1, popt[0], popt[1], popt[2]) - logistic_model(popt[1], popt[0], popt[1], popt[2]),
            'tomorrow_growth': logistic_model(x_data[-1]+1, popt[0], popt[1], popt[2]) - y_data[-1],
            'max_inf': max_inf,
            'max_inf_error': max_inf_error,
            'x_scale': popt[0],
            'x_scale_error': errors[0],
            'popt': popt,
            'pcov': pcov
        }
    except RuntimeError as r:
        print("No sigmoid fit due to exception: {}".format(r))
        return None



def exponential_model(day, ln_daily_growth, x_shift):
    "Exponential model formula"
    return np.exp(ln_daily_growth*(day-x_shift)) + Y_BASE


def fit_exponential_model(x_data, y_data):
    "Fits exponential model to data"
    sigma = [1] * len(y_data)
    # sigma[-1] = 0.1
    popt, pcov = curve_fit(exponential_model, x_data, y_data, sigma = sigma)
    params = popt
    errors = np.sqrt(np.diag(pcov))

    return {
        'ln_daily_growth': params[0],
        'ln_daily_growth_error': errors[0],
        'daily_growth': np.exp(params[0] + errors[0]**2 / 2),
        'tomorrow_growth': exponential_model(x_data[-1]+1, popt[0], popt[1]) - y_data[-1],
        'raw_daily_growth': np.exp(params[0]),
        'daily_growth_error': np.sqrt((np.exp(errors[0]**2)-1)*np.exp(2*params[0]+errors[0]**2)),
        'x_shift': params[1],
        'x_shift_error': errors[1],
        'popt': popt,
        'pcov': pcov
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

    days = range(x_data[0], x_data[0] + days_to_simulate)
    out_date = [BASE_DATE + datetime.timedelta(days=x) for x in range(x_data[0], x_data[0] + days_to_simulate)]
    out_y = y_data + [float('nan')]*(days_to_simulate - len(y_data))
    if not log_result is None:
        out_log = [logistic_model(x, *log_result['popt']) for x in days]
    out_exp = [exponential_model(x, *exp_result['popt']) for x in days]

    return {
        'date': out_date,
        'y': out_y,
        'logistic': out_log,
        'exponential': out_exp
    }


def main():
    "Entry point"

    death_mode = False
    global BASE_DATE
    global Y_BASE

    if len(sys.argv) > 1:
        death_mode = sys.argv[1] == '--deaths'

    if death_mode:
        print("Death mode")
        file_name = 'covid_deaths.txt'
        cases_axis_name = 'Összes halál'
        y_axis_name = 'Összes halott'
        element_marker = 'k+'
        plot_file_suffix = '-deaths'
        plot_title = 'COVID-19 görbeillesztés - összes halott'
    else:
        file_name = 'covid_data.txt'
        cases_axis_name = 'Jelentett esetek'
        y_axis_name = 'Összes eset'
        element_marker = 'ro'
        plot_file_suffix = ''
        plot_title = 'COVID-19 görbeillesztés - összes eset'


    x_data, y_data = parse_covid_data(file_name)
    Y_BASE = y_data[0]

    log_result = fit_logistic_model(x_data, y_data)
    if not log_result is None:
        peak_date_str = "Szigmoid inflexiós pont: " \
            "{} ± {:.2f} nap (Max meredekség: {:.2f}, f(x+1) - y(x) ≈ {:.2f})".format(
                log_result['peak_date'].strftime('%Y-%m-%d'), log_result['peak_date_error'],
                log_result['peak_growth'], log_result['tomorrow_growth']
            )
        print(peak_date_str)
        max_inf_str = "Szigmoid maximum: {:.2f} ± {:.2f}".format(
            log_result['max_inf'] + Y_BASE, log_result['max_inf_error'])
        print(max_inf_str)
    else:
        peak_date_str = "Szigmoid modell nem illeszkedik az adatokra."
        max_inf_str = ""
        print("Logistic curve is too bad fit for current data")

    exp_result = fit_exponential_model(x_data, y_data)
    print(exp_result)
    daily_growth_str = "Napi növekedés az exponenciális modell alapján: {:.2f}% ± {:.2}%. (Duplázódás: {:.2f} naponta, f(x+1) - y(x) ≈ {:.2f})".format(
            exp_result['daily_growth']*100-100, exp_result['daily_growth_error']*100, math.log(2)/math.log(exp_result['daily_growth']),
            exp_result['tomorrow_growth']
        )
    print(daily_growth_str)
    print("ln daily growth: {}, x_shift: {}".format(exp_result["ln_daily_growth"], exp_result["x_shift"]))

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
             element_marker, label=cases_axis_name)
    if not log_result is None:
        plt.plot(curve_data['date'], curve_data['logistic'],
                 'g-', label='Szigmoid modell')
    plt.plot(curve_data['date'], curve_data['exponential'],
             'b-', label='Exponenciális modell')
    plt.ylabel(y_axis_name)
    plt.xlabel('Dátum')
    if log_result is None:
        max_y = 2*max(y_data)
    else:
        max_y = max(curve_data['logistic'] + y_data)
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   max_inf_str + "\n" +
                   peak_date_str + "\n" +
                   daily_growth_str, va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(curve_data['date']), Y_BASE, max_y])
    plt.legend()
    plt.grid()
    plt.title("{} {}".format(plot_title, LAST_DATE))
    file_name = 'plot-'+LAST_DATE+plot_file_suffix+'.png'
    plt.savefig(file_name)
    print("Plot saved to {}".format(file_name))


if __name__ == "__main__":
    main()
