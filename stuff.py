#!/usr/bin/env python3

"""
Quickly written curve fitting script for covid data.
"""

import math
import sys
import datetime
from scipy.optimize import curve_fit
import numpy as np
import matplotlib

matplotlib.use('Agg')
if 1 < 2:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt


TODAY = datetime.datetime.now().strftime('%Y-%m-%d')


def parse_covid_data(filename):
    "Loads the date from file into two lists separate x and y list"
    with open(filename) as file:
        content = file.readlines()

    x_data = []
    y_data = []
    last_date = ''
    base_date = ''

    for line in content:
        fields = line.split()
        if len(fields) < 2:
            continue
        parsed_date = datetime.datetime.strptime(fields[0], '%Y-%m-%d')
        if base_date == '':
            base_date = parsed_date
        last_date = fields[0]
        date = (parsed_date - base_date).days
        number = float(fields[1])
        x_data.append(date)
        y_data.append(number)

    return {
        'x_data': x_data,
        'y_data': y_data,
        'base_date': base_date,
        'last_date_str': last_date
    }


def get_logistic_model(y_base):
    "Generates logistic model function for the given Y base"
    def logistic_model(day, x_scale, peak, max_cases):
        "Logistic model formula"
        return max_cases/(1+np.exp(-(day-peak)/x_scale)) + y_base
    return logistic_model


def fit_logistic_model(x_data, y_data, y_base, base_date):
    "Fits data into logistic curve"

    try:
        sigma = [1] * len(y_data)
        # sigma[-1] = 0.1
        model = get_logistic_model(y_base)
        popt, pcov = curve_fit(model, x_data, y_data, p0=[
            2, 60, 100000], sigma=sigma)
        errors = np.sqrt(np.diag(pcov))
        peak_date = (base_date + datetime.timedelta(days=popt[1]))
        peak_date_error = errors[1]
        max_inf = popt[2]
        max_inf_error = errors[2]

        if peak_date_error > 1e7 or max_inf_error > 1e7:
            print("No sigmoid fit due to too large covariance.")
            return None

        if max_inf_error > max_inf:
            print(
                "No sigmoid fit because the uncertainty of the "
                "maximum is larger than the maximum itself.")
            return None

        return {
            'peak': popt[1],
            'peak_date': peak_date,
            'peak_date_error': peak_date_error,
            'peak_growth': model(popt[1]+1, popt[0], popt[1], popt[2])
            - model(popt[1], popt[0], popt[1], popt[2]),
            'tomorrow_growth': model(x_data[-1]+1, popt[0], popt[1], popt[2]) - y_data[-1],
            'max_inf': max_inf,
            'max_inf_error': max_inf_error,
            'x_scale': popt[0],
            'x_scale_error': errors[0],
            'popt': popt,
            'pcov': pcov
        }
    except RuntimeError as rte:
        print("No sigmoid fit due to exception: {}".format(rte))
        return None


def get_exponential_model(y_base):
    "Generates exponential model function for the given Y base"

    def exponential_model(day, ln_daily_growth, x_shift):
        "Exponential model formula"

        return np.exp(ln_daily_growth*(day-x_shift)) + y_base
    return exponential_model


def fit_exponential_model(x_data, y_data, y_base):
    "Fits exponential model to data"
    sigma = [1] * len(y_data)
    # sigma[-1] = 0.1
    model = get_exponential_model(y_base)
    popt, pcov = curve_fit(model, x_data, y_data, sigma=sigma)
    params = popt
    errors = np.sqrt(np.diag(pcov))

    return {
        'ln_daily_growth': params[0],
        'ln_daily_growth_error': errors[0],
        'daily_growth': np.exp(params[0] + errors[0]**2 / 2),
        'tomorrow_growth': model(x_data[-1]+1, popt[0], popt[1]) - y_data[-1],
        'raw_daily_growth': np.exp(params[0]),
        'daily_growth_error': np.sqrt((np.exp(errors[0]**2)-1)*np.exp(2*params[0]+errors[0]**2)),
        'x_shift': params[1],
        'x_shift_error': errors[1],
        'popt': popt,
        'pcov': pcov
    }


def create_curve_data(x_data, y_data, y_base, base_date, log_result, exp_result):
    """
    Creates the curves to be used when plotting data based
    on the calculated results.
    """
    if log_result is None:
        days_to_simulate = 2*(x_data[-1] - x_data[0] + 1)
    else:
        days_to_simulate = max(
            2*(log_result['peak_date'] - base_date).days, x_data[-1] - x_data[0] + 1)

    days = range(x_data[0], x_data[0] + days_to_simulate)
    out_date = [base_date + datetime.timedelta(days=x)
                for x in range(x_data[0], x_data[0] + days_to_simulate)]

    out_y = y_data + [float('nan')]*(days_to_simulate - len(y_data))

    if not log_result is None:
        out_log = [get_logistic_model(y_base)(
            x, *log_result['popt']) for x in days]
    else:
        out_log = [float('nan')] * days_to_simulate

    out_exp = [get_exponential_model(y_base)(
        x, *exp_result['popt']) for x in days]

    return {
        'date': out_date,
        'y': out_y,
        'logistic': out_log,
        'exponential': out_exp
    }

def print_curves(curve_data):
    "Prints the curve data into terminal."

    print("{:<15}{:<15}{:<15}{:<15}".format(
        "Date", "Actual", "Predicted log", "Predicted exp"))
    for i in range(0, len(curve_data['date'])):
        print("{:<15}{:>15}{:>15.2f}{:>15.2f}".format(
            curve_data['date'][i].strftime('%Y-%m-%d'),
            curve_data['y'][i],
            curve_data['logistic'][i],
            curve_data['exponential'][i]
        ))

# TODO for me: Maybe add y_base to covid_data
def save_plot(curve_data, covid_data, y_base, log_result, texts):
    "Generates and saves the plot."

    axes = plt.gca()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    axes.xaxis.set_minor_locator(mdates.DayLocator())

    plt.figure(figsize=[10.24, 7.68])
    plt.plot(curve_data['date'], curve_data['y'],
             texts['element_marker'], label=texts['cases_axis_name'])
    if not log_result is None:
        plt.plot(curve_data['date'], curve_data['logistic'],
                 'g-', label='Szigmoid modell')
    plt.plot(curve_data['date'], curve_data['exponential'],
             'b-', label='Exponenciális modell')
    plt.ylabel(texts['y_axis_name'])
    plt.xlabel('Dátum')
    if log_result is None:
        max_y = 2*max(covid_data['y_data'])
    else:
        max_y = max(curve_data['logistic'] + covid_data['y_data'])
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   texts['max_inf_str'] + "\n" +
                   texts['peak_date_str'] + "\n" +
                   texts['daily_growth_str'], va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(curve_data['date']), y_base, max_y])
    plt.legend()
    plt.grid()
    plt.title("{} {}".format(texts['plot_title'], covid_data['last_date_str']))
    file_name = 'plot-'+covid_data['last_date_str'] + \
        texts['plot_file_suffix']+'.png'
    plt.savefig(file_name)
    print("Plot saved to {}".format(file_name))



def main():
    "Entry point"

    death_mode = False

    if len(sys.argv) > 1:
        death_mode = sys.argv[1] == '--deaths'

    if death_mode:
        print("Death mode")
        texts = {
            'file_name': 'covid_deaths.txt',
            'cases_axis_name': 'Összes halál',
            'y_axis_name': 'Összes halott',
            'element_marker': 'k+',
            'plot_file_suffix': '-deaths',
            'plot_title': 'COVID-19 görbeillesztés - összes halott',
        }
    else:
        texts = {
            'file_name': 'covid_data.txt',
            'cases_axis_name': 'Jelentett esetek',
            'y_axis_name': 'Összes eset',
            'element_marker': 'ro',
            'plot_file_suffix': '',
            'plot_title': 'COVID-19 görbeillesztés - összes eset',
        }

    # x_data, y_data, base_date, last_date
    covid_data = parse_covid_data(texts['file_name'])
    y_base = covid_data['y_data'][0]

    log_result = fit_logistic_model(
        covid_data['x_data'], covid_data['y_data'], y_base, covid_data['base_date'])
    if not log_result is None:
        texts['peak_date_str'] = "Szigmoid inflexiós pont: " \
            "{} ± {:.2f} nap (Max meredekség: {:.2f}, f(x+1) - y(x) ≈ {:.2f})".format(
                log_result['peak_date'].strftime(
                    '%Y-%m-%d'), log_result['peak_date_error'],
                log_result['peak_growth'], log_result['tomorrow_growth']
            )
        texts['max_inf_str'] = "Szigmoid maximum: {:.2f} ± {:.2f}".format(
            log_result['max_inf'] + y_base, log_result['max_inf_error'])
        print(texts['max_inf_str'])
    else:
        texts['peak_date_str'] = "Szigmoid modell nem illeszkedik az adatokra."
        texts['max_inf_str'] = ""
        print("Logistic curve is too bad fit for current data")

    exp_result = fit_exponential_model(
        covid_data['x_data'], covid_data['y_data'], y_base)
    print(exp_result)
    texts['daily_growth_str'] = ("Napi növekedés az exponenciális modell alapján:"
                        " {:.2f}% ± {:.2}%."
                        " (Duplázódás: {:.2f} naponta, f(x+1) - y(x) ≈ {:.2f})").format(
                            exp_result['daily_growth']*100-100, exp_result['daily_growth_error'] *
                            100, math.log(
                                2)/math.log(exp_result['daily_growth']),
                            exp_result['tomorrow_growth']
    )
    print(texts['daily_growth_str'])
    print("ln daily growth: {}, x_shift: {}".format(
        exp_result["ln_daily_growth"], exp_result["x_shift"]))

    curve_data = create_curve_data(
        covid_data['x_data'],
        covid_data['y_data'],
        y_base,
        covid_data['base_date'],
        log_result,
        exp_result
    )

    print_curves(curve_data)

    save_plot(curve_data, covid_data, y_base, log_result, texts)

if __name__ == "__main__":
    main()
