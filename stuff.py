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
END_PEAK_GROWTH_RATE = 0.001


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


def get_gen_logistic_model():
    "Generates general logistic model function for the given Y base"
    # I have no idea how to avoid too many arguments warnings here...
    # pylint: disable=R0913
    def log_model(day, x_scale, peak, height, y_floor, ksi):
        "General logistic model formula"

        if not math.isfinite(ksi):
            raise RuntimeError("ksi is invalid. Value = {}".format(ksi))
        return height/pow(1+ksi*np.exp(-(day-peak)/x_scale), 1/ksi) + y_floor
    return log_model

def logistic_model(day, x_scale, peak, height, y_floor):
    "Logistic model formula"
    return height/pow(1+np.exp(-(day-peak)/x_scale), 1) + y_floor


def find_end_of_logistic(logresult, base_date):
    "Finds end of the logisitic curve, where slope starts to be less than 1"

    endpoint = logresult['peak']
    func = logresult['function']
    while (func(endpoint + 1, *logresult['popt']) - func(endpoint, *logresult['popt']) >=
           logresult['peak_growth']*END_PEAK_GROWTH_RATE):
        endpoint = endpoint + 1
    return base_date + datetime.timedelta(days=endpoint)


def fit_logistic_model(x_data, y_data, base_date):
    "Fits data into logistic curve"

    try:
        sigma = [1] * len(y_data)
        # sigma[-1] = 0.1
        model = logistic_model
        result = curve_fit(model, x_data, y_data, p0=[
            2, 60, 100000, y_data[0]], sigma=sigma)
        popt = result[0]
        pcov = result[1]
        errors = np.sqrt(np.diag(pcov))
        peak_date_error = errors[1]
        if popt[2] < 0:
            print(
                "Symmetric log: Reversal happened! " +
                "The graph has negative height. The floor is the top!")
            max_inf = popt[3]
        else:
            max_inf = popt[2] + popt[3]
        max_inf_error = errors[2]

        if peak_date_error > 1e7 or max_inf_error > 1e7:
            print("No sigmoid fit due to too large covariance. max_inf_error: {:.2f}".format(
                max_inf_error))
            return None

        if max_inf_error > max_inf:
            print(
                "No sigmoid fit because the uncertainty of the "
                "maximum is larger than the maximum itself.")
            return None

        peak_date = (base_date + datetime.timedelta(days=popt[1]))
        res = {
            'function': model,
            'peak': popt[1],
            'peak_date': peak_date,
            'peak_date_error': peak_date_error,
            'peak_growth': model(popt[1]+1, *popt)
            - model(popt[1], *popt),
            'tomorrow_diff':
                model(x_data[-1]+1, *popt) - y_data[-1],
            'tomorrow_growth':
                model(x_data[-1], *popt) - model(x_data[-1]-1, *popt),
            'max_inf': max_inf,
            'max_inf_error': max_inf_error,
            'x_scale': popt[0],
            'x_scale_error': errors[0],
            'y_base': popt[3],
            'popt': popt,
            'pcov': pcov,
            'name': 'Szimmetrikus szigmoid'
        }
        res['final_date'] = find_end_of_logistic(res, base_date)
        return res
    except RuntimeError as rte:
        print("No sigmoid fit due to exception: {}".format(rte))
        return None


def fit_gen_logistic_model(x_data, y_data, base_date):
    "Fits data into general logistic curve"

    try:
        model = get_gen_logistic_model()
        result = curve_fit(logistic_model, x_data, y_data, p0=[
            2, 60, 100000, y_data[0]])
        popt_s = result[0]

        result = curve_fit(model, x_data, y_data,
                           p0=popt_s.tolist() + [1])
        popt = result[0]
        pcov = result[1]

        errors = np.sqrt(np.diag(pcov))
        peak_date_error = errors[1]
        if popt[2] < 0:
            print(
                "Generic log: Reversal happened! " +
                "The graph has negative height. The floor is the top!")
            max_inf = popt[3]
        else:
            max_inf = popt[2] + popt[3]
        max_inf_error = errors[2]

        if peak_date_error > 1e7 or max_inf_error > 1e7:
            print("No generic logistic fit due to too large covariance. "
                  "max_inf_error: {:.2f}".format(
                      max_inf_error))
            return None

        if max_inf_error > max_inf:
            print(
                "No generic logistic fit because the uncertainty of the "
                "maximum is larger than the maximum itself.")
            return None

        peak_date = (base_date + datetime.timedelta(days=popt[1]))

        res = {
            'function': model,
            'peak': popt[1],
            'peak_date': peak_date,
            'peak_date_error': peak_date_error,
            'peak_growth': model(popt[1]+1, *popt)
            - model(popt[1], *popt),
            'tomorrow_diff':
                model(x_data[-1]+1, *popt) - y_data[-1],
            'tomorrow_growth':
                model(x_data[-1], *popt) - model(x_data[-1]-1, *popt),
            'max_inf': max_inf,
            'max_inf_error': max_inf_error,
            'x_scale': popt[0],
            'x_scale_error': errors[0],
            'y_base': popt[3],
            'popt': popt,
            'pcov': pcov,
            'name': 'Általános szigmoid'
        }
        res['final_date'] = find_end_of_logistic(res, base_date)
        return res
    except RuntimeError as rte:
        print("No generic logistic fit due to exception: {}".format(rte))
        return None

def exponential_model(day, ln_daily_growth, x_shift, y_base):
    "Exponential model formula"

    return np.exp(ln_daily_growth*(day-x_shift)) + y_base


DAILY_GROWTH_GUESS = np.log(1.1)

def compute_exponential_initial_guess(x_data, y_data):
    "asdassda"

    a_data = [x_data[0], x_data[-1]]
    b_data = [y_data[0], y_data[-1]]

    b_tmp = np.exp(DAILY_GROWTH_GUESS*(a_data[1]-a_data[0]))
    y_coord = (b_data[1]-b_tmp*b_data[0])/(1-b_tmp)
    x_coord = np.log(np.exp(DAILY_GROWTH_GUESS*a_data[0])/(b_data[0]-y_coord))/DAILY_GROWTH_GUESS

    return [x_coord, y_coord]


def fit_exponential_model(x_data, y_data):
    "Fits exponential model to data"

    try:
        sigma = [1] * len(y_data)
        # sigma[-1] = 0.1
        model = exponential_model
        initial_guess = compute_exponential_initial_guess(x_data, y_data)
        print("Initial exponential guess parameters: {}".format(initial_guess))
        result = curve_fit(model, x_data, y_data, sigma=sigma, p0=[
                           DAILY_GROWTH_GUESS, initial_guess[0], initial_guess[1]])
        popt = result[0]
        pcov = result[1]
        params = popt
        errors = np.sqrt(np.diag(pcov))

        if errors[0] > 1e7 or errors[1] > 1e7 or errors[2] > 1e7:
            print(
                "No exponential fit due to too large corariance. Errors: {}".format(errors))
            return None

        return {
            'ln_daily_growth': params[0],
            'ln_daily_growth_error': errors[0],
            'daily_growth': np.exp(params[0] + errors[0]**2 / 2),
            'tomorrow_diff': model(x_data[-1]+1, *popt) - y_data[-1],
            'tomorrow_growth':
                model(x_data[-1], *popt) - model(x_data[-1]-1, *popt),
            'raw_daily_growth': np.exp(params[0]),
            'daily_growth_error': np.sqrt(
                (np.exp(errors[0]**2)-1) *
                np.exp(2*params[0]+errors[0]**2)
            ),
            'x_shift': params[1],
            'x_shift_error': errors[1],
            'popt': popt,
            'pcov': pcov,
            'name': 'Exponenciális'
        }
    except RuntimeError as rte:
        print("No exponential fit due to exception {}".format(rte))
        return None


def create_curve_data(x_data, y_data, base_date, log_results, exp_result):
    """
    Creates the curves to be used when plotting data based
    on the calculated results.
    """

    data_map = dict(zip(x_data, y_data))

    days_to_simulate = None

    # Choose the logistic curve with higher asymptote
    choosen_log_result = log_results['symmetric']
    if (log_results['general'] is not None) and (log_results['symmetric'] is not None) and (
            log_results['general']['max_inf'] > log_results['symmetric']["max_inf"]):
        choosen_log_result = log_results['general']

    # Choose range such that the end is at he position where the slope of the graph is less than 1.
    if choosen_log_result is not None:
        days_to_simulate = (choosen_log_result['final_date'] - base_date).days

    # If we don't have a logistic curve (so no days yet), but have exponential
    # then just simulate twice as many days as we have so far:
    if (days_to_simulate is None) and exp_result is not None:
        days_to_simulate = 2*(x_data[-1] - x_data[0] + 1)

    # If we already have days to simulate make sure all data is on the chart.
    if choosen_log_result is not None:
        days_to_simulate = max(x_data[-1] - x_data[0] + 1, days_to_simulate)

    if days_to_simulate is None:
        days_to_simulate = len(y_data)

    days = range(x_data[0], x_data[0] + days_to_simulate)
    out_date = [base_date + datetime.timedelta(days=x)
                for x in days]

    out_y = [data_map.get(x, float('nan')) for x in days]

    if log_results['symmetric'] is not None:
        out_log = [logistic_model(
            x, *log_results['symmetric']['popt']) for x in days]
    else:
        out_log = None

    if log_results['general'] is not None:
        out_genlog = [get_gen_logistic_model()(
            x, *log_results['general']['popt']) for x in days]
    else:
        out_genlog = None

    if exp_result is not None:
        out_exp = [exponential_model(
            x, *exp_result['popt']) for x in days]
    else:
        out_exp = None

    return {
        'date': out_date,
        'y': out_y,
        'logistic': out_log,
        'general_logistic': out_genlog,
        'exponential': out_exp
    }


def print_curves(curve_data):
    "Prints the curve data into terminal."

    print("{:<15}{:<15}{:<15}{:<15}{:<15}".format(
        "Date", "Actual", "Predicted log", "Predicted gen. log", "Predicted exp"))
    for i in range(0, len(curve_data['date'])):
        print("{:<15}{:>15}{:>15.2f}{:>15.2f}{:>15.2f}".format(
            curve_data['date'][i].strftime('%Y-%m-%d'),
            curve_data['y'][i],
            curve_data['logistic'][i] if curve_data['logistic'] is not None else float(
                "nan"),
            curve_data['general_logistic'][i]
            if curve_data['general_logistic'] is not None else float("nan"),
            curve_data['exponential'][i] if curve_data['exponential'] is not None else float(
                "nan")
        ))


def save_plot(curve_data, covid_data, texts):
    "Generates and saves the plot."

    axes = plt.gca()
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    axes.xaxis.set_minor_locator(mdates.DayLocator())

    plt.figure(figsize=[10.24, 7.68])
    plt.plot(curve_data['date'], curve_data['y'],
             texts['element_marker'], label=texts['cases_axis_name'])
    if curve_data['logistic'] is not None:
        plt.plot(curve_data['date'], curve_data['logistic'],
                 'g-', label='Szimmetrikus szigmoid modell')
    if curve_data['general_logistic'] is not None:
        plt.plot(curve_data['date'], curve_data['general_logistic'],
                 'r-', label='Általános szigmoid modell')

    if curve_data['exponential'] is not None:
        plt.plot(curve_data['date'], curve_data['exponential'],
                 'b-', label='Exponenciális modell')
    plt.ylabel(texts['y_axis_name'])
    plt.xlabel('Dátum')
    max_y = None
    if curve_data['logistic'] is not None:
        max_y = max(curve_data['logistic'] +
                    covid_data['y_data'] + ([max_y] if max_y else []))
    if curve_data['general_logistic'] is not None:
        max_y = max(curve_data['general_logistic'] +
                    covid_data['y_data'] + ([max_y] if max_y else []))
    max_y = max_y or 2 * \
        (covid_data['y_data'][-1] - covid_data['y_data'][0]) + \
        covid_data['y_data'][0]
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.9])
    plt.gcf().text(0.01, 0.01,
                   texts['max_inf_str'] + "\n" +
                   texts['peak_date_str'] + "\n" +
                   texts['daily_growth_str'], va='bottom'
                   )
    plt.axis([min(curve_data['date']), max(
        curve_data['date']), covid_data['y_data'][0], max_y])
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

    sym_log_result = fit_logistic_model(
        covid_data['x_data'], covid_data['y_data'], covid_data['base_date'])
    gen_log_result = fit_gen_logistic_model(
        covid_data['x_data'], covid_data['y_data'], covid_data['base_date'])
    if sym_log_result is not None:
        print("Symmetric log result popt: {}".format(
            [float('{:.2f}'.format(x)) for x in sym_log_result['popt']]))

    if gen_log_result is not None:
        print("Generic log result popt: {}".format(
            [float('{:.2f}'.format(x)) for x in gen_log_result['popt']]))

        

    log_result = sym_log_result
    if (gen_log_result is not None) and (sym_log_result is not None) and (
            gen_log_result['max_inf'] > sym_log_result["max_inf"]):
        log_result = gen_log_result

    if log_result is not None:
        texts['peak_date_str'] = (
            "{} inflexiós pont: "
            "{} ± {:.2f} nap"
            " (Max meredekség: {:.2f}, f(x+1) - y(x) ≈ {:.2f}, f(x+1) - f(x) ≈ {:.2f})").format(
            log_result['name'],
            log_result['peak_date'].strftime(
                '%Y-%m-%d'), log_result['peak_date_error'],
            log_result['peak_growth'],
            log_result['tomorrow_diff'],
            log_result['tomorrow_growth']
        )
        texts['max_inf_str'] = "{} maximum: {:.2f} ± {:.2f} (df/dx < {:.2f} helye): {})".format(
            log_result['name'],
            log_result['max_inf'],
            log_result['max_inf_error'],
            log_result['peak_growth'] * END_PEAK_GROWTH_RATE,
            log_result['final_date'].strftime(
                '%Y-%m-%d')
        )
        print(texts['max_inf_str'])
        print(texts['peak_date_str'])
    else:
        texts['peak_date_str'] = "Szigmoid modell nem illeszkedik az adatokra."
        texts['max_inf_str'] = ""
        print("Logistic curve is too bad fit for current data")

    exp_result = fit_exponential_model(
        covid_data['x_data'], covid_data['y_data'])
    print(exp_result)
    if exp_result is not None:
        print("Exp result popt: {}".format(
            [float('{:.2f}'.format(x)) for x in exp_result['popt']]))
        texts['daily_growth_str'] = (
            "Napi növekedés az exponenciális modell alapján:"
            " {:.2f}% ± {:.2}%."
            " (Duplázódás: {:.2f} naponta, f(x+1) - y(x) ≈ {:.2f}, f(x+1) - f(x) ≈ {:.2f})").format(
            exp_result['daily_growth']*100-100, exp_result['daily_growth_error'] *
            100, math.log(
                2)/math.log(exp_result['daily_growth']),
            exp_result['tomorrow_diff'],
            exp_result['tomorrow_growth']
        )
        print(texts['daily_growth_str'])
        print("ln daily growth: {}, x_shift: {}".format(
            exp_result["ln_daily_growth"], exp_result["x_shift"]))
    else:
        texts['daily_growth_str'] = "Exponenciális modell nem illeszkedik az adatokra"

    log_results = {
        'symmetric': sym_log_result,
        'general': gen_log_result
    }

    curve_data = create_curve_data(
        covid_data['x_data'],
        covid_data['y_data'],
        covid_data['base_date'],
        log_results,
        exp_result
    )

    print_curves(curve_data)

    save_plot(curve_data, covid_data, texts)


if __name__ == "__main__":
    main()
