#!/usr/bin/env python3

import numpy as np
import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

with open('covid_data.txt') as f:
    content = f.readlines()

xData = []
yData = []

baseDate = datetime.datetime(2020, 3, 1)

for line in content:
    fields = line.split()
    date = (datetime.datetime.strptime(fields[0], '%Y-%m-%d')
            - baseDate).days
    number = int(fields[1])
    xData.append(date)
    yData.append(number)


def logistic_model(x, a, b, c):
    return c/(1+np.exp(-(x-b)/a))


# Fit logistic
fit = curve_fit(logistic_model, xData, yData, p0=[2, 50, 100000])
errors = np.sqrt(np.diag(fit[1]))
peak_date = (baseDate + datetime.timedelta(days=fit[0][1]))
peak_date_error = errors[1]
peak_date_str = "Predicted peak based on logistic model: {} ± {:.2f} days".format(
    peak_date.strftime('%Y-%m-%d'), peak_date_error)
print(peak_date_str)
max_inf = fit[0][2]
max_inf_error = errors[2]
max_inf_str = "Predicted max infections: {:.2f} ± {:.2f}".format(
    max_inf, max_inf_error)
print(max_inf_str)

# Fit exponential


def exponential_model(x, b, c):
    return np.exp(b*(x-c))


expfit = curve_fit(exponential_model, xData, yData)
print("Raw growth: {}".format(expfit[0][0]))
errors = np.sqrt(np.diag(expfit[1]))
print("Raw error: {}".format(errors[0]))
print("Exp error: {}".format(np.exp(errors[0])))

print("Case growth per day: {:.2f}%".format(100*np.exp(expfit[0][0])-100 ))

# Check match:
days_to_simulate = 2*(peak_date - baseDate).days
i = 0
dayBase = xData[0]
print("Date\tActual\tPredicted log\tPredicted exp")
outDate = []
outY = []
outLog = []
outExp = []
while i < days_to_simulate:
    x = dayBase + i
    newDate = (baseDate + datetime.timedelta(days=x))
    # newDate = x
    newY = yData[i] if i < len(yData) else float('nan')
    newLog = logistic_model(x, fit[0][0], fit[0][1], fit[0][2])
    newExp = exponential_model(x, expfit[0][0], expfit[0][1])
    outDate.append(newDate)
    outY.append(newY)
    outLog.append(newLog)
    outExp.append(newExp)
    print("{}\t{}\t{}\t{}".format(
        newDate, newY, newLog, newExp
    ))
    i = i + 1

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator())

plt.plot(outDate, outY, 'ro', outDate, outLog, 'g-')
plt.ylabel('cases')
plt.xlabel('date')
maxLog = max(outLog)
plt.text(min(outDate), maxLog, max_inf_str + "\n" + peak_date_str)
plt.axis([min(outDate), max(outDate), 0, maxLog])
plt.savefig('plot.png')
