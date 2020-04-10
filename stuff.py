#!/usr/bin/env python3

import numpy as np
import datetime
from scipy.optimize import curve_fit

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

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

# Fit logistic    
fit = curve_fit(logistic_model, xData, yData, p0=[3, 70, 10000])
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
peak_date = (baseDate + datetime.timedelta(days=fit[0][1]))
print("Predicted peak based on logistic model: {} +- {:.2f} days".format(
    peak_date.strftime('%Y-%m-%d'), errors[1]))
print("Predicted max infections: {:.2f} +- {:.2f}".format(fit[0][2], errors[2]))

# Fit exponential
def exponential_model(x, a, b, c):
    return a*np.exp(b*(x-c))

expfit = curve_fit(exponential_model, xData, yData)
print("Case growth per day: {:.2f}%".format(100*np.exp(expfit[0][1])-100))
errors = [np.sqrt(expfit[1][i][i]) for i in [0,1,2]]

# Check match:
days_to_simulate = 2*(peak_date - baseDate).days
i = 0
dayBase = xData[0]
while i < days_to_simulate:
    x = dayBase + i
    print("{} {} Predicted: {}".format(
        (baseDate + datetime.timedelta(days=x)).strftime('%Y-%m-%d'),
        yData[i] if i < len(yData) else "",
        logistic_model(x, fit[0][0], fit[0][1], fit[0][2])
        ))
    i = i + 1
