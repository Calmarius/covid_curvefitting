#!/usr/bin/env python3
"Adds data to COVID data set as if cases were decaying"

import datetime
import sys

import adddata


def add_decay_data(filename, days_to_halve, number_of_points):
    "Adds decay data"

    last_line = adddata.get_last_line(filename)
    fields = last_line.split()
    current_date = datetime.datetime.strptime(fields[0], '%Y-%m-%d')
    current_value = int(fields[1])
    current_growth = int(fields[2])
    factor = 0.5 ** (1.0/days_to_halve)
    with open(filename, 'a') as file:
        for i in range(1, number_of_points):
            new_date = current_date + datetime.timedelta(days=i)
            new_growth = current_growth * (factor ** i)
            current_value += new_growth
            new_line = "{} {} +{}\n".format(new_date.strftime('%Y-%m-%d'),
                                            int(current_value), int(new_growth))
            print(new_line)
            file.write(new_line)


def main():
    "Entry point"

    if len(sys.argv) <= 2:
        print("Usage: {} <days to halve> <number of points>".format(
            sys.argv[0]))
        sys.exit(1)

    days_to_halve = int(sys.argv[1])
    number_of_points = int(sys.argv[2])

    add_decay_data('covid_data.txt', days_to_halve, number_of_points)
    add_decay_data('covid_deaths.txt', days_to_halve, number_of_points)


if __name__ == "__main__":
    main()
