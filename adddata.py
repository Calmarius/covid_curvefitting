#!/usr/bin/env python3
"Adds new data to the datafiles."

import sys
import os
import datetime


def get_last_line(filename):
    "Gets last line from file"

    with open(filename, 'rb') as file:
        file.seek(-2, os.SEEK_END)
        while file.read(1) != b'\n':
            file.seek(-2, os.SEEK_CUR)
        return file.readline().decode()


def add_new_data(filename, new_data, today = False):
    "Adds new data to file"

    last_line = get_last_line(filename)
    fields = last_line.split()
    parsed_date = datetime.datetime.strptime(fields[0], '%Y-%m-%d')
    number = int(fields[1])

    if today:
        new_date = datetime.date.today()
    else:
        new_date = parsed_date + datetime.timedelta(days=1)

    new_number = number + new_data

    new_line = "{} {} +{}\n".format(new_date.strftime('%Y-%m-%d'),
                                  new_number, new_data)
    with open(filename, "a") as file:
        file.write(new_line)
    print("Line added: {}".format(new_line))


def main():
    "Entry point"

    if len(sys.argv) <= 2:
        print("Usage: {} <new cases> <new deaths>".format(sys.argv[0]))
        sys.exit(1)

    today_mode = False

    if len(sys.argv) >= 3 and sys.argv[3] == '--today':
        today_mode = True


    new_cases = int(sys.argv[1])
    new_deaths = int(sys.argv[2])

    add_new_data('covid_data.txt', new_cases, today_mode)
    add_new_data('covid_deaths.txt', new_deaths, today_mode)


if __name__ == "__main__":
    main()
