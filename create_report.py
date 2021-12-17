#!/usr/bin/env python3
"Creates reports to post"

import datetime
import json
import sys


def parse_date(date_str):
    "Parses ISO dates"

    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def diff_key_base(model, prev_model, key, prec=0, is_date=False):
    "Diffs the data from two models"

    if is_date:
        diff = (parse_date(model[key]) -
                parse_date(prev_model[key])).days
    else:
        diff = model[key] - prev_model[key]
    return f"{diff:+.{prec}f}".replace(".", ",")


def fltfmt(number, prec=0):
    "Format float with comma"

    return f"{number:.{prec}f}".replace(".", ",")


def main():
    "Entry point"

    if len(sys.argv) <= 2:
        print(f"Usage {sys.argv[0]} <prev_json> <current_json>")

    prev_json = sys.argv[1]
    current_json = sys.argv[2]

    with open(prev_json) as fdsc:
        prev = json.load(fdsc)

    with open(current_json) as fdsc:
        current = json.load(fdsc)

    log_model = current["log"]
    prev_log_model = prev["log"]
    exp_model = current["exp"]
    prev_exp_model = prev["exp"]
    today = datetime.datetime.today()
    post_peak = False

    def diff_key(key, prec=0, is_date=False):
        return diff_key_base(log_model, prev_log_model, key, prec, is_date)

    def diff_key_exp(key, prec=0, is_date=False):
        return diff_key_base(exp_model, prev_exp_model, key, prec, is_date)

    def diff_key_root(key, prec=0, is_date=False):
        return diff_key_base(current, prev, key, prec, is_date)

    if log_model is None:
        print("Szigmoid modell nem illeszkedik az adatokra")
    else:
        peak_date = parse_date(log_model['peak'])
        datediff = (today - peak_date).days
        print(f"{log_model['name']} modell:")
        print()
        if datediff >= 7:
            # We are in the post peak mode so we write the top of the curve and top date.
            post_peak = True
            print(f"- Görbe teteje: {fltfmt(log_model['top_of_curve'])}" +
                  f" ({diff_key('top_of_curve')})")
            print(f"- Görbe vége ({fltfmt(log_model['growth_at_top'], 2)}/nap helye):" +
                  f" {log_model['top_of_curve_date']}" +
                  f" ({diff_key('top_of_curve_date', 0, True)} nap)")

        # Pre-peak mode we don't write the top.
        print(
            f"- Inflexiós pont: {log_model['peak']} ({diff_key('peak', 0, True)} nap)")
        print(f"- Max meredekség: {fltfmt(log_model['peak_growth'],2)}/nap" +
              f" ({diff_key('peak_growth', 2)}/nap)")
        print("- A javuláshoz holnap ennyinél kellene kevesebbet jelenteniük:" +
              f" {fltfmt(log_model['tomorrow_diff'])} ({diff_key('tomorrow_diff')})")
        print(f"- Görbe meredeksége: {fltfmt(log_model['tomorrow_growth'],2)}/nap" +
              f" ({diff_key('tomorrow_growth', 2)}/nap)")
        print()

    if not post_peak and exp_model is not None:
        print("Exponenciális modell: ")
        print("")
        print(f"- Napi növekedés: {fltfmt(exp_model['growth'],2)}%" +
              f" ({diff_key_exp('growth', 2)}%p)")
        print(f"- Duplázódás: {fltfmt(exp_model['duplication'],2)} naponta" +
              f" ({diff_key_exp('duplication', 2)})")
        print("- A javuláshoz holnap ennyinél kellene kevesebbet jelenteniük:" +
              f" {fltfmt(exp_model['tomorrow_diff'])} ({diff_key_exp('tomorrow_diff')})")
        print(f"- Görbe meredeksége: {fltfmt(exp_model['tomorrow_growth'],2)}/nap" +
              f" ({diff_key_exp('tomorrow_growth', 2)}/nap)")

    print(f"- Heti mozgóátlag: {current['weekly_moving_average']}" +
          f" ({diff_key_root('weekly_moving_average')})")


if __name__ == "__main__":
    main()
