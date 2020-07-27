import cdsapi
import argparse
import calendar
import datetime


c = cdsapi.Client()



monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--year', type = int)
parser.add_argument('--month', type = int)

args = parser.parse_args()

num_days = calendar.monthrange(args.year, args.month)[1]
days = [datetime.date(args.year, args.month, day).day for day in range(1, num_days+1)]

fname = monthDict[args.month] + '_' + str(args.year)

print(args.year, args.month)
print(days)
print(fname)

