#
# This file is part of metavirommodel
# (https://github.com/I-Bouros/metavirommodel)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
"""Processing script for the precipitation data from [1]_ .

It computes the time-dependent rainfall values for Uganda which are
then stored in a separate csv file.

References
----------
.. [1] Tumusiime Andrew Gahwera; Odongo Steven Eyobu; Mugume Isaac, 2023,
       "Cleaned Weather Dataset for Uganda",
       https://doi.org/10.7910/DVN/PQLYHP, Harvard Dataverse, V5,
       UNF:6:hvhyd9OKlHyHrf+K6q1Yig== [fileUNF]

"""

import os
import pandas as pd
from datetime import datetime
import time
from time import mktime
import numpy as np


def read_full_precipitation_data(pr_file: str):
    """
    Parses the csv document containing the full precipitation data.

    Parameters
    ----------
    pr_file : str
        The name of the precipitation data file used.

    Returns
    -------
    pandas.Dataframe
        Dataframe of the rainfall values.

    """
    # Select data from the given state
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), pr_file))

    return data


def process_precipitation_data(
        data: pd.DataFrame,
        start_date: str = '01/01/2020 06:00',
        end_date: str = '31/12/2022 15:00'):
    """
    Computes the daily total rainfall between the given dates.

    Parameters
    ----------
    data : pandas.Dataframe
        Dataframe of the full rainfall data of cases in a given region.
    start_date : str
        The initial date (date/month/year hour:min) from which the
        precipitation levels are calculated.
    end_date : str
        The final date (ydate/month/year hour:min) from which the
        precipitation levels are calculated.

    Returns
    -------
    pandas.Dataframe
        Processed total daily precipitation levels in a given region as
        a dataframe.

    """
    # Process dates
    data['processed-date'] = [process_dates(x) for x in data['date']]
    data = data.sort_values('processed-date')

    # Keep only those values within the date range
    data = data[data['processed-date'] >= process_dates(start_date)]
    data = data[data['processed-date'] <= process_dates(end_date)]

    # Add a column 'Time', which is the number of days from start_date
    start = process_dates(start_date)
    data['Day'] = [(x - start).days + 1
                   for x in data['processed-date']]

    # Keep only those columns we need
    data = data[['Day', 'Precmm']]

    precp = pd.DataFrame(columns=['Day', 'Precmm'])
    for t in data['Day'].unique():
        daily_data = data[data['Day'] == t]
        newrow = pd.DataFrame([{
            'Day': t,
            'Precmm': np.sum(daily_data['Precmm'].values)
        }])
        precp = pd.concat([precp, newrow])

    return precp


def process_dates(date: str):
    """
    Processes dates into `datetime` format.

    Parameters
    ----------
    date : str
        Date (date/month/year hour:min) as it appears in the data frame.

    Returns
    -------
    datetime.datetime
        Date processed into correct format.

    """
    struct = time.strptime(date, '%d/%m/%Y %H:%M')
    return datetime.fromtimestamp(mktime(struct))


def main():
    """
    Combines the timelines of deviation percentages and baseline
    activity-specific contact matrices to get weekly, region-specific
    contact matrices.

    Returns
    -------
    csv
        Processed files for the baseline and region-specific time-dependent
        contact matrices for each different region found in the default file.

    """
    data = read_full_precipitation_data('masindi63654.csv')

    # Rename the columns of interest
    data = data.rename(columns={'Date': 'date'})

    # Keep only columns of interest
    data = data[['date', 'Precmm']]

    # Process precipitation results
    precipitation = process_precipitation_data(
        data,
        start_date='01/01/2020 06:00',
        end_date='31/12/2022 15:00')

    # Transform recorded matrix of serial intervals to csv file
    precipitation.to_csv(
        os.path.join(os.path.dirname(__file__), 'Precipitation.csv'),
        index=False)


if __name__ == '__main__':
    main()
