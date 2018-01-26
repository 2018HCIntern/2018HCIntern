import logging
import pandas as pd
import numpy as np
import click


@click.command()
@click.option('--csv-path', help='Input CSV Path')
def preprocess(csv_path, save_file_name):
    if csv_path is None:
        logging.error('Path Input Error')
        exit()

    """
    Grant.Application.ID  --> Remove
    Grant.Status  --> Rename to y
    Sponsor.Code  --> Unique + Remove Alphabet
    Grant.Category.Code  --> Unique + One Hot
    Contract.Value.Band...see.note.A  --> Max
    Start.date  --> Dividing
    RFCD.Code.1 ~ 5  --> Dummies + Percentage
    RFCD.Percentage.1 ~ 5  --> Dummies + Percentage
    SEO.Code.1 ~ 5  --> Dummies + Percentage
    SEO.Percentage.1 ~ 5  --> Dummies + Percentage
    Person.ID.1 ~ 15  --> count [O]
    Role.1 ~ 15  --> One Hot
    Year.of.Birth.1 ~ 15  --> get age and average
    Country.of.Birth.1 ~ 15
    Home.Language.1 ~ 15
    Dept.No..1 ~ 15
    Faculty.No..1 ~ 15  -->
    With.PHD.1 ~ 15  --> count [O]
    No..of.Years.in.Uni.at.Time.of.Grant.1 ~ 15
    Number.of.Successful.Grant.1 ~ 15  --> Sum [O]
    Number.of.Unsuccessful.Grant.1 ~ 15  --> Sum [O]
    A..1 ~ 15  --> Sum [O]
    A.1 ~ 15  --> Sum [O]
    B.1 ~ 15  --> Sum [O]
    C.1 ~ 15  --> Sum [O]
    """

    idf = pd.read_csv(csv_path, low_memory=False)
    odf = pd.DataFrame()
    sum_type = {
        'A.': 'A.',
        'A': 'A',
        'B': 'B',
        'C': 'C',
        'Number.of.Successful.Grant': 'Succ',
        'Number.of.Unsuccessful.Grant': 'Unsucc'
    }

    count_type = {
        'Person.ID': 'People',
        'With.PHD': 'PHD'
    }

    diff_type = {
        'Country.of.Birth': 'Country',
        'Role': 'Role',
        'Dept.No': 'Dept'
    }

    for attr in sum_type:
        icols = [attr + '.%d' % x for x in range(1, 16)]
        odf[sum_type[attr]] = idf[icols].apply(lambda x: np.nansum(x), axis=1)

    for attr in count_type:
        icols = [attr + '.%d' % x for x in range(1, 16)]
        odf[count_type[attr]] = idf[icols].notna(lambda x: np.sum(x), axis=1)

    for attr in diff_type:
        icols = [attr + '.%d' % x for x in range(1, 16)]
        odf[diff_type[attr]] = idf[icols].apply(lambda x: len(set(x[x.notna()])), axis=1)

    return odf
