import pandas as pd

title_basics_df = pd.read_csv('data/title.basics.tsv.gz.tsv', sep='\t', index_col=False, dtype='unicode')
title_principals_df = pd.read_csv('data/title.principals.tsv.gz.tsv', sep='\t', index_col=False, dtype='unicode')
name_basics_df = pd.read_csv('data/name.basics.tsv.gz.tsv', sep='\t', index_col=False, dtype='unicode')

df1 = title_basics_df.query('titleType == "movie"')
df2 = df1.query(r'genres != "\\N"')
df2 = df2.query(r'startYear != "\\N"')
merge12 = pd.merge(df2, title_principals_df, on='tconst', how='left')
merge123 = pd.merge(merge12, name_basics_df, on='nconst', how='left')
merge123.drop(['originalTitle', 'endYear', 'runtimeMinutes', 'ordering', 'job', 'characters', 'birthYear', 'deathYear', 'primaryProfession', 'knownForTitles', 'tconst', 'nconst', 'titleType'], axis=1, inplace=True)
merge123.to_csv('data/out.csv')
