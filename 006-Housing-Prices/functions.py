
def unistats(df):
    import pandas as pd
    output_df = pd.DataFrame \
        (columns=['Count', 'Missing', 'Unique', 'Dtype', 'Numeric', 'Mode', 'Mean', 'Min', '25%', 'Median', '75%', 'Max', 'Std', 'Skew', 'Kurt'])

    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(),
                                  df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), df[col].mode().values[0],
                                  df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(),
                                  df[col].quantile(0.75), df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
        else:
            output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(),
                                  df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), df[col].mode().values[0],
                                  '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return output_df.sort_values(by=['Numeric', "Skew", 'Unique'], ascending=False)