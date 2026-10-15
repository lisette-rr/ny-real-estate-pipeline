import pandas as pd

def load_and_clean_data(filepath="ny_real_estate_sold_properties_2026.csv"):
    """
    Method serves to put all the necessary data cleaning found necessary in the 
    Jupyter notebook in one place. Drops columns that are unnecessary, imputes missing values,
    and filters to just have data from cities that have at least 100 housing units.

    Parameters:
    - filepath: the path of the data file

    Return:
    - df_final: the final cleaned DataFrame
    """

    # reads in the data
    df = pd.read_csv(filepath)
    
    # dropping columns that are repetive (have high correlations with other features)
    # uninformative, or they would lead to data leakage
    df = df.drop(columns=['sub_type', 'text', 'sold_list_ratio', 'price_per_sqft_sold', 
                          'price_per_sqft_list', 'sold_vs_asking', 'listPrice', 
                          'size_tier', 'has_list_price', 'sqft_available',
                          'baths', 'baths_full', 'baths_full_calc'])
    
    # filling garage NA values with 0
    df['garage'] = df['garage'].fillna(0)

    # dropping has_garage (high correlation with garage and less informative [binary])
    df = df.drop(columns=['has_garage'])
    
    # filling missing values in with 1 story
    df['stories'] = df['stories'].fillna(1).astype(int)

    # only focusing on properties with 1 - 5 stories due to outliers
    df = df[(df['stories'] <= 5) & (df['stories'] > 0)].copy()

    # imputing sqft and beds with the medians of specific groupings
    df = df.dropna(subset=['sqft', 'beds'], how='all')
    df['sqft'] = df['sqft'].fillna(df.groupby('beds')['sqft'].transform('median'))
    df['beds'] = df['beds'].fillna(df.groupby('total_baths')['beds'].transform('median')).round()
    df = df.dropna(subset=['sqft', 'beds'])
    
    # removes the singular apartment row
    df = df[df['type'] != 'apartment'].copy()
    
    # focuses on the top 9 cities and removes rare cities
    threshold = 100
    city_counts = df['city'].value_counts()
    rare_cities = city_counts[city_counts < threshold].index
    df['city'] = df['city'].replace(rare_cities, 'Other')
    df = df.drop(columns=['zip_code'])
    df_final = df[df['city'] != 'Other'].copy()
    
    # returning the final data used for training
    return df_final
    