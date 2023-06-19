def prepare_data(data):
    
    import pandas as pd
    import re
    import numpy as np
    from datetime import datetime, timedelta
    
    __name__ = 'prepare_data'

    data = data[data['price'].notnull()].copy()
    
    columns = ['price', 'Area']
    binary_columns = [
        'hasElevator',
        'hasParking',
        'hasBars',
        'hasStorage',
        'hasAirCondition',
        'hasBalcony',
        'hasMamad',
        'handicapFriendly'
    ]

    # Convert columns to numeric
    data[columns] = data[columns].apply(lambda column: pd.to_numeric(column.astype(str).str.replace('[^\d.-]', '').str.replace(',', ''), errors='coerce'))

    # Replace missing values with average of 'price' column
    price_mean = data['price'].mean()
    data['price'].fillna(price_mean, inplace=True)

    # Convert columns to string and remove decimal '.0'
    data[['price', 'Area']] = data[['price', 'Area']].astype(str).applymap(lambda x: x.replace('.0', ''))

    # Replace empty strings with NaN
    data[['price', 'Area']] = data[['price', 'Area']].replace('', np.nan)
    
    
    data['room_number'] = data['room_number'].astype(str)  # Convert column to string type
    data['room_number'] = data['room_number'].str.extract(r'(\d+\.?\d*)').astype(float)


    def remove_non_text(text):
        cleaned_text = re.sub(r'[^א-ת\s]', '', str(text))
        cleaned_text = cleaned_text.replace('\n', '')
        return cleaned_text

    data['Street'] = data['Street'].apply(remove_non_text)
    data['city_area'] = data['city_area'].apply(remove_non_text)

    data.rename(columns=lambda x: x.strip(), inplace=True)

    data['description'] = data['description'].apply(remove_non_text)

    data['floor'] = data['floor_out_of'].str.split('מתוך').str[0].str.split().str[-1]
    data['floor'] = pd.to_numeric(data['floor'], errors='coerce')

    data['floor'] = data['floor'].astype(str)
    data['floor'] = data['floor'].str.replace('.0', '')
    data['floor'] = data['floor'].replace('', np.nan)
    data['floor'] = data['floor'].astype(float)
    data['floor'] = data['floor'].fillna(0).astype(int)

    data['total_floors'] = data['floor_out_of'].str.extract(r'מתוך (\d+)').astype(float).fillna(0).astype(int)

    data = data.drop('floor_out_of', axis=1)

    binary_values_map = {
        True: 1,
        False: 0,
        'TRUE': 1,
        'FALSE': 0,
        'yes': 1,
        'no': 0,
        'יש': 1,
        'אין': 0,
        'יש מחסן': 1,
        'אין מחסן': 0,
        'יש חנייה': 1,
        'אין חנייה': 0,
        'יש מעלית': 1,
        'אין מעלית': 0,
        'יש סורגים': 1,
        'אין סורגים': 0,
        'יש מיזוג אויר': 1,
        'אין מיזוג אויר': 0,
        'נגיש לנכים': 1,
        'לא נגיש לנכים': 0,
        'יש ממ"ד': 1,
        'אין ממ"ד': 0,
        'יש מרפסת': 1,
        'אין מרפסת': 0,
        'נגיש': 1,
        'לא נגיש': 0,
        'אין חניה': 0,
        'יש חניה': 1,
        'כן': 1,
        'לא': 0,
        'יש מיזוג אוויר': 1,
        'אין ממ״ד': 0,
        'יש ממ״ד': 1
    }

    data[binary_columns] = data[binary_columns].replace(binary_values_map).fillna(data[binary_columns])
    
    translation_dict = {
    'גמיש': 'flexible',
    'לא צויין': 'not defined',
    'מיידי': 'less_than_6 months'
    }

    data['entranceDate'] = data['entranceDate'].replace(translation_dict)

    today = datetime.now().date()  # Get today's date

    # Define the categorization function
    def categorize_date(cell):
        if cell not in ['flexible', 'not defined', 'less_than_6 months']:
            try:
                date_obj = datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S').date()
                difference = (today - date_obj).days  # Calculate the difference in days

                if difference < 180:
                    return 'less_than_6 months'
                elif 180 <= difference <= 365:
                    return 'months_6_12'
                else:
                    return 'above_year'
            except ValueError:
                return cell  # Return the original value if the conversion fails
        else:
            return cell  # Return the original value for cells that match the condition

    # Apply the categorization function to the "entranceDate" column
    data['entranceDate'] = data['entranceDate'].apply(categorize_date)

    data['City'] = data['City'].str.strip()
    data['City'] = data['City'].replace('נהרייה', 'נהריה')
    
    # Remove values containing the characters 'nan' from the 'Area' column
    data = data[~data['Area'].astype(str).str.contains('nan')]

        # Convert 'price' column to float64
    data['price'] = data['price'].astype('float64')

    # Convert '0/1' to int in hasElevator column
    
    data['hasElevator'] = data['hasElevator'].fillna(data['hasElevator'].median())
    data['hasElevator'] = data['hasElevator'].astype('float')

    # Convert '0/1' to int in hasParking column
    data['hasParking'] = data['hasParking'].fillna(data['hasParking'].median())
    data['hasParking'] = data['hasParking'].astype('float')

    # Convert '0/1' to int in hasBars column
    data['hasBars'] = data['hasBars'].fillna(data['hasBars'].median())
    data['hasBars'] = data['hasBars'].astype('float')
    
    # Convert '0/1' to int in hasStorage column
    data['hasStorage'] = data['hasStorage'].fillna(data['hasStorage'].median())
    data['hasStorage'] = data['hasStorage'].astype('float')
    
        # Convert '0/1' to int in hasAirCondition column
    data['hasAirCondition'] = data['hasAirCondition'].fillna(data['hasAirCondition'].median())
    data['hasAirCondition'] = data['hasAirCondition'].astype('float')
    
        # Convert '0/1' to int in handicapFriendly column
    data['hasMamad'] = data['hasMamad'].fillna(data['hasMamad'].median())
    data['hasMamad'] = data['hasMamad'].astype('float')

        # Convert '0/1' to int in hasBalcony column
    data['hasBalcony'] = data['hasBalcony'].fillna(data['hasBalcony'].median())
    data['hasBalcony'] = data['hasBalcony'].astype('float')

            # Convert '0/1' to int in handicapFriendly column
    data['handicapFriendly'] = data['handicapFriendly'].fillna(0)
    data['handicapFriendly'] = data['handicapFriendly'].astype('float')
      
        
    data['type'] = data['type'].replace(['דירת גג', 'מיני פנטהאוז'],'פנטהאוז').replace(['דופלקס', 'טריפלקס', 'דירת נופש'],'דירה').replace(['קוטג טורי', 'קוטג', 'דו משפחתי', 'בניין'],'בית פרטי').replace(['נחלה','מגרש'],'אחר')


    return data
