import csv

def read_data_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def extract_features_basic(data):
    return [row[:-2] for row in data]  # Exclude 'RainToday' and 'RainTomorrow' columns

def extract_features_numeric(data):
    numeric_data = []

    for row in data[1:]:
        numeric_values = []

        # Convert numeric columns to float or handle categorical values
        for i, value in enumerate(row[1:]):
            if value == 'NA':
                numeric_values.append(0.0)  # Replace missing value with 0.0
            else:
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    # Handle categorical values
                    if i in [5, 7, 8]:  # Indices of categorical columns
                        unique_values = ['N', 'E', 'S', 'W'] if i == 5 else ['N', 'E', 'S', 'W', 'NE', 'NW', 'SE', 'SW']
                        for j, unique_value in enumerate(unique_values):
                            if value == unique_value:
                                categorical_values = [1.0 if k == j else 0.0 for k in range(len(unique_values))]
                                numeric_values.extend(categorical_values)
                                break

        numeric_data.append(numeric_values)

    return numeric_data

def extract_features_categorical(data):
    return [[row[5], row[7], row[8]] for row in data[1:]]  # Include only categorical columns


# Example usage:
file_path = 'Weather/weather.csv'  # Replace with the path to your CSV file
data = read_data_from_csv(file_path)

# Splitting into x_train and x_test
train_size = int(0.8 * len(data))
x_train = data[:train_size]
x_test = data[train_size:]

x_train_basic = extract_features_basic(x_train)
x_train_numeric = extract_features_numeric(x_train)
x_train_categorical = extract_features_categorical(x_train)

x_test_basic = extract_features_basic(x_test)
x_test_numeric = extract_features_numeric(x_test)
x_test_categorical = extract_features_categorical(x_test)
