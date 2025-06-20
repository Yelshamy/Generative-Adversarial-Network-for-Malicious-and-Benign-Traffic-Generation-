import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
import seaborn as sns


dataset = pd.read_csv("Y:/Uni/Sem6/Modelling and Simulation/Dataset.csv")
sample_da = []

# Index counter
i = 0

# Append rows to the array until we reach 100,000 or the dataset ends
while i < 100000 and i < len(dataset):
    row = dataset.iloc[i].to_dict()  # Convert row to dictionary (optional)
    sample_da.append(row)
    i += 1

sample_ds = pd.DataFrame(sample_da)
sample_ds.head()
#print(sample_ds)
# Split the 'id.orig_h' IP column into four octets
ip_split = sample_ds['id.orig_h'].str.split('.', expand=True)
ip_split.columns = ['IP 1', 'IP 2', 'IP 3', 'IP 4']
ip_split = ip_split.astype(int)
# Concatenate the new columns to the original DataFrame
df_with_ip_parts = pd.concat([sample_ds, ip_split], axis=1)
# Preview the result
#print(df_with_ip_parts[['id.orig_h', 'IP 1', 'IP 2', 'IP 3', 'IP 4']].head())
# Make a copy of the dataset to apply label encoding
label_encoded_df = sample_ds.copy()
for col in sample_ds.columns:
    null_count = sample_ds[col].isnull().sum()
    if null_count > 0:
        print("Column", col, " has ", null_count, " null values")
    inf_count = (sample_ds[col] == float('inf')).sum()
    if inf_count > 0:
        print("Column", col, " has ", inf_count, " infinite values")

for col in sample_ds.columns:
    if sample_ds[col].dtype == 'object':
        print("Column", col, " has ", dataset[col].nunique(), " categories")
    else:
        print("Column", col, " is not categorical")
label_enc = LabelEncoder()
categorical_cols = ['proto']  # Add 'service', 'flag' if they exist in your dataset
for col in categorical_cols:
    sample_ds[col] = label_enc.fit_transform(sample_ds[col])

sample_ds = pd.get_dummies(sample_ds, columns=['proto'], prefix='proto')
#print(sample_ds)

# Example: Automatically detect numerical columns (excluding IPs and categories)
numerical_cols = sample_ds.select_dtypes(include=['float64', 'int64']).columns.tolist()
scaler = StandardScaler()
standardized_data = scaler.fit_transform(sample_ds[numerical_cols])
#print(standardized_data)
#----------------------------------------------------------------------------------------------------------------------------
label_column = sample_ds['label']
print(label_column)
real_traffic = []
fake_traffic = []
for i, label in enumerate(label_column):
    if isinstance(label, str) and label.lower() == "benign":
        real_traffic.append(i)
        print(f"Row {i} is real traffic\n")
    elif isinstance(label, str) and label.lower() == "malicious":
        fake_traffic.append(i)
        print(f"Row {i} is fake traffic\n")

real_data = sample_ds.iloc[real_traffic]  # Only rows marked as 'Benign'
real_numerical_data = standardized_data[real_traffic, :]  # NumPy array of real traffic only
mean_vector = np.mean(real_numerical_data, axis=0)
cov_matrix = np.cov(real_numerical_data, rowvar=False)
# Generate synthetic samples from the same distribution
synthetic_samples = np.random.multivariate_normal(mean_vector, cov_matrix, size=1000)
synthetic_df = pd.DataFrame(synthetic_samples, columns=numerical_cols)
print(synthetic_df.head())
# Example: Compare distribution of 1 feature
sns.kdeplot(real_numerical_data[:, 0], label="Real", fill=True)
sns.kdeplot(synthetic_samples[:, 0], label="Fake", fill=True)
plt.legend()
plt.title("Feature Distribution: Real vs Fake")
plt.show()
#--------------------------------------------------------------------------------------------------------------------
target_size = 100000

# Generate synthetic samples (100000 rows) from multivariate normal
synthetic_samples_full = np.random.multivariate_normal(mean_vector, cov_matrix, size=target_size)

# Inverse transform to get original scale
original_scale_synthetic_full = scaler.inverse_transform(synthetic_samples_full)

# Create a DataFrame from the generated numerical data
synthetic_df_full = pd.DataFrame(original_scale_synthetic_full, columns=numerical_cols)

# Generate random IP address components and reconstruct IPs
ip_parts_full = np.random.randint(1, 255, size=(target_size, 4))
synthetic_df_full['id.orig_h'] = ['{}.{}.{}.{}'.format(*row) for row in ip_parts_full]

# Reconstruct protocol using random valid proto labels
if 'proto' in categorical_cols:
    proto_random = np.random.randint(0, len(label_enc.classes_), target_size)
    synthetic_df_full['proto'] = label_enc.inverse_transform(proto_random)

# Add labels (50% Benign, 50% Malicious)
synthetic_df_full['label'] = ['Benign' if i < target_size / 2 else 'Malicious' for i in range(target_size)]

# Organize columns for readability
columns_order = ['id.orig_h', 'proto', 'label'] + [col for col in synthetic_df_full.columns if col not in ['id.orig_h', 'proto', 'label']]
human_readable_df_full = synthetic_df_full[columns_order]

# Show first few rows
print(human_readable_df_full)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

