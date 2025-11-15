import kagglehub
import pandas as pd

path = kagglehub.dataset_download("carrie1/ecommerce-data")
print(path)

df = pd.read_csv(path + "/data.csv", encoding="ISO-8859-1")
df.to_parquet("ecommerce.parquet")