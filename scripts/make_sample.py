import pandas as pd
df = pd.read_csv("data/raw/valle_central_ventas_semanales.csv", low_memory=False)
prods = (df[["id_producto","categoria"]].drop_duplicates()
         .groupby("categoria", group_keys=False).sample(n=3, random_state=649))
sample = df[df.id_producto.isin(prods.id_producto)]
sample.to_csv("data/sample/valle_central_sample.csv", index=False)
print(len(sample), sample.id_producto.nunique(), sample.id_sucursal.nunique())
