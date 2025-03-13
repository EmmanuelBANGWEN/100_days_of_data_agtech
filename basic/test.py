import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# data = np.array([
#     [1,2,3],
#     [4,5,6]
# ])
# print(data)
# arr = np.array([1, 2, 3])
# arr1 = np.array([[1, 2, 3, 2], [1, 3, 4, 2]])
# print(arr1[1, 2])
# print(arr1 + 2)
# print(arr1.ndim)
# print(arr1.size)
# print(arr1.shape)
# print(arr1)
# df = pd.DataFrame(arr1)
# print(df)
# df1 = pd.read_csv('area_et_value.csv')
# df1 = pd.read_csv('pesticide/rainfall.csv')
# df1 = pd.read_csv('pesticide/temp.csv')
# df1 = pd.read_csv('pesticide/yield_df.csv')
# df1 = pd.read_csv('pesticide/yield.csv')
# df1 = pd.read_csv('dataset_agtech.csv')
# grouped = df1['Value']
# print(df3['Value'].describe())
# print(grouped.describe())
# array = np.array(grouped)
# area = df1['Year']
# plt.scatter(area, grouped)
# type(array)
# plt.show()
# print(df1.head(200))
# df1_cout_production = df3['Value']
# df1_prix_vente = df3['Area'].index[6]
# plt.subplot(1, 2, 1)
# df2 = df1.groupby("Area")["Value"].mean()

# plt.hist(df1_cout_production, bins=5)
# plt.bar(df1_prix_vente, df1_cout_production)
# df2 = df1[['Area'],['Value']]
# plt.show()
# df2.to_csv('area_et_value.csv')
# print(df1)

# print(df1['Type de Sol'])
# print(df1.iloc[1])
# df1_grouped = df1['Surface Cultivée (ha)'].sum()
# print(df1_grouped)
# df1 = df1.filln
# x = [1, 2, 3, 4, 5]
# y = [1, 4, 9, 16, 25]

# plt.plot(x, y, color='green', linestyle='--', marker='o')
# plt.title('graphique test')
# plt.xlabel('Valeur X')
# plt.ylabel('Valeur Y')
# data = [1, 2, 2, 3, 3, 3, 4, 5]
# plt.hist(data, bins=5)

# plt.show()


class Calcule_Rendement:
    def __init__(self, culture, zone):
        self.culture = culture
        self.zone = zone
    def calcul(self, culture, zone,quantite_per_ha, superficie):
        rendement = quantite_per_ha * superficie

        return f"""

        vous avez cultivés une quantité de {quantite_per_ha} kg  par hectare
        la culture : {culture} dans le zone de : {zone} sur une superficie de {superficie} ha 
        et vous avez obtenue une rendement de : {rendement} kg
                    
                    """

plante = Calcule_Rendement("maïs", "dschang")
print(plante.calcul("maïs", "dschang" ,2, 20))
