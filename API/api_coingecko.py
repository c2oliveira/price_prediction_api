url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30'
response = requests.get(url)
if response.status_code == 200:
    data_crypto = response.json()
else:
    raise Exception("Erro na requisição da API do CoinGecko.")
  prices = data_crypto['prices']
df_crypto = pd.DataFrame(prices, columns=['timestamp', 'price'])
# Convertendo o timestamp para data
df_crypto['date'] = pd.to_datetime(df_crypto['timestamp'], unit='ms').dt.date

# Agrupando por data e pegando o preço de fechamento (último valor do dia)
df_daily = df_crypto.groupby('date').agg({'price': 'last'}).reset_index()

# Salvando os dados em um arquivo CSV
df_daily.to_csv('crypto_prices.csv', index=False)
print("\nDados históricos de criptomoedas salvos em 'crypto_prices.csv'.")

# Criando a variável alvo: se o preço do próximo dia é maior (1) ou menor (0) que o dia atual
df_daily['target'] = (df_daily['price'].shift(-1) > df_daily['price']).astype(int)
df_daily = df_daily.dropna().reset_index(drop=True)

# Criação de uma feature adicional: variação do preço em relação ao dia anterior
df_daily['price_diff'] = df_daily['price'].diff()
df_daily = df_daily.dropna().reset_index(drop=True)

# Seleção das features e do alvo para o modelo
X_crypto = df_daily[['price', 'price_diff']]
y_crypto = df_daily['target']

# Divisão dos dados em treino e teste
X_train_crypto, X_test_crypto, y_train_crypto, y_test_crypto = train_test_split(X_crypto, y_crypto, test_size=0.3, random_state=42)

# Treinamento de um modelo de regressão logística para classificação
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

modelo_crypto = LogisticRegression()
modelo_crypto.fit(X_train_crypto, y_train_crypto)

# Previsão e avaliação da acurácia
y_pred_crypto = modelo_crypto.predict(X_test_crypto)
acc_crypto = accuracy_score(y_test_crypto, y_pred_crypto)
print("\n----- Previsão de Criptomoedas -----")
print("Acurácia do modelo:", acc_crypto)

# Visualizando a importância das variáveis (coeficientes do modelo)
coef_df = pd.DataFrame({'Feature': X_crypto.columns, 'Coeficiente': modelo_crypto.coef_[0]})
print(coef_df)

plt.figure(figsize=(6, 4))
sns.barplot(data=coef_df, x='Feature', y='Coeficiente')
plt.title("Importância das Variáveis na Classificação")
plt.ylabel("Coeficiente")
plt.show()
