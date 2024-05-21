# Metryki oceny modeli neuronowych

- **RMSE** - Root Mean Squared Error
  - Wzór: $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$

- **nRMSE** - normalized Root Mean Squared Error
  - Wzór: $nRMSE = \frac{RMSE}{max(y) - min(y)}$

- **MAPE** - Mean Absolute Percentage Error
  - Wzór: $MAPE = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y_i}}{y_i}\right|$

- **nMAPE** - normalized Mean Absolute Percentage Error
  - Wzór: $nMAPE = \frac{MAPE}{max(y) - min(y)}$