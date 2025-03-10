



# Multiscale Geographical Random Forest: A Novel Spatial ML Approach for Traffic Safety Modeling Integrating Street-View Semantic Visual Features

This repository presents a novel spatial Machine Learning (ML) framework thath extends the conventional Geographically Random Forest (GRF) by incorporating an adaptive multiscale bandwidth selection process. This approach calibrates each local model at its optimal spatial scale, thereby capturing spatial heterogeneity more effectively than fixed-scale methods.

Inspired by previous Geographical Random Forest (GRF) studies:

1. Georganos, S., Grippa, T., Niang Gadiaga, A., et al. (2021). Geographical random forests: a spatial extension of the random forest algorithm to address spatial heterogeneity in remote sensing and population modelling. Geocarto International, 36(2), 121-136.
2. Sun, K., Zhou, R. Z., Kim, J., & Hu, Y. (2024). PyGRF: An improved Python Geographical Random Forest model and case studies in public health and natural disasters. Transactions in GIS.

# Model Framework

![Model.jpg](https://github.com/PengfeiCui99/Multiscale-Geographical-Random-Forest-MGRF-/MGRF_codes/Model.jpg)

# Repository Structure

* MGRF_codes/
    * PyMGRF.py: Contains the Multiscale Geographical Random Forest (MGRF) functions.
    * Rum_MGRF.ipynb: An example notebook demonstrating how to use the framework for macro-level crash modeling.

# Quick Start

1. Training the model

Use the provided notebook to train the models on your dataset. For example:

```python
from MGRF_codes import PyMGRF

model = PyMGRF.PyMGRFBuilder(
    global_n_estimators=32, 
    local_n_estimators=70, 
    max_features=1/3, 
    band_width_min=30, 
    band_width_max=140, 
    target = "R2", # Optional target "MSE" or "Moran"
    num = 12, 
    kernel= "adaptive", 
    train_weighted=True, 
    predict_weighted=True, 
    resampled=True, 
    random_state=28)

# Fit model with training data
global_oob_prediction, local_oob_prediction = model.fit(X_train, y_train, train_coords)
print("Fitting OK!")
```

2. Testing the model

Evaluate model performance on a test dataset:

```python
predict_combined, predict_global, predict_local = model.predict(X_test, test_coords, predict_bw = 70, local_weight=0.8)
print("Test OK!")
predict_local = np.array(predict_local)
predict_global = np.array(predict_global)
```

3. Model Interpretation

Utilize **feature importance** for model interpretation to understand feature contributions.

```python
global_model, local_models = model.get_models()
# Global feature importance
global_feature_importances = global_model.feature_importances_
print("Global feature importance：", global_feature_importances)

# Local feature importance
# Show first local model
first_local_model = local_models[0]
print("First local model：", first_local_model)

# Feature importance of first local model
first_local_feature_importances = first_local_model.feature_importances_
print("Feature importance of first local model：", first_local_feature_importances)

for idx, model in enumerate(local_models):
    print(f"Feature importance of local model: {idx}", model.feature_importances_)
```


Utilize **SHAP** for model interpretation to understand feature contributions.

```python
# Creating SHAP interpreter
explainer = shap.TreeExplainer(global_model)
shap_values = explainer.shap_values(gdf_Miami[selected_columns])
plt.figure(figsize=(10, 8))
# Plot SHAP
shap.summary_plot(shap_values, gdf_Miami[selected_columns], show=False)
plt.tight_layout()
plt.show()
```



If you have any questions or encounter any issues, please feel free to contact the authors.

Happy modeling!
