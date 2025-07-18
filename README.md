# Predicting Housing Prices

## Goals
In this project, I will be exploring the regression models we've learned about to build the best possible model to predict housing prices in Ames, Iowa.

The primary stakeholder is the saler's real estate agent who wants to know how much a house is likely to sell for so they can list it for the correct price and it sells quickly.  Obviously, we only have data on the final sale price (time on market, etc), so we want a model that predicts the final sale price accurately.

The focus of this model is accurate prediction, not interpretability.  The real eastate agent has already studied this and can explain it to the homeowner.  

## Repository Structure
This root directory holds this README.md file as well as the `requirements.txt` and `.gitignore` files (for smooth running of the repo).  

### `data` Directory
This is where all of the data is stored.  The original data is stored in a `raw` subdirectory to ensure that it is separate and untouched by all excitement.  

### `notebooks` Directory
The Jupyter notebooks used as explained in this table:

| Notebook Title | Purpose |
| --- | --- |
| `01_cleaning.ipynb` | Initial cleaning of the data, uses the `housing` module written to automate the cleaning and feature extraction processes |
| `02_eda.ipynb` | Details on each feature, correlations with target and each other are discussed, collinear features are identified |
| `03_model_intro.ipynb` | Initial round of modeling |
| `04_model_refine.ipynb` | Modeling refinement |


### `images` Directory
This directory holds all of the images generated to be shown in this document.



## Modeling

### Baseline
So the baseline model is the simple average of sale prices.  Another step beyond this is a model based on just the house's square footage and overall quality rating.  

### Linear Model
Looked at coefficients to determine the factors that impacted the sale price the most. 

![bar chart of select coefficients](/images/coefs_few.jpg)

Unsurprisingly, location was the strongest predictor of a houses's sale price.  Based on my personal experience buying and selling a house, the quality and condition of the house also play a large role in the house's sale price.  The `overall_qual` shows this most clearly, but the other aspects, such as whether the exterior had stone (a high quality material, at least compared to plastic) also increased the price.  

A house being adjacent or near a positive community feature (such as a park) so increased the sale price.

Location can negatively impact the sale price.  Houses near or adjacent to railroads had lower prices than equivalent houses elsewhere.  

### Ridge and LASSO Models
The ridge and LASSO models use regularization to minimize the coefficients of less important features.  This really helped me focus on the features that had the largest impact on the sale price.  Despite all this cool math, they made only a small improvement on the regular linear model for the same features.

The LASSO models had problems converging on a solution, so I dropped them from my modeling tools moving forward.

### KNN Model
The $k$ Nearest Neighbors (knn) models are also interesting, they are not very interpretabilty but have the possibility of having high predictability

### Tree Regressors and Bagging

This project really showcased the high variance tendancies of decision tree regressors.  My best tree-based model had an accuracy of over 98% on the training data which dropped to just under 88% for my validation data!


## Refining Models
After playing around with my parameters, I ran more rounds of modeling and scoring.  In the end, the best model was a ridge model with $\alpha = 2.8$.  

![actual vs. predicted sale price](/images/ridge_val_data.jpg)
Here the training data set is in blue and the validation data set is in orange.  The magenta line is where the actual and predicted sale price would be the same.  It's an accurate model houses with sale prices below $\$300,000$ but runs into problems above $\$300,000$ where it tends to underestimate the sale price.  This likely due to fewer data points and three houses that sold for a fraction of what they were worth (probably an in-family sale or soemthing like that).

## Summary of Things Learned

Despite all of the time I spent encoding the ordinal data, my simplier models tended to have higher $R^2$ scores.  I think this because the more complex models can easily become high-variance (or over-fit).

If I had more time, I would continue to refine the features I'm modeling with and focusing on the linear, ridge, knn, and tree models (LASSO and bagging are not well suited for this situation).

