<h2>
  Classify fetal health in order to prevent child and maternal mortality.

  About this dataset
</h2>
 

Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress. 
The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births. 

Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented. 

In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more. 
 


<h2>Data</h2>
This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetricians into 3 classes: 

`Normal` 
`Suspect` 
`Pathological` 

Create a model to classify the outcome of Cardiotocogram (CTG) exam (which represents the well-being of the fetus). 
 


<h2>Notes</h2>
Note that this is a multiclass problem that can also be treated as regression (since the labels are progressive). 
 


<h2>Tasks</h2>

- Present a visual distribution of the 3 classes.</s> Is the data balanced? How do you plan to circumvent the data imbalance problem, if there is one? (Hint: stratification needs to be included.) (1) 

- Present 10 features that are most reflective to fetal health conditions (there are more than one way of selecting features and any of these are acceptable). Present if the correlation is statistically significant (using 95% and 90% critical values). (2) 

- Develop two different models to classify CTG features into the three fetal health states (I intentionally did not name which two models. Note that this is a multiclass problem that can also be treated as regression, since the labels are numeric.) (2+2)

- Visually present the confusion matrix (1) 
  - TP, FP, TN, FN 

- With a testing set of size of 30% of all available data, calculate (1.5) 
  - Area under the ROC Curve 
  - F1 Score 
  - Area under the Precision-Recall Curve 
  - (For both models in 3) 

- Without considering the class label attribute, use k-means clustering to cluster the records in different clusters and visualize them (use k to be 5, 10, 15). (2.5) 


<h2>What to submit?</h2> 

- a. all code 
 
- b. a document that adequately describes your approach for each of the questions. 
  
- c. visualizations and other results. 

 
