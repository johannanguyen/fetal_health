<h2>Task 1</h2>
We chose to show the data distribution between each of the three classes using Histograms. Since there was a data imbalance with most of the data maintaining a <b>fetal_health</b> classification of <b>1</b>, we use a Stratified Train-Test Split which splits the data to preserve the same proportions of data in each class.
<div>
  <img src="https://github.com/johannanguyen/fetal_health/blob/master/screenshots/distribution1.png" height="250" width="300">
  <img src="https://github.com/johannanguyen/fetal_health/blob/master/screenshots/distribution2.png" height="250" width="300">
  <img src="https://github.com/johannanguyen/fetal_health/blob/master/screenshots/distribution3.png" height="250" width="300">
</div>
<hr>

<h2>Task 2</h2>
To find the 10 most reflective features of fetal health, we calculated the Pearon's Correlation Coefficient of each feature using the <b>scipy.stats.pearsonr</b> function. Once that was calculated, we sorted their absolute values in ascending order and extracted the last 10 from the list. Since <b>scipy.stats.pearsonr</b> returns both the Correlation Coefficient and a pvalue, we compared the pvalue against <b>0.1</b> and <b>0.05</b> to prove that the correlation is statistically significant based on 90% and 95% confidence intervals.<br><br>

<b>uterine_contractions</b><br>
Correlation: 0.20489372127986774<br>
P-value: 1.3846747025644431e-21<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>histogram_median</b><br>
Correlation: 0.20503299554125773<br>
P-value: 1.2988891337541575e-21<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>histogram_variance</b><br>
Correlation: 0.20662962204272547<br>
P-value: 6.218926688568785e-22<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>mean_value_of_long_term_variability</b><br>
Correlation: 0.22679706542348083<br>
P-value: 3.33667040120377e-26<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>histogram_mean</b><br>
Correlation: 0.22698517650772904<br>
P-value: 3.030067468415955e-26<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>histogram_mode</b><br>
Correlation: 0.2504118122228778<br>
P-value: 9.305867662128216e-32<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>accelerations</b><br>
Correlation: 0.36406579288786367<br>
P-value: 1.2435259928479686e-67<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>percentage_of_time_with_abnormal_long_term_variability</b><br>
Correlation: 0.42614641992406527<br>
P-value: 1.5027669343305033e-94<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>abnormal_short_term_variability</b><br>
Correlation: 0.47119075284667544<br>
P-value: 5.9217105098313305e-118<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>

<b>prolongued_decelerations</b><br>
Correlation: 0.48485918632134833<br>
P-value: 8.854416080655132e-126<br>
Significance Test 90%: True<br>
Significane Test 95%: True<br><br>
<hr>

<h2>Task 3</h2>
<hr>

<h2>Task 4</h2>
<hr>

<h2>Task 5</h2>
<hr>

<h2>Task 6</h2>
