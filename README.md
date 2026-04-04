# Housing Data Parser and Predictor
## RENEWAL PROBABILITY MODEL DESIGN, EVALUATION, AND TRANSFORMER CATEGORIZATION
`1-renewal_model.ipynb` parses through the datasets provided and develops a Gradient Boosting model for classification of lease renewable decisions as well as prediction of the probability of renewal. This code utilizes *sci-kit learn* for the models and *pandas* for the data processing.

`2-reason_categorization.ipynb` parses the database files consisting of chat logs between residents and assistants. It attempts to analyze the chats and categorize the reason for not renewing leases into one of four categories.
1. Rent Price
2. Maintenance Issues
3. Life Changes (Jobs, Moving)
4. Questions (Uncertain chat logs)
Arriving at these categories follows a multistep analysis process studing semantic similarities between sentences utilizing sentence transformers modules. This particular notebook utilizes `gte-base`.

## Dependencies
- pandas
- numpy
- matplotlib
- sklearn
- pytorch
- torchvision
- transformers
- sentence_transformers

## Features
This code is designed to accomplish three overarching tasks of processing the housing data, developing benchmark models, and then a final Gradient Boosted Classifier model to run predictions on renewal probabilities.

Afterwards, a rough three step analysis is conducted in order to understand the categories at play in the housing market renewal decisions. All semantic encodings are done using `gte-base`. Sentences are formed for each `'resident_id`' by concatenating all of their chat logs into one big sentence. This likely introduces extraneous and misleading information into the sentences. `'user_msgs'` contains all the concatenated sentences.


### Preprocessing of Housing Data
Housing data provided is processed to be applied to train a model. The following steps are taken to preprocess the data:
- `'msg_count'` counts the number of messages sent in the chat logs to gauge engagement per `'resident_id'`.
- `'pet_details'` are converged to integers, ignoring the types of pets.
- `'referrals'` are encoded using one-hot embedding.
- All dates are converted to numerics where the integer is years and decimals indicate decimals of years (2026.5 is the middle day of 2026).
- `'ADA_support`' are encoded using binary encoding.
- Within the maintenance data, `'avg_completion_time'` and `'avg_completion_diff'` are calculated, representing the average time it takes for a maintenance ticket to be completed and the average different between the target and actual completion time (units in years). If `'completion_time'` is empty, we assume the job was never completed. Missing values are replaced by the median. All of the rest of the data are thrown away and compressed in `'number_of_requests'` and `'number_of_completed_requests'` indicating total number of requests and total number of completed requests respectively per `'unit_number'`. We also calculate `'percent_completed_requests'` which is the percentage of requests which are marked as completed.
- `'occupancy_count'` is computed to figure out household size in each unit.
- `'rent_diff'` and `'percent_rent_diff'` are calculated showing the absolute and relative percentage of rent increase to the next lease. An outlier analysis is ran on `'percent_rent_diff'` using the Interquartile Range (IQR) to perform data winsorization.
- `'lease_length'` and `'lease_renew_window'` are calcualted showing the length of the last lease and the time between first contact regarding the next lease and when the last lease ends (units in years).

All of this data is aggregated by `'unit_number'` and `'lease_end_date'`. The former is because lease renewal depends on the household and not the individual. The latter is incase multiple datapoints exist for different tenants in the same unit. The features are aggregated appropriately.

### Model Development
In order to handle tabular data, we opt for a Gradient Boosting model. For each model (except the Raw Gradient Boosting Model), a permutation importance analysis is conducted on each feature in order to remove unimportant features. Each model is then evaluated using k-Fold Cross Validation to determine accuracy (number of hits) and Area Under the Curve (AUC) analyses. Accuracy is self explanatory and AUC is a good way to test the power of the model itself. The main model is a Gradient Boosting Classifier (GBC) model trained on our processed and pruned feature space. Four other models are developed for benchmarking:
1. Minimum Gradient Boosting Classifier (MIN): This model is the same Gradient Boosting Classifier model trained on only the three most important features indicated through post-calculation analysis. These are `'percentage_rent_diff'`, `'number_of_requests'`, and `'age'`. This is intended to show how good a model using minimal features can be.
2. Raw Gradient Boosting Classifier (RAW): This model is the same Gradient Boosting Classifier model trained using unprocessed data, wherever there is numerical data, and no new features are introduced. Features are not pruned. The purpose is to demonstrate what would happen if we did no preprocessing.
3. Logistic Regression (LOG): Intended to show what a simple linear model can accomplish given the same dataset.
4. Zero-R (ZR): Simply chooses the most frequent classification. Provides a bare benchline.
5. Weighted Coin Flip (WCF): Randomly samples classification based on occurences in the training set. Provides a bare benchline.
Analysis is done using k-Fold Cross Validation with 5 folds is conducted with the following metrics
```
--- MODEL  RESULTS --- 
Model	Accur.	AUC
GBC	0.8569	0.8866
MIN	0.8371	0.8848
RAW	0.7796	0.8545
LOG	0.6784	0.7482
Z-R	0.5595	0.5000
WCF	0.4934	0.5000
```
This indicates that we are outdoing the bare benchlines, while also showing that data preprocessing improved the process. It is seen that most of the accuracy can be achieved using the three most important features, but the rest of the features do help.

### Model Deployment
The Gradient Boosted Classifier model is using to predict the `'renewal_decision'` of the test dataset, as well as the `'probability_of_renewal'`. The output is saved as `'test_predictions.csv'` aloing with `'resident_id'` and `'unit_number'`.


### k-Means Clustering
A k-means clustering analysis is conducted first to break the text into rough groups. k = 10 and k = 4 are attempted. From this the rough groups outlined above are identified.

### Projection Analysis
2D plots are analyzed along the dimensions corresponding to the categories outlined above. We find strong correlation between all of these categories, though some are slightly less correlated. We also attempt to check the happiness of the sentences and the confusion of the sentences. This shows some correlation with the categories, but not always. This is a good starting point for future PCA analyses.

Further analysis of projecting the sentences into category embeddings (defined by encoding example sentences for each category). It shows that all four reasons are strongly present in all sentences, indicating perhaps there is redundancy in our choice of categories, as we saw from the strong correlation.

### Cosine Similarity Scores

Cosine similarity is calculated for its popular usage to determine semantic similarities between embeddings. The consine similarities between each chat log and the representative category example is calculated and the chat logs for each resident is assigned a category based on the most similar category. We find that this is almost always rent pricing. The total count is presented here:
```
--- TOTAL CATEGORY COUNT ---
reason
Rent           363
Life             6
Question         5
Maintenance      2
```
The results are saved as `category_predictions.csv`.
