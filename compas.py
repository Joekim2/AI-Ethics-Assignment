import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from aif360.algorithms.postprocessing import EqOddsPostprocessing

# Load and prepare the data
def load_compas_data():
    # Load COMPAS dataset
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    
    # Data preprocessing similar to ProPublica's analysis
    df = df[(df['days_b_screening_arrest'] <= 30) & 
            (df['days_b_screening_arrest'] >= -30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    
    # Create binary features
    df['race'] = df['race'].apply(lambda x: 'Caucasian' if x == 'Caucasian' else 'Non-Caucasian')
    df['two_year_recid'] = df['two_year_recid'].astype(int)
    df['score_binary'] = (df['decile_score'] > 5).astype(int)
    
    return df[['race', 'two_year_recid', 'score_binary', 'decile_score', 'age', 'sex']]

# Load data
df = load_compas_data()
print("Dataset shape:", df.shape)
print("\nRace distribution:")
print(df['race'].value_counts())

# Create AIF360 dataset
dataset = BinaryLabelDataset(
    df=df,
    label_names=['two_year_recid'],
    protected_attribute_names=['race'],
    favorable_label=0,
    unfavorable_label=1
)

# Calculate bias metrics
privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # Non-Caucasian

metric = BinaryLabelDatasetMetric(
    dataset, 
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print("\n=== BIAS METRICS ===")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
print(f"Disparate Impact: {metric.disparate_impact():.4f}")
print(f"Average Odds Difference: {metric.average_abs_odds_difference():.4f}")

# Calculate performance metrics by race
def calculate_metrics_by_race(df):
    results = {}
    for race in df['race'].unique():
        subset = df[df['race'] == race]
        tn = len(subset[(subset['two_year_recid'] == 0) & (subset['score_binary'] == 0)])
        fp = len(subset[(subset['two_year_recid'] == 0) & (subset['score_binary'] == 1)])
        fn = len(subset[(subset['two_year_recid'] == 1) & (subset['score_binary'] == 0)])
        tp = len(subset[(subset['two_year_recid'] == 1) & (subset['score_binary'] == 1)])
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        results[race] = {
            'FPR': fpr,
            'FNR': fnr,
            'Precision': precision,
            'Count': len(subset)
        }
    return results

metrics = calculate_metrics_by_race(df)

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. False Positive Rate by Race
races = list(metrics.keys())
fpr_values = [metrics[race]['FPR'] for race in races]
colors = ['lightblue', 'salmon']

ax1.bar(races, fpr_values, color=colors, alpha=0.7)
ax1.set_title('False Positive Rate by Race', fontsize=14, fontweight='bold')
ax1.set_ylabel('False Positive Rate')
for i, v in enumerate(fpr_values):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 2. False Negative Rate by Race
fnr_values = [metrics[race]['FNR'] for race in races]
ax2.bar(races, fnr_values, color=colors, alpha=0.7)
ax2.set_title('False Negative Rate by Race', fontsize=14, fontweight='bold')
ax2.set_ylabel('False Negative Rate')
for i, v in enumerate(fnr_values):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 3. Score distribution by race
caucasian_scores = df[df['race'] == 'Caucasian']['decile_score']
non_caucasian_scores = df[df['race'] == 'Non-Caucasian']['decile_score']

ax3.hist([caucasian_scores, non_caucasian_scores], 
         bins=10, alpha=0.7, label=['Caucasian', 'Non-Caucasian'],
         color=['lightblue', 'salmon'])
ax3.set_title('COMPAS Score Distribution by Race', fontsize=14, fontweight='bold')
ax3.set_xlabel('COMPAS Decile Score')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Recidivism rate vs prediction
recidivism_rates = []
prediction_rates = []
for race in races:
    subset = df[df['race'] == race]
    recidivism_rate = subset['two_year_recid'].mean()
    prediction_rate = subset['score_binary'].mean()
    recidivism_rates.append(recidivism_rate)
    prediction_rates.append(prediction_rate)

x = np.arange(len(races))
width = 0.35
ax4.bar(x - width/2, recidivism_rates, width, label='Actual Recidivism', alpha=0.7)
ax4.bar(x + width/2, prediction_rates, width, label='Predicted High Risk', alpha=0.7)
ax4.set_title('Actual vs Predicted Risk by Race', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(races)
ax4.legend()

plt.tight_layout()
plt.savefig('compas_bias_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed metrics
print("\n=== DETAILED PERFORMANCE METRICS ===")
for race in metrics:
    print(f"\n{race}:")
    print(f"  False Positive Rate: {metrics[race]['FPR']:.3f}")
    print(f"  False Negative Rate: {metrics[race]['FNR']:.3f}")
    print(f"  Precision: {metrics[race]['Precision']:.3f}")
    print(f"  Sample Size: {metrics[race]['Count']}")

# Bias mitigation example
print("\n=== BIAS MITIGATION (Reweighing) ===")
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)

metric_transf = BinaryLabelDatasetMetric(
    dataset_transf, 
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print(f"Statistical Parity Difference (after mitigation): {metric_transf.statistical_parity_difference():.4f}")
print(f"Disparate Impact (after mitigation): {metric_transf.disparate_impact():.4f}")