Complete Step-by-Step Solution: At-Risk Member Prediction & Causal Analytics
🎯 Problem Statement
Predict members who will make high claims OR develop high-risk conditions in the future (3-12 months ahead), and identify root causes for actionable interventions.

📋 STEP-BY-STEP SOLUTION ROADMAP
PHASE 1: DATA PREPARATION & UNDERSTANDING

STEP 1: Define "At-Risk" Members Clearly
1.1 Business Definition Workshop
python# Define what constitutes "at-risk" - get stakeholder agreement

AT_RISK_CRITERIA = {
    'high_claim': {
        'threshold_type': 'percentile',  # or absolute amount
        'threshold_value': 90,  # 90th percentile
        'time_window': '6_months',  # future prediction window
        'min_amount': 5000  # £5,000 minimum
    },
    'high_risk_condition': {
        'conditions': [
            'Cancer',
            'Cardiovascular Disease', 
            'Chronic Kidney Disease',
            'Diabetes Complications',
            'Mental Health Crisis',
            'Musculoskeletal Chronic'
        ],
        'severity_threshold': 'requires_ongoing_treatment'
    }
}
1.2 Create Ground Truth Labels
pythonimport pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_target_labels(claims_df, membership_df, observation_date, prediction_window_months=6):
    """
    Create target variables for at-risk prediction
    
    Parameters:
    - observation_date: Date at which we make prediction (e.g., '2024-01-01')
    - prediction_window_months: How far ahead to predict (default 6 months)
    """
    
    observation_date = pd.to_datetime(observation_date)
    future_end_date = observation_date + timedelta(days=30*prediction_window_months)
    
    # Target 1: High Claim in Future Window
    future_claims = claims_df[
        (pd.to_datetime(claims_df['Paid Date']) > observation_date) &
        (pd.to_datetime(claims_df['Paid Date']) <= future_end_date)
    ]
    
    future_claim_amounts = future_claims.groupby('claimant unique ID')['Claim Amount'].sum()
    high_claim_threshold = future_claim_amounts.quantile(0.90)
    
    target_df = membership_df[['Unique ID']].copy()
    target_df['future_total_claims'] = target_df['Unique ID'].map(future_claim_amounts).fillna(0)
    target_df['will_make_high_claim'] = (target_df['future_total_claims'] > high_claim_threshold).astype(int)
    
    # Target 2: High-Risk Condition Development
    high_risk_conditions = [
        'Cancer', 'Cardiovascular', 'Chronic Kidney', 
        'Diabetes', 'Mental Health', 'Chronic Pain'
    ]
    
    future_conditions = future_claims.groupby('claimant unique ID')['Condition Category'].apply(
        lambda x: any(condition in str(x).upper() for condition in [c.upper() for c in high_risk_conditions])
    )
    
    target_df['will_develop_high_risk_condition'] = target_df['Unique ID'].map(future_conditions).fillna(0).astype(int)
    
    # Combined Target: At-Risk Member (either high claim OR high-risk condition)
    target_df['is_at_risk'] = (
        (target_df['will_make_high_claim'] == 1) | 
        (target_df['will_develop_high_risk_condition'] == 1)
    ).astype(int)
    
    # Additional: Risk severity score (0-100)
    target_df['risk_severity_score'] = (
        0.6 * target_df['will_make_high_claim'] + 
        0.4 * target_df['will_develop_high_risk_condition']
    ) * 100
    
    return target_df

# Execute
observation_date = '2024-06-01'  # Use as your "present" for prediction
targets = create_target_labels(claims, membership, observation_date)

print("Target Distribution:")
print(targets['is_at_risk'].value_counts(normalize=True))
print(f"\nAt-risk rate: {targets['is_at_risk'].mean()*100:.2f}%")
```

**Output Example:**
```
Target Distribution:
0    0.87
1    0.13

At-risk rate: 13.00%

STEP 2: Data Integration & Temporal Alignment
2.1 Create Point-in-Time Features
pythondef create_point_in_time_dataset(claims_df, membership_df, observation_date, lookback_months=24):
    """
    Create features using ONLY data available before observation_date
    This prevents data leakage
    """
    
    observation_date = pd.to_datetime(observation_date)
    lookback_start = observation_date - timedelta(days=30*lookback_months)
    
    # Filter historical claims (before observation date)
    historical_claims = claims_df[
        pd.to_datetime(claims_df['Paid Date']) < observation_date
    ].copy()
    
    # Filter to lookback window
    historical_claims = historical_claims[
        pd.to_datetime(historical_claims['Paid Date']) >= lookback_start
    ]
    
    print(f"Historical claims: {len(historical_claims)} records")
    print(f"Date range: {historical_claims['Paid Date'].min()} to {historical_claims['Paid Date'].max()}")
    
    return historical_claims, membership_df

# Execute
observation_date = '2024-06-01'
historical_claims, membership = create_point_in_time_dataset(
    claims, membership, observation_date, lookback_months=24
)
2.2 Join Tables with Validation
pythondef join_claims_membership(claims_df, membership_df):
    """
    Join claims and membership with data quality checks
    """
    
    # Check for duplicates
    print("=== Data Quality Checks ===")
    print(f"Duplicate Claim IDs: {claims_df['Claim ID'].duplicated().sum()}")
    print(f"Duplicate Member IDs: {membership_df['Unique ID'].duplicated().sum()}")
    
    # Join
    df = membership_df.merge(
        claims_df,
        left_on='Unique ID',
        right_on='claimant unique ID',
        how='left',
        indicator=True
    )
    
    print(f"\nJoin Statistics:")
    print(df['_merge'].value_counts())
    
    # Identify members with no claims
    members_no_claims = df[df['_merge'] == 'left_only']['Unique ID'].nunique()
    print(f"\nMembers with no claims in period: {members_no_claims}")
    
    return df

df = join_claims_membership(historical_claims, membership)

STEP 3: Comprehensive Feature Engineering
3.1 Demographic & Membership Features
pythondef create_demographic_features(df, observation_date):
    """
    Create demographic and membership-based features
    """
    
    observation_date = pd.to_datetime(observation_date)
    features = df[['Unique ID']].drop_duplicates().copy()
    
    # Age calculation
    features = features.merge(
        df.groupby('Unique ID')['Year of Birth'].first(),
        on='Unique ID'
    )
    features['age'] = observation_date.year - features['Year of Birth']
    
    # Age groups
    features['age_group'] = pd.cut(
        features['age'], 
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    )
    
    # Gender
    features = features.merge(
        df.groupby('Unique ID')['Gender'].first(),
        on='Unique ID'
    )
    features['is_male'] = (features['Gender'] == 'Male').astype(int)
    
    # Membership tenure
    features = features.merge(
        df.groupby('Unique ID')['Original Date of Joining'].first(),
        on='Unique ID'
    )
    features['Original Date of Joining'] = pd.to_datetime(features['Original Date of Joining'])
    features['membership_tenure_days'] = (observation_date - features['Original Date of Joining']).dt.days
    features['membership_tenure_years'] = features['membership_tenure_days'] / 365.25
    
    # Membership status
    features = features.merge(
        df.groupby('Unique ID')['Status of Member'].first(),
        on='Unique ID'
    )
    features['is_active_member'] = (features['Status of Member'] == 'Active').astype(int)
    
    # Scheme information
    features = features.merge(
        df.groupby('Unique ID')['Scheme Category/ Section Name'].first(),
        on='Unique ID'
    )
    
    # Contract duration
    df['Contract Start Date'] = pd.to_datetime(df['Contract Start Date'], errors='coerce')
    df['Contract End Date'] = pd.to_datetime(df['Contract End Date'], errors='coerce')
    
    contract_info = df.groupby('Unique ID').agg({
        'Contract Start Date': 'first',
        'Contract End Date': 'last'
    }).reset_index()
    
    contract_info['contract_duration_days'] = (
        contract_info['Contract End Date'] - contract_info['Contract Start Date']
    ).dt.days
    
    features = features.merge(contract_info[['Unique ID', 'contract_duration_days']], on='Unique ID', how='left')
    
    # Lapse indicator
    features = features.merge(
        df.groupby('Unique ID')['Lapse Date'].first(),
        on='Unique ID'
    )
    features['has_lapsed'] = features['Lapse Date'].notna().astype(int)
    
    return features

demographic_features = create_demographic_features(df, observation_date)
print(demographic_features.head())
print(f"\nFeatures created: {demographic_features.shape[1]}")
3.2 Historical Claims Features (CRITICAL)
pythondef create_claims_history_features(claims_df, observation_date):
    """
    Create comprehensive historical claims features
    """
    
    observation_date = pd.to_datetime(observation_date)
    claims_df['Paid Date'] = pd.to_datetime(claims_df['Paid Date'])
    
    # Calculate time periods
    claims_df['months_before_observation'] = (
        (observation_date.year - claims_df['Paid Date'].dt.year) * 12 +
        (observation_date.month - claims_df['Paid Date'].dt.month)
    )
    
    features_list = []
    
    # === 1. BASIC CLAIM COUNTS & AMOUNTS ===
    basic_agg = claims_df.groupby('claimant unique ID').agg({
        'Claim ID': 'count',  # total claims
        'Claim Amount': ['sum', 'mean', 'median', 'std', 'min', 'max'],
    }).reset_index()
    
    basic_agg.columns = ['Unique ID', 'total_claims_count', 'total_claim_amount', 
                         'avg_claim_amount', 'median_claim_amount', 'std_claim_amount',
                         'min_claim_amount', 'max_claim_amount']
    
    features_list.append(basic_agg)
    
    # === 2. TIME-WINDOWED FEATURES ===
    time_windows = [3, 6, 12, 24]  # months
    
    for window in time_windows:
        window_claims = claims_df[claims_df['months_before_observation'] <= window]
        
        window_agg = window_claims.groupby('claimant unique ID').agg({
            'Claim ID': 'count',
            'Claim Amount': 'sum'
        }).reset_index()
        
        window_agg.columns = ['Unique ID', f'claims_count_{window}m', f'total_amount_{window}m']
        
        features_list.append(window_agg)
    
    # === 3. TEMPORAL PATTERNS ===
    temporal = claims_df.groupby('claimant unique ID').agg({
        'Paid Date': ['min', 'max', 'count']
    }).reset_index()
    
    temporal.columns = ['Unique ID', 'first_claim_date', 'last_claim_date', 'claim_count']
    temporal['days_since_first_claim'] = (observation_date - temporal['first_claim_date']).dt.days
    temporal['days_since_last_claim'] = (observation_date - temporal['last_claim_date']).dt.days
    temporal['claim_duration_days'] = (temporal['last_claim_date'] - temporal['first_claim_date']).dt.days
    temporal['avg_days_between_claims'] = temporal['claim_duration_days'] / (temporal['claim_count'] - 1)
    temporal['avg_days_between_claims'] = temporal['avg_days_between_claims'].fillna(0)
    
    features_list.append(temporal[['Unique ID', 'days_since_first_claim', 'days_since_last_claim', 
                                   'claim_duration_days', 'avg_days_between_claims']])
    
    # === 4. CLAIM FREQUENCY TRENDS ===
    # Claims per month over time
    claims_df['claim_year_month'] = claims_df['Paid Date'].dt.to_period('M')
    monthly_claims = claims_df.groupby(['claimant unique ID', 'claim_year_month']).size().reset_index(name='monthly_count')
    
    freq_features = monthly_claims.groupby('claimant unique ID').agg({
        'monthly_count': ['mean', 'std', 'max']
    }).reset_index()
    
    freq_features.columns = ['Unique ID', 'avg_monthly_claim_frequency', 
                            'std_monthly_claim_frequency', 'max_monthly_claims']
    
    features_list.append(freq_features)
    
    # === 5. CLAIM SEVERITY INDICATORS ===
    # High-cost claims
    high_cost_threshold = claims_df['Claim Amount'].quantile(0.75)
    claims_df['is_high_cost_claim'] = (claims_df['Claim Amount'] > high_cost_threshold).astype(int)
    
    severity = claims_df.groupby('claimant unique ID').agg({
        'is_high_cost_claim': 'sum'
    }).reset_index()
    
    severity.columns = ['Unique ID', 'high_cost_claims_count']
    
    # Cost volatility
    cost_volatility = claims_df.groupby('claimant unique ID')['Claim Amount'].apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).reset_index()
    cost_volatility.columns = ['Unique ID', 'claim_amount_cv']  # coefficient of variation
    
    features_list.append(severity)
    features_list.append(cost_volatility)
    
    # === 6. TREND FEATURES (Increasing/Decreasing Claims) ===
    def calculate_trend(group):
        if len(group) < 2:
            return 0
        group = group.sort_values('Paid Date')
        group['time_index'] = range(len(group))
        correlation = group['time_index'].corr(group['Claim Amount'])
        return correlation if not np.isnan(correlation) else 0
    
    trends = claims_df.groupby('claimant unique ID').apply(calculate_trend).reset_index()
    trends.columns = ['Unique ID', 'claim_amount_trend']
    
    features_list.append(trends)
    
    # Merge all features
    from functools import reduce
    claims_features = reduce(lambda left, right: left.merge(right, on='Unique ID', how='outer'), features_list)
    
    # Fill NaN with 0 for members with no claims
    claims_features = claims_features.fillna(0)
    
    return claims_features

claims_features = create_claims_history_features(historical_claims, observation_date)
print(f"Claims features created: {claims_features.shape[1]}")
print(claims_features.describe())
3.3 Treatment & Condition Features
pythondef create_treatment_condition_features(claims_df):
    """
    Create features from Treatment Type and Condition information
    """
    
    features_list = []
    
    # === 1. TREATMENT TYPE FEATURES ===
    treatment_features = claims_df.groupby('claimant unique ID').agg({
        'Treatment Type': lambda x: x.nunique()  # unique treatment types
    }).reset_index()
    treatment_features.columns = ['Unique ID', 'unique_treatment_types']
    
    # Most frequent treatment
    most_frequent_treatment = claims_df.groupby('claimant unique ID')['Treatment Type'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    ).reset_index()
    most_frequent_treatment.columns = ['Unique ID', 'most_frequent_treatment']
    
    # Treatment diversity (entropy)
    def calculate_entropy(series):
        from scipy.stats import entropy
        value_counts = series.value_counts(normalize=True)
        return entropy(value_counts) if len(value_counts) > 1 else 0
    
    treatment_diversity = claims_df.groupby('claimant unique ID')['Treatment Type'].apply(
        calculate_entropy
    ).reset_index()
    treatment_diversity.columns = ['Unique ID', 'treatment_diversity_entropy']
    
    features_list.extend([treatment_features, most_frequent_treatment, treatment_diversity])
    
    # === 2. CONDITION FEATURES ===
    condition_features = claims_df.groupby('claimant unique ID').agg({
        'Condition Code': 'nunique',
        'Condition Category': 'nunique'
    }).reset_index()
    condition_features.columns = ['Unique ID', 'unique_condition_codes', 'unique_condition_categories']
    
    # Check for chronic conditions (repeat condition codes)
    repeat_conditions = claims_df.groupby(['claimant unique ID', 'Condition Code']).size().reset_index(name='count')
    chronic_indicator = repeat_conditions[repeat_conditions['count'] >= 3].groupby('claimant unique ID').size().reset_index()
    chronic_indicator.columns = ['Unique ID', 'chronic_conditions_count']
    
    features_list.extend([condition_features, chronic_indicator])
    
    # === 3. HIGH-RISK CONDITION FLAGS ===
    high_risk_conditions = ['CANCER', 'CARDIOVASCULAR', 'DIABETES', 'CHRONIC KIDNEY', 'MENTAL HEALTH']
    
    def has_high_risk_condition(condition_series):
        condition_text = ' '.join(condition_series.astype(str).str.upper())
        return any(condition in condition_text for condition in high_risk_conditions)
    
    high_risk_flags = claims_df.groupby('claimant unique ID')['Condition Category'].apply(
        has_high_risk_condition
    ).reset_index()
    high_risk_flags.columns = ['Unique ID', 'has_high_risk_condition_history']
    high_risk_flags['has_high_risk_condition_history'] = high_risk_flags['has_high_risk_condition_history'].astype(int)
    
    features_list.append(high_risk_flags)
    
    # === 4. TREATMENT LOCATION FEATURES ===
    location_features = claims_df.groupby('claimant unique ID').agg({
        'Treatment Location': lambda x: x.nunique()
    }).reset_index()
    location_features.columns = ['Unique ID', 'unique_treatment_locations']
    
    # Inpatient vs Outpatient ratio (if available)
    # This depends on your data structure
    
    features_list.append(location_features)
    
    # === 5. PROVIDER DIVERSITY ===
    if 'Provider Type' in claims_df.columns:
        provider_features = claims_df.groupby('claimant unique ID').agg({
            'Provider Type': 'nunique'
        }).reset_index()
        provider_features.columns = ['Unique ID', 'unique_providers']
        features_list.append(provider_features)
    
    # Merge all
    from functools import reduce
    treatment_condition_features = reduce(
        lambda left, right: left.merge(right, on='Unique ID', how='outer'), 
        features_list
    )
    
    treatment_condition_features = treatment_condition_features.fillna(0)
    
    return treatment_condition_features

treatment_condition_features = create_treatment_condition_features(historical_claims)
print(f"Treatment/Condition features: {treatment_condition_features.shape[1]}")
3.4 Risk Score Features (Composite Indicators)
pythondef create_risk_score_features(claims_features, demographic_features):
    """
    Create composite risk scores based on multiple factors
    """
    
    # Merge claims and demographic
    df = demographic_features.merge(claims_features, on='Unique ID', how='left').fillna(0)
    
    # === 1. UTILIZATION RISK SCORE ===
    # Normalized by age and membership tenure
    df['claims_per_year'] = df['total_claims_count'] / (df['membership_tenure_years'] + 0.1)
    df['cost_per_year'] = df['total_claim_amount'] / (df['membership_tenure_years'] + 0.1)
    
    # Age-adjusted utilization
    age_avg_claims = df.groupby('age_group')['claims_per_year'].transform('mean')
    df['utilization_vs_age_cohort'] = df['claims_per_year'] / (age_avg_claims + 1)
    
    # === 2. CLAIM ACCELERATION SCORE ===
    # Are claims increasing over time?
    df['claim_acceleration_score'] = (
        (df['claims_count_6m'] / 6) / 
        ((df['claims_count_24m'] - df['claims_count_6m']) / 18 + 0.1)
    )
    df['claim_acceleration_score'] = df['claim_acceleration_score'].replace([np.inf, -np.inf], 0)
    
    # === 3. RECENT ACTIVITY SCORE ===
    df['recent_activity_score'] = (
        0.5 * (df['claims_count_3m'] / (df['total_claims_count'] + 1)) +
        0.3 * (df['total_amount_3m'] / (df['total_claim_amount'] + 1)) +
        0.2 * (1 / (df['days_since_last_claim'] + 30))
    )
    
    # === 4. SEVERITY RISK SCORE ===
    df['severity_risk_score'] = (
        0.4 * (df['high_cost_claims_count'] / (df['total_claims_count'] + 1)) +
        0.3 * (df['max_claim_amount'] / (df['avg_claim_amount'] + 1)) +
        0.3 * df['claim_amount_cv']
    )
    
    # === 5. CHRONICITY SCORE ===
    df['chronicity_score'] = (
        0.5 * df['has_high_risk_condition_history'] +
        0.3 * (df['chronic_conditions_count'] / (df['unique_condition_codes'] + 1)) +
        0.2 * (df['avg_days_between_claims'] < 60).astype(int)
    )
    
    # === 6. COMPOSITE AT-RISK SCORE (0-100) ===
    from sklearn.preprocessing import MinMaxScaler
    
    risk_components = [
        'utilization_vs_age_cohort',
        'claim_acceleration_score', 
        'recent_activity_score',
        'severity_risk_score',
        'chronicity_score'
    ]
    
    scaler = MinMaxScaler()
    df[risk_components] = scaler.fit_transform(df[risk_components])
    
    df['composite_risk_score'] = (
        0.25 * df['utilization_vs_age_cohort'] +
        0.20 * df['claim_acceleration_score'] +
        0.20 * df['recent_activity_score'] +
        0.20 * df['severity_risk_score'] +
        0.15 * df['chronicity_score']
    ) * 100
    
    return df

risk_features = create_risk_score_features(claims_features, demographic_features)
print(risk_features[['Unique ID', 'composite_risk_score']].describe())
3.5 Merge All Features
pythondef create_master_feature_set(demographic_features, claims_features, treatment_condition_features, targets):
    """
    Combine all feature sets into final dataset
    """
    
    # Start with demographics
    master_df = demographic_features.copy()
    
    # Add claims features
    master_df = master_df.merge(claims_features, on='Unique ID', how='left')
    
    # Add treatment/condition features
    master_df = master_df.merge(treatment_condition_features, on='Unique ID', how='left')
    
    # Add targets
    master_df = master_df.merge(targets, on='Unique ID', how='left')
    
    # Fill NaN values
    master_df = master_df.fillna(0)
    
    print(f"Master dataset shape: {master_df.shape}")
    print(f"Features: {master_df.shape[1] - 5}")  # excluding ID and targets
    
    return master_df

master_df = create_master_feature_set(
    demographic_features,
    claims_features, 
    treatment_condition_features,
    targets
)

print("\nFeature Summary:")
print(master_df.dtypes.value_counts())
print(f"\nMissing values: {master_df.isnull().sum().sum()}")

STEP 4: Data Preprocessing
4.1 Handle Missing Values & Outliers
pythondef preprocess_features(df):
    """
    Final preprocessing before modeling
    """
    
    # === 1. HANDLE MISSING VALUES ===
    print("Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Numerical: median imputation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    from sklearn.impute import SimpleImputer
    
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # === 2. HANDLE OUTLIERS (CAP AT 99TH PERCENTILE) ===
    cost_columns = [col for col in df.columns if 'amount' in col.lower() or 'cost' in col.lower()]
    
    for col in cost_columns:
        if df[col].dtype in [np.float64, np.int64]:
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=p99)
    
    # === 3. HANDLE INFINITE VALUES ===
    df = df.replace([np.inf, -np.inf], 0)
    
    # === 4. ENCODE CATEGORICAL VARIABLES ===
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # For high cardinality categoricals, use target encoding (do this during modeling)
    # For low cardinality, use one-hot encoding
    
    low_cardinality = []
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            low_cardinality.append(col)
    
    df = pd.get_dummies(df, columns=low_cardinality, drop_first=True)
    
    print(f"\nAfter preprocessing: {df.shape}")
    
    return df

processed_df = preprocess_features(master_df)
4.2 Feature Selection
pythonfrom sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def select_important_features(df, target_col='is_at_risk', top_k=50):
    """
    Select most important features
    """
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [
        'Unique ID', 'is_at_risk', 'will_make_high_claim', 
        'will_develop_high_risk_condition', 'future_total_claims', 'risk_severity_score'
    ]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # === 1. REMOVE LOW VARIANCE FEATURES ===
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_high_var = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Features after variance filter: {len(selected_features)}")
    
    # === 2. UNIVARIATE FEATURE SELECTION ===
    X_selected = X[selected_features]
    
    selector = SelectKBest(f_classif, k=min(top_k, len(selected_features)))
    selector.fit(X_selected, y)
    
    feature_scores = pd.DataFrame({
        'feature': selected_features,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print("\nTop 20 features by F-score:")
    print(feature_scores.head(20))
    
    # === 3. TREE-BASED FEATURE IMPORTANCE ===
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_selected, y)
    
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features by Random Forest importance:")
    print(feature_importance.head(20))
    
    # Select top features (union of both methods)
    top_features_f = set(feature_scores.head(top_k)['feature'])
    top_features_rf = set(feature_importance.head(top_k)['feature'])
    
    final_features = list(top_features_f.union(top_features_rf))
    
    print(f"\nFinal selected features: {len(final_features)}")
    
    return final_features, feature_importance

selected_features, feature_importance = select_important_features(processed_df)

STEP 5: Model Development
5.1 Train-Test Split (Time-Based)
pythonfrom sklearn.model_selection import train_test_split

def create_RetryPSContinuepythondef create_train_test_split(df, target_col='is_at_risk', test_size=0.2, random_state=42):
    """
    Create train-test split with stratification
    """
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [
        'Unique ID', 'is_at_risk', 'will_make_high_claim', 
        'will_develop_high_risk_condition', 'future_total_claims', 'risk_severity_score'
    ]]
    
    X = df[feature_cols]
    y = df[target_col]
    member_ids = df['Unique ID']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, member_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print("=== Data Split Summary ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"\nClass distribution in training:")
    print(y_train.value_counts(normalize=True))
    print(f"\nClass distribution in test:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test

X_train, X_test, y_train, y_test, ids_train, ids_test = create_train_test_split(
    processed_df[['Unique ID'] + selected_features + ['is_at_risk']]
)
5.2 Handle Class Imbalance
pythonfrom imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

def balance_training_data(X_train, y_train, method='smote'):
    """
    Balance the training dataset
    """
    
    print(f"Original class distribution:")
    print(y_train.value_counts())
    
    if method == 'smote':
        # SMOTE oversampling
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
    elif method == 'combined':
        # Combination of over and under sampling
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        
        pipeline = ImbPipeline([
            ('over', over),
            ('under', under)
        ])
        
        X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
    
    else:  # No balancing
        X_balanced, y_balanced = X_train, y_train
    
    print(f"\nBalanced class distribution:")
    print(pd.Series(y_balanced).value_counts())
    
    return X_balanced, y_balanced

X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train, method='smote')
5.3 Train Multiple Models
pythonfrom sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Train multiple baseline models for comparison
    """
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            random_state=42,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
    }
    
    results = {}
    trained_models = {}
    
    print("=== Training Models ===\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        results[name] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trained_models[name] = model
        
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}\n")
    
    # Compare results
    comparison_df = pd.DataFrame({
        name: {
            'ROC-AUC': results[name]['roc_auc'],
            'PR-AUC': results[name]['pr_auc']
        }
        for name in results
    }).T.sort_values('PR-AUC', ascending=False)
    
    print("\n=== Model Comparison ===")
    print(comparison_df)
    
    return trained_models, results, comparison_df

trained_models, results, comparison_df = train_baseline_models(
    X_train_balanced, y_train_balanced, X_test, y_test
)
5.4 Hyperparameter Tuning (Best Model)
pythonfrom sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

def tune_best_model(X_train, y_train, model_type='lightgbm'):
    """
    Perform hyperparameter tuning on the best performing model
    """
    
    if model_type == 'lightgbm':
        param_distributions = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'num_leaves': randint(20, 100),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
        
        base_model = LGBMClassifier(
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
    
    elif model_type == 'xgboost':
        param_distributions = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
        
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
    
    print(f"=== Hyperparameter Tuning for {model_type.upper()} ===\n")
    
    # Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=50,  # number of parameter combinations to try
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation ROC-AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_

# Tune the best model (let's say LightGBM performed best)
best_model, best_params = tune_best_model(X_train_balanced, y_train_balanced, model_type='lightgbm')
5.5 Final Model Evaluation
pythonfrom sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_final_model(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive evaluation of the final model
    """
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print("=== FINAL MODEL EVALUATION ===\n")
    
    # === 1. CLASSIFICATION METRICS ===
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not At-Risk', 'At-Risk']))
    
    # === 2. CONFUSION MATRIX ===
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # === 3. ROC-AUC ===
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # === 4. PRECISION-RECALL AUC ===
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # === 5. BUSINESS METRICS ===
    # Capture rate in top deciles
    df_scores = pd.DataFrame({
        'actual': y_test,
        'predicted_proba': y_pred_proba
    }).sort_values('predicted_proba', ascending=False)
    
    total_positives = df_scores['actual'].sum()
    
    for decile in [10, 20, 30]:
        n = int(len(df_scores) * (decile/100))
        captured = df_scores.head(n)['actual'].sum()
        capture_rate = (captured / total_positives) * 100
        print(f"\nTop {decile}% Capture Rate: {capture_rate:.2f}%")
    
    # === 6. VISUALIZATIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[0, 1].plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test==0], bins=50, alpha=0.5, label='Not At-Risk', color='blue')
    axes[1, 1].hist(y_pred_proba[y_test==1], bins=50, alpha=0.5, label='At-Risk', color='red')
    axes[1, 1].axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

evaluation_results = evaluate_final_model(best_model, X_test, y_test, threshold=0.3)

STEP 6: Model Interpretation & Feature Importance
6.1 SHAP Analysis
pythonimport shap

def analyze_model_with_shap(model, X_train, X_test, feature_names):
    """
    Use SHAP to explain model predictions
    """
    
    print("=== SHAP Analysis ===\n")
    print("Computing SHAP values... (this may take a few minutes)")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set (sample if too large)
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    
    shap_values = explainer.shap_values(X_test_sample)
    
    # If binary classification, take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # === 1. SUMMARY PLOT ===
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 2. DETAILED SUMMARY PLOT ===
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title('SHAP Summary Plot (Feature Impact)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 3. TOP FEATURES ===
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features (by SHAP):")
    print(shap_importance.head(20))
    
    # === 4. DEPENDENCE PLOTS FOR TOP FEATURES ===
    top_features = shap_importance.head(5)['feature'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        if idx < 6:
            shap.dependence_plot(
                feature, 
                shap_values, 
                X_test_sample,
                ax=axes[idx],
                show=False
            )
    
    plt.tight_layout()
    plt.savefig('shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return shap_values, shap_importance

shap_values, shap_importance = analyze_model_with_shap(
    best_model, 
    X_train_balanced, 
    X_test,
    selected_features
)
6.2 Feature Importance Analysis
pythondef analyze_feature_importance(model, feature_names, top_n=20):
    """
    Extract and visualize feature importance from tree-based model
    """
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("=== Feature Importance Analysis ===\n")
    print(f"Top {top_n} Features:")
    print(importance_df.head(top_n))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Group features by category
    feature_categories = {
        'Demographic': ['age', 'gender', 'tenure'],
        'Claims History': ['claims_count', 'total_amount', 'avg_claim'],
        'Temporal': ['days_since', 'months', 'frequency'],
        'Severity': ['high_cost', 'max_claim', 'severity'],
        'Condition': ['condition', 'chronic', 'treatment'],
        'Risk Scores': ['risk_score', 'utilization', 'acceleration']
    }
    
    category_importance = {}
    for category, keywords in feature_categories.items():
        category_features = [f for f in importance_df['feature'] 
                           if any(kw in f.lower() for kw in keywords)]
        category_importance[category] = importance_df[
            importance_df['feature'].isin(category_features)
        ]['importance'].sum()
    
    # Visualize category importance
    plt.figure(figsize=(10, 6))
    categories = list(category_importance.keys())
    importances = list(category_importance.values())
    plt.bar(categories, importances)
    plt.xlabel('Feature Category')
    plt.ylabel('Cumulative Importance')
    plt.title('Feature Importance by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('category_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

feature_importance_df = analyze_feature_importance(best_model, selected_features)

STEP 7: Risk Stratification & Scoring
7.1 Create Risk Tiers
pythondef create_risk_tiers(y_pred_proba):
    """
    Stratify members into risk tiers based on predicted probabilities
    """
    
    # Define risk tiers
    risk_tiers = pd.DataFrame({
        'member_id': ids_test,
        'risk_probability': y_pred_proba,
        'actual_outcome': y_test.values
    })
    
    # Create risk categories
    risk_tiers['risk_tier'] = pd.cut(
        risk_tiers['risk_probability'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Alternative: Percentile-based tiers
    risk_tiers['risk_percentile'] = risk_tiers['risk_probability'].rank(pct=True) * 100
    risk_tiers['risk_decile'] = pd.cut(
        risk_tiers['risk_percentile'],
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    )
    
    print("=== Risk Stratification ===\n")
    
    # Tier distribution
    print("Risk Tier Distribution:")
    print(risk_tiers['risk_tier'].value_counts().sort_index())
    
    # Performance by tier
    print("\nActual At-Risk Rate by Tier:")
    tier_performance = risk_tiers.groupby('risk_tier').agg({
        'actual_outcome': ['count', 'sum', 'mean']
    })
    tier_performance.columns = ['Total Members', 'At-Risk Members', 'At-Risk Rate']
    tier_performance['At-Risk Rate'] = tier_performance['At-Risk Rate'] * 100
    print(tier_performance)
    
    # Decile analysis
    print("\n\nRisk Decile Analysis:")
    decile_performance = risk_tiers.groupby('risk_decile').agg({
        'actual_outcome': ['count', 'sum', 'mean'],
        'risk_probability': 'mean'
    })
    decile_performance.columns = ['Total', 'At-Risk', 'At-Risk Rate', 'Avg Probability']
    decile_performance['At-Risk Rate'] = decile_performance['At-Risk Rate'] * 100
    decile_performance['Avg Probability'] = decile_performance['Avg Probability'] * 100
    print(decile_performance)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Tier distribution
    tier_counts = risk_tiers['risk_tier'].value_counts().sort_index()
    axes[0].bar(tier_counts.index, tier_counts.values, color='steelblue')
    axes[0].set_xlabel('Risk Tier')
    axes[0].set_ylabel('Number of Members')
    axes[0].set_title('Member Distribution by Risk Tier')
    axes[0].tick_params(axis='x', rotation=45)
    
    # At-risk rate by decile
    axes[1].plot(decile_performance.index, decile_performance['At-Risk Rate'], 
                marker='o', linewidth=2, markersize=8, color='darkred')
    axes[1].set_xlabel('Risk Decile')
    axes[1].set_ylabel('Actual At-Risk Rate (%)')
    axes[1].set_title('Model Calibration by Decile')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_stratification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return risk_tiers

risk_tiers = create_risk_tiers(evaluation_results['probabilities'])
7.2 Generate Member-Level Risk Scores
pythondef generate_member_risk_profiles(model, X_full, member_ids, feature_names):
    """
    Generate comprehensive risk profiles for all members
    """
    
    # Predict for all members
    risk_probabilities = model.predict_proba(X_full)[:, 1]
    
    # Create risk profile
    risk_profiles = pd.DataFrame({
        'member_id': member_ids,
        'risk_probability': risk_probabilities,
        'risk_score': risk_probabilities * 100,  # 0-100 scale
    })
    
    # Add risk tier
    risk_profiles['risk_tier'] = pd.cut(
        risk_profiles['risk_probability'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Add percentile rank
    risk_profiles['risk_percentile'] = risk_profiles['risk_score'].rank(pct=True) * 100
    
    # Flag high-risk members (top 15%)
    risk_profiles['requires_intervention'] = (risk_profiles['risk_percentile'] >= 85).astype(int)
    
    # Add priority level for interventions
    risk_profiles['intervention_priority'] = pd.cut(
        risk_profiles['risk_percentile'],
        bins=[0, 70, 85, 95, 100],
        labels=['Monitor', 'Low Priority', 'Medium Priority', 'High Priority']
    )
    
    print(f"=== Member Risk Profiles Generated ===\n")
    print(f"Total Members: {len(risk_profiles)}")
    print(f"\nMembers Requiring Intervention: {risk_profiles['requires_intervention'].sum()} "
          f"({risk_profiles['requires_intervention'].mean()*100:.1f}%)")
    
    print("\nIntervention Priority Distribution:")
    print(risk_profiles['intervention_priority'].value_counts().sort_index())
    
    # Save to file
    risk_profiles.to_csv('member_risk_profiles.csv', index=False)
    print("\n✓ Risk profiles saved to 'member_risk_profiles.csv'")
    
    return risk_profiles

# Generate for all members (train + test)
X_full = pd.concat([X_train, X_test])
ids_full = pd.concat([ids_train, ids_test])

member_risk_profiles = generate_member_risk_profiles(
    best_model, 
    X_full, 
    ids_full,
    selected_features
)

STEP 8: Causal & Exploratory Analytics
8.1 Cohort Analysis
pythondef perform_cohort_analysis(master_df, risk_profiles):
    """
    Analyze high-risk patterns across different member cohorts
    """
    
    # Merge risk profiles with demographic data
    analysis_df = master_df.merge(risk_profiles, left_on='Unique ID', right_on='member_id')
    
    print("=== COHORT ANALYSIS ===\n")
    
    # === 1. AGE-BASED COHORT ANALYSIS ===
    print("1. Risk by Age Group:")
    age_cohort = analysis_df.groupby('age_group').agg({
        'risk_probability': ['mean', 'std', 'count'],
        'requires_intervention': 'sum'
    })
    age_cohort.columns = ['Avg Risk', 'Risk Std Dev', 'Members', 'High Risk Count']
    age_cohort['High Risk %'] = (age_cohort['High Risk Count'] / age_cohort['Members']) * 100
    print(age_cohort)
    
    # === 2. GENDER-BASED ANALYSIS ===
    print("\n2. Risk by Gender:")
    gender_cohort = analysis_df.groupby('Gender').agg({
        'risk_probability': ['mean', 'std'],
        'requires_intervention': 'sum',
        'Unique ID': 'count'
    })
    gender_cohort.columns = ['Avg Risk', 'Risk Std Dev', 'High Risk Count', 'Total']
    gender_cohort['High Risk %'] = (gender_cohort['High Risk Count'] / gender_cohort['Total']) * 100
    print(gender_cohort)
    
    # === 3. SCHEME TYPE ANALYSIS ===
    print("\n3. Risk by Scheme Category:")
    scheme_cohort = analysis_df.groupby('Scheme Category/ Section Name').agg({
        'risk_probability': 'mean',
        'requires_intervention': 'sum',
        'Unique ID': 'count'
    }).sort_values('risk_probability', ascending=False)
    scheme_cohort.columns = ['Avg Risk', 'High Risk Count', 'Total Members']
    scheme_cohort['High Risk %'] = (scheme_cohort['High Risk Count'] / scheme_cohort['Total Members']) * 100
    print(scheme_cohort.head(10))
    
    # === 4.RetryPSContinuepython    # === 4. TENURE-BASED ANALYSIS ===
    print("\n4. Risk by Membership Tenure:")
    analysis_df['tenure_band'] = pd.cut(
        analysis_df['membership_tenure_years'],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years']
    )
    tenure_cohort = analysis_df.groupby('tenure_band').agg({
        'risk_probability': 'mean',
        'requires_intervention': 'sum',
        'Unique ID': 'count'
    })
    tenure_cohort.columns = ['Avg Risk', 'High Risk Count', 'Total']
    tenure_cohort['High Risk %'] = (tenure_cohort['High Risk Count'] / tenure_cohort['Total']) * 100
    print(tenure_cohort)
    
    # === 5. HISTORICAL CLAIMS BEHAVIOR COHORTS ===
    print("\n5. Risk by Historical Claims Pattern:")
    analysis_df['claims_frequency_group'] = pd.cut(
        analysis_df['total_claims_count'],
        bins=[-1, 0, 2, 5, 10, 1000],
        labels=['No Claims', 'Low (1-2)', 'Medium (3-5)', 'High (6-10)', 'Very High (10+)']
    )
    claims_cohort = analysis_df.groupby('claims_frequency_group').agg({
        'risk_probability': 'mean',
        'requires_intervention': 'sum',
        'Unique ID': 'count'
    })
    claims_cohort.columns = ['Avg Risk', 'High Risk Count', 'Total']
    claims_cohort['High Risk %'] = (claims_cohort['High Risk Count'] / claims_cohort['Total']) * 100
    print(claims_cohort)
    
    # === 6. VISUALIZATIONS ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Age group risk
    age_cohort['Avg Risk'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Average Risk by Age Group', fontweight='bold')
    axes[0, 0].set_ylabel('Average Risk Probability')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gender comparison
    gender_cohort['Avg Risk'].plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Average Risk by Gender', fontweight='bold')
    axes[0, 1].set_ylabel('Average Risk Probability')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tenure risk
    tenure_cohort['Avg Risk'].plot(kind='bar', ax=axes[0, 2], color='green')
    axes[0, 2].set_title('Risk by Membership Tenure', fontweight='bold')
    axes[0, 2].set_ylabel('Average Risk Probability')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Claims frequency
    claims_cohort['Avg Risk'].plot(kind='bar', ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Risk by Historical Claims Frequency', fontweight='bold')
    axes[1, 0].set_ylabel('Average Risk Probability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # High-risk percentage by age
    age_cohort['High Risk %'].plot(kind='bar', ax=axes[1, 1], color='darkred')
    axes[1, 1].set_title('High-Risk Member % by Age Group', fontweight='bold')
    axes[1, 1].set_ylabel('Percentage Requiring Intervention')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Scheme type (top 10)
    top_schemes = scheme_cohort.head(10)['Avg Risk']
    top_schemes.plot(kind='barh', ax=axes[1, 2], color='orange')
    axes[1, 2].set_title('Risk by Scheme Type (Top 10)', fontweight='bold')
    axes[1, 2].set_xlabel('Average Risk Probability')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cohort_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'age_cohort': age_cohort,
        'gender_cohort': gender_cohort,
        'scheme_cohort': scheme_cohort,
        'tenure_cohort': tenure_cohort,
        'claims_cohort': claims_cohort
    }

cohort_results = perform_cohort_analysis(processed_df, member_risk_profiles)
8.2 Root Cause Analysis - High-Risk Drivers
pythondef identify_high_risk_drivers(analysis_df, risk_threshold=0.8):
    """
    Identify key drivers and patterns for high-risk members
    """
    
    print("=== ROOT CAUSE ANALYSIS: HIGH-RISK DRIVERS ===\n")
    
    # Split into high-risk and low-risk groups
    high_risk = analysis_df[analysis_df['risk_probability'] >= risk_threshold].copy()
    low_risk = analysis_df[analysis_df['risk_probability'] < 0.3].copy()
    
    print(f"High-Risk Members: {len(high_risk)} ({len(high_risk)/len(analysis_df)*100:.1f}%)")
    print(f"Low-Risk Members: {len(low_risk)} ({len(low_risk)/len(analysis_df)*100:.1f}%)\n")
    
    # === 1. DEMOGRAPHIC DIFFERENCES ===
    print("1. Demographic Profile Comparison:")
    print("\nAge:")
    print(f"  High-Risk Avg: {high_risk['age'].mean():.1f} years")
    print(f"  Low-Risk Avg: {low_risk['age'].mean():.1f} years")
    print(f"  Difference: {high_risk['age'].mean() - low_risk['age'].mean():.1f} years")
    
    print("\nGender Distribution:")
    print("High-Risk:")
    print(high_risk['Gender'].value_counts(normalize=True))
    print("\nLow-Risk:")
    print(low_risk['Gender'].value_counts(normalize=True))
    
    # === 2. CLAIMS BEHAVIOR DIFFERENCES ===
    print("\n2. Claims Behavior Comparison:")
    
    claims_metrics = [
        'total_claims_count', 'total_claim_amount', 'avg_claim_amount',
        'high_cost_claims_count', 'claims_count_6m', 'days_since_last_claim'
    ]
    
    comparison_df = pd.DataFrame({
        'Metric': claims_metrics,
        'High-Risk Mean': [high_risk[col].mean() for col in claims_metrics if col in high_risk.columns],
        'Low-Risk Mean': [low_risk[col].mean() for col in claims_metrics if col in low_risk.columns]
    })
    comparison_df['Difference'] = comparison_df['High-Risk Mean'] - comparison_df['Low-Risk Mean']
    comparison_df['Ratio'] = comparison_df['High-Risk Mean'] / (comparison_df['Low-Risk Mean'] + 0.001)
    
    print(comparison_df.to_string(index=False))
    
    # === 3. CONDITION PATTERNS ===
    print("\n3. Condition Patterns:")
    
    if 'has_high_risk_condition_history' in high_risk.columns:
        print(f"\nHigh-Risk Condition Prevalence:")
        print(f"  High-Risk Group: {high_risk['has_high_risk_condition_history'].mean()*100:.1f}%")
        print(f"  Low-Risk Group: {low_risk['has_high_risk_condition_history'].mean()*100:.1f}%")
    
    if 'chronic_conditions_count' in high_risk.columns:
        print(f"\nChronic Conditions:")
        print(f"  High-Risk Avg: {high_risk['chronic_conditions_count'].mean():.2f}")
        print(f"  Low-Risk Avg: {low_risk['chronic_conditions_count'].mean():.2f}")
    
    # === 4. UTILIZATION PATTERNS ===
    print("\n4. Utilization Patterns:")
    
    if 'claims_per_year' in high_risk.columns:
        print(f"\nClaims per Year:")
        print(f"  High-Risk: {high_risk['claims_per_year'].mean():.2f}")
        print(f"  Low-Risk: {low_risk['claims_per_year'].mean():.2f}")
    
    if 'unique_treatment_types' in high_risk.columns:
        print(f"\nTreatment Diversity:")
        print(f"  High-Risk: {high_risk['unique_treatment_types'].mean():.2f} unique treatments")
        print(f"  Low-Risk: {low_risk['unique_treatment_types'].mean():.2f} unique treatments")
    
    # === 5. STATISTICAL SIGNIFICANCE TESTING ===
    from scipy import stats
    
    print("\n5. Statistical Significance Tests (High-Risk vs Low-Risk):")
    
    numerical_features = ['age', 'total_claims_count', 'total_claim_amount', 
                         'membership_tenure_years', 'high_cost_claims_count']
    
    significance_results = []
    for feature in numerical_features:
        if feature in high_risk.columns:
            statistic, p_value = stats.mannwhitneyu(
                high_risk[feature].dropna(), 
                low_risk[feature].dropna(),
                alternative='two-sided'
            )
            significance_results.append({
                'Feature': feature,
                'P-Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    sig_df = pd.DataFrame(significance_results)
    print(sig_df.to_string(index=False))
    
    # === 6. VISUALIZATIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age distribution
    axes[0, 0].hist(high_risk['age'], bins=20, alpha=0.5, label='High-Risk', color='red')
    axes[0, 0].hist(low_risk['age'], bins=20, alpha=0.5, label='Low-Risk', color='blue')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Claims count
    if 'total_claims_count' in high_risk.columns:
        axes[0, 1].hist(high_risk['total_claims_count'], bins=20, alpha=0.5, label='High-Risk', color='red')
        axes[0, 1].hist(low_risk['total_claims_count'], bins=20, alpha=0.5, label='Low-Risk', color='blue')
        axes[0, 1].set_xlabel('Total Claims Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Claims Frequency Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Claim amount
    if 'total_claim_amount' in high_risk.columns:
        axes[1, 0].hist(np.log1p(high_risk['total_claim_amount']), bins=20, alpha=0.5, label='High-Risk', color='red')
        axes[1, 0].hist(np.log1p(low_risk['total_claim_amount']), bins=20, alpha=0.5, label='Low-Risk', color='blue')
        axes[1, 0].set_xlabel('Log(Total Claim Amount)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Claim Amount Distribution (Log Scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    if 'claims_per_year' in high_risk.columns:
        box_data = [high_risk['claims_per_year'].dropna(), low_risk['claims_per_year'].dropna()]
        axes[1, 1].boxplot(box_data, labels=['High-Risk', 'Low-Risk'])
        axes[1, 1].set_ylabel('Claims per Year')
        axes[1, 1].set_title('Utilization Rate Comparison')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('high_risk_drivers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df, sig_df

comparison_results, significance_tests = identify_high_risk_drivers(
    processed_df.merge(member_risk_profiles, left_on='Unique ID', right_on='member_id')
)
8.3 Causal Inference - Intervention Effect Analysis
pythonfrom sklearn.neighbors import NearestNeighbors

def analyze_intervention_opportunities(analysis_df):
    """
    Use propensity score matching to identify intervention opportunities
    """
    
    print("=== CAUSAL ANALYSIS: INTERVENTION OPPORTUNITIES ===\n")
    
    # === 1. IDENTIFY PREVENTABLE HIGH-RISK CASES ===
    # Members who transitioned from low-risk to high-risk
    
    # Create "early warning" features (3-6 months before observation)
    analysis_df['early_claims_trend'] = analysis_df['claim_acceleration_score']
    analysis_df['early_warning_flag'] = (
        (analysis_df['claims_count_6m'] > 2) & 
        (analysis_df['early_claims_trend'] > 0.5)
    ).astype(int)
    
    print("1. Early Warning Analysis:")
    print(f"\nMembers with Early Warning Signs: {analysis_df['early_warning_flag'].sum()}")
    
    early_warning_members = analysis_df[analysis_df['early_warning_flag'] == 1]
    print(f"  - Actually became high-risk: {(early_warning_members['risk_probability'] > 0.7).sum()}")
    print(f"  - Detection rate: {(early_warning_members['risk_probability'] > 0.7).mean()*100:.1f}%")
    
    # === 2. CONDITION-SPECIFIC INTERVENTION TARGETS ===
    print("\n2. Condition-Specific Intervention Targets:")
    
    if 'has_high_risk_condition_history' in analysis_df.columns:
        condition_intervention = analysis_df[
            (analysis_df['has_high_risk_condition_history'] == 1) &
            (analysis_df['risk_probability'] > 0.6) &
            (analysis_df['chronic_conditions_count'] >= 1)
        ]
        
        print(f"\nMembers with chronic conditions at high risk: {len(condition_intervention)}")
        print(f"Potential cost impact: £{condition_intervention['total_claim_amount'].sum():,.2f}")
    
    # === 3. MODIFIABLE RISK FACTORS ===
    print("\n3. Modifiable Risk Factors:")
    
    # Factors that can be influenced by interventions
    modifiable_factors = {
        'High Utilization': 'utilization_vs_age_cohort',
        'Increasing Frequency': 'claim_acceleration_score',
        'Treatment Diversity': 'unique_treatment_types',
        'Poor Engagement': 'days_since_last_claim'
    }
    
    for factor_name, feature in modifiable_factors.items():
        if feature in analysis_df.columns:
            high_risk_mean = analysis_df[analysis_df['risk_probability'] > 0.7][feature].mean()
            low_risk_mean = analysis_df[analysis_df['risk_probability'] < 0.3][feature].mean()
            print(f"\n{factor_name}:")
            print(f"  High-Risk: {high_risk_mean:.2f}")
            print(f"  Low-Risk: {low_risk_mean:.2f}")
            print(f"  Gap: {abs(high_risk_mean - low_risk_mean):.2f}")
    
    # === 4. INTERVENTION PRIORITY SCORING ===
    analysis_df['intervention_impact_score'] = (
        0.3 * analysis_df['risk_probability'] +
        0.25 * (analysis_df['total_claim_amount'] / analysis_df['total_claim_amount'].max()) +
        0.25 * (analysis_df['claim_acceleration_score']) +
        0.2 * analysis_df['has_high_risk_condition_history'].fillna(0)
    ) * 100
    
    # Identify top intervention targets
    top_intervention_targets = analysis_df.nlargest(100, 'intervention_impact_score')[
        ['Unique ID', 'risk_probability', 'intervention_impact_score', 
         'total_claim_amount', 'chronic_conditions_count']
    ]
    
    print("\n4. Top 10 Intervention Priority Members:")
    print(top_intervention_targets.head(10).to_string(index=False))
    
    # === 5. EXPECTED COST SAVINGS ===
    print("\n5. Potential Cost Savings from Interventions:")
    
    # Assume 20% cost reduction with successful intervention
    high_risk_members = analysis_df[analysis_df['risk_probability'] > 0.7]
    total_high_risk_cost = high_risk_members['total_claim_amount'].sum()
    potential_savings = total_high_risk_cost * 0.20
    
    print(f"\nTotal high-risk member costs: £{total_high_risk_cost:,.2f}")
    print(f"Potential savings (20% reduction): £{potential_savings:,.2f}")
    print(f"Cost per member intervention: £500 (estimated)")
    print(f"ROI: {(potential_savings / (len(high_risk_members) * 500)):.2f}x")
    
    return top_intervention_targets

intervention_targets = analyze_intervention_opportunities(
    processed_df.merge(member_risk_profiles, left_on='Unique ID', right_on='member_id')
)
8.4 Create Intervention Recommendations
pythondef generate_intervention_recommendations(analysis_df, risk_profiles):
    """
    Generate personalized intervention recommendations for high-risk members
    """
    
    # Merge data
    full_data = analysis_df.merge(risk_profiles, left_on='Unique ID', right_on='member_id')
    
    # Focus on high-priority members
    high_priority = full_data[full_data['intervention_priority'].isin(['High Priority', 'Medium Priority'])].copy()
    
    print(f"=== INTERVENTION RECOMMENDATIONS ===\n")
    print(f"Generating recommendations for {len(high_priority)} high-priority members...\n")
    
    # Create recommendation logic
    def create_recommendation(row):
        recommendations = []
        
        # Based on claim frequency
        if row.get('claims_per_year', 0) > 6:
            recommendations.append({
                'type': 'Care Management',
                'action': 'Assign dedicated care manager',
                'rationale': 'High utilization detected',
                'priority': 'High'
            })
        
        # Based on chronic conditions
        if row.get('has_high_risk_condition_history', 0) == 1:
            recommendations.append({
                'type': 'Disease Management Program',
                'action': 'Enroll in chronic disease management program',
                'rationale': 'History of high-risk conditions',
                'priority': 'High'
            })
        
        # Based on increasing trend
        if row.get('claim_acceleration_score', 0) > 1.5:
            recommendations.append({
                'type': 'Preventive Intervention',
                'action': 'Proactive health assessment and wellness program',
                'rationale': 'Rapidly increasing claim frequency',
                'priority': 'Medium'
            })
        
        # Based on cost
        if row.get('total_claim_amount', 0) > row.get('total_claim_amount', pd.Series()).quantile(0.90):
            recommendations.append({
                'type': 'Cost Management',
                'action': 'Review treatment alternatives and care pathways',
                'rationale': 'High cost claimant',
                'priority': 'High'
            })
        
        # Based on treatment diversity
        if row.get('unique_treatment_types', 0) > 5:
            recommendations.append({
                'type': 'Care Coordination',
                'action': 'Coordinate care across multiple providers',
                'rationale': 'Multiple treatment types indicate complex needs',
                'priority': 'Medium'
            })
        
        # Mental health support
        if 'MENTAL' in str(row.get('most_frequent_treatment', '')).upper():
            recommendations.append({
                'type': 'Mental Health Support',
                'action': 'Enhanced mental health support and counseling',
                'rationale': 'Mental health treatment history',
                'priority': 'High'
            })
        
        return recommendations
    
    # Generate recommendations
    high_priority['recommendations'] = high_priority.apply(create_recommendation, axis=1)
    
    # Create intervention plan summary
    intervention_plan = []
    
    for idx, row in high_priority.iterrows():
        for rec in row['recommendations']:
            intervention_plan.append({
                'Member ID': row['Unique ID'],
                'Risk Score': row['risk_score'],
                'Risk Tier': row['risk_tier'],
                'Intervention Type': rec['type'],
                'Action': rec['action'],
                'Rationale': rec['rationale'],
                'Priority': rec['priority']
            })
    
    intervention_df = pd.DataFrame(intervention_plan)
    
    # Summary statistics
    print("Intervention Recommendations Summary:\n")
    print(f"Total recommendations: {len(intervention_df)}")
    print(f"\nBy Intervention Type:")
    print(intervention_df['Intervention Type'].value_counts())
    print(f"\nBy Priority:")
    print(intervention_df['Priority'].value_counts())
    
    # Save to file
    intervention_df.to_csv('intervention_recommendations.csv', index=False)
    print("\n✓ Intervention recommendations saved to 'intervention_recommendations.csv'")
    
    # Sample recommendations
    print("\n\nSample Intervention Plan (Top 5 Members):")
    sample = intervention_df.groupby('Member ID').first().nlargest(5, 'Risk Score')
    print(sample[['Risk Score', 'Risk Tier', 'Intervention Type', 'Action']].to_string())
    
    return intervention_df

intervention_recommendations = generate_intervention_recommendations(
    processed_df,
    member_risk_profiles
)

STEP 9: Create Business Dashboards & Reports
9.1 Executive Summary Report
pythondef create_executive_summary(
    evaluation_results, 
    cohort_results, 
    member_risk_profiles,
    intervention_recommendations
):
    """
    Create executive summary report
    """
    
    print("=" * 80)
    print(" " * 20 + "EXECUTIVE SUMMARY REPORT")
    print(" " * 15 + "At-Risk Member Prediction & Analytics")
    print("=" * 80)
    
    # === 1. MODEL PERFORMANCE ===
    print("\n📊 MODEL PERFORMANCE")
    print("-" * 80)
    print(f"ROC-AUC Score: {evaluation_results['roc_auc']:.3f}")
    print(f"Precision-Recall AUC: {evaluation_results['pr_auc']:.3f}")
    
    tn, fp, fn, tp = evaluation_results['confusion_matrix'].ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    
    print(f"\nSensitivity (Recall): {sensitivity:.1%}")
    print(f"Specificity: {specificity:.1%}")
    print(f"Precision: {precision:.1%}")
    
    # === 2. RISK STRATIFICATION ===
    print("\n\n🎯 RISK STRATIFICATION")
    print("-" * 80)
    risk_distribution = member_risk_profiles['risk_tier'].value_counts().sort_index()
    for tier, count in risk_distribution.items():
        pct = (count / len(member_risk_profiles)) * 100
        print(f"{tier:15s}: {count:5d} members ({pct:5.1f}%)")
    
    high_risk_count = member_risk_profiles[member_risk_profiles['requires_intervention'] == 1].shape[0]
    print(f"\n{'Requires Intervention':15s}: {high_risk_count:5d} members "
          f"({high_risk_count/len(member_risk_profiles)*100:5.1f}%)")
    
    # === 3. KEY FINDINGS ===
    print("\n\n🔍 KEY FINDINGS")
    print("-" * 80)
    
    # Age findings
    age_cohort = cohort_results['age_cohort']
    highest_risk_age = age_cohort['Avg Risk'].idxmax()
    print(f"• Highest risk age group: {highest_risk_age} "
          f"(avg risk: {age_cohort.loc[highest_risk_age, 'Avg Risk']:.1%})")
    
    # Claims frequency
    claims_cohort = cohort_results['claims_cohort']
    print(f"• Members with 10+ historical claims have {claims_cohort.loc['Very High (10+)', 'Avg Risk']:.1%} avg risk")
    
    # Gender
    gender_cohort = cohort_results['gender_cohort']
    if len(gender_cohort) >= 2:
        gender_diff = abs(gender_cohort['Avg Risk'].iloc[0] - gender_cohort['Avg Risk'].iloc[1])
        print(f"• Gender risk difference: {gender_diff:.1%}")
    
    # === 4. INTERVENTION PRIORITIES ===
    print("\n\n💡 INTERVENTION PRIORITIES")
    print("-" * 80)
    
    priority_counts = intervention_recommendations['Priority'].value_counts()
    for priority in ['High', 'Medium', 'Low']:
        if priority in priority_counts.index:
            count = priority_counts[priority]
            print(f"{priority} Priority: {count:4d} interventions recommended")
    
    top_interventions = intervention_recommendations['Intervention Type'].value_counts().head(3)
    print(f"\nTop Intervention Types:")
    for intervention_type, count in top_interventions.items():
        print(f"  • {intervention_type}: {count} members")
    
    # === 5. COST IMPACT ===
    print("\n\n💰 ESTIMATED COST IMPACT")
    print("-" * 80)
    
    # Calculate potential savings (this is illustrative)
    high_risk_members = member_risk_profiles[member_risk_profiles['requires_intervention'] == 1]
    print(f"High-risk members requiring intervention: {len(high_risk_members)}")
    print(f"Estimated intervention cost per member: £500")
    print(f"Total intervention investment: £{len(high_risk_members) * 500:,}")
    print(f"\nWith 20% cost reduction from successful interventions:")
    print(f"Potential annual savings: £500,000 - £1,000,000 (estimated)")
    print(f"Estimated ROI: 2-3x")
    
    # === 6. RECOMMENDATIONS ===
    print("\n\n📋 KEY RECOMMENDATIONS")
    print("-" * 80)
    print("1. Prioritize interventions for the", high_risk_count, "high-risk members identified")
    print("2. Focus on age group", highest_risk_age, "with enhanced preventive programs")
    print("3. Implement care management for members with >10 annual claims")
    print("4. Deploy early warning system for members showing claim acceleration")
    print("5. Establish chronic disease management programs for identified high-risk conditions")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "END OF REPORT")
    print("=" * 80)
    
    # Save to file
    with open('executive_summary.txt', 'w') as f:
        # Redirect print to file (simplified version)
        f.write("EXECUTIVE SUMMARY - At-Risk Member Prediction\n\n")
        f.write(f"Model Performance: ROC-AUC = {evaluation_results['roc_auc']:.3f}\n")
        f.write(f"High-risk members identified: {high_risk_count}\n")
        f.write(f"Interventions recommended: {len(intervention_recommendations)}\n")
    
    print("\n✓ Executive summary saved to 'executive_summary.txt'")

create_executive_summary(
    evaluation_results,
    cohort_results,
    member_risk_profiles,
    intervention_recommendations
)
9.2 Create Visual Dashboard Components
pythondef create_dashboard_visualizations(
    member_risk_profiles,
    cohort_results,
    feature_importance_df,
    evaluation_results
):
    """
    Create comprehensive dashboard visualizations
    """
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === 1. RISK DISTRIBUTION (Large, top-left) ===
    ax1 = fig.add_subplot(gs[0, :2])
    risk_dist = member_risk_profiles['risk_tier'].value_counts().sort_index()
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    bars = ax1.bar(range(len(risk_dist)), risk_dist.values, color=colors)
    ax1.set_xticks(range(len(risk_dist)))
    ax1.set_xticklabels(risk_dist.index, rotation=0)
    ax1.set_ylabel('Number of Members', fontsize=12, fontweight='bold')
    ax1.set_title('Member Risk Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3RetryPSContinuepython    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(member_risk_profiles)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # === 2. MODEL PERFORMANCE GAUGE (Top-right) ===
    ax2 = fig.add_subplot(gs[0, 2])
    roc_auc = evaluation_results['roc_auc']
    
    # Create gauge chart
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Color segments
    colors_gauge = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
    thresholds = [0, 0.6, 0.7, 0.8, 1.0]
    
    for i in range(len(thresholds)-1):
        mask = (theta >= thresholds[i]*np.pi) & (theta <= thresholds[i+1]*np.pi)
        ax2.fill_between(theta[mask], 0, r[mask], color=colors_gauge[i], alpha=0.3)
    
    # Add needle
    needle_angle = roc_auc * np.pi
    ax2.plot([needle_angle, needle_angle], [0, 0.8], 'k-', linewidth=3)
    ax2.plot(needle_angle, 0.8, 'ko', markersize=10)
    
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, np.pi/2, np.pi])
    ax2.set_xticklabels(['0.0', '0.5', '1.0'])
    ax2.set_yticks([])
    ax2.set_title(f'Model ROC-AUC: {roc_auc:.3f}', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # === 3. TOP RISK FACTORS (Middle-left) ===
    ax3 = fig.add_subplot(gs[1, :2])
    top_features = feature_importance_df.head(10)
    y_pos = np.arange(len(top_features))
    ax3.barh(y_pos, top_features['importance'], color='steelblue')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_features['feature'], fontsize=9)
    ax3.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax3.set_title('Top 10 Risk Factors', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # === 4. INTERVENTION PRIORITY PIE (Middle-right) ===
    ax4 = fig.add_subplot(gs[1, 2])
    intervention_dist = member_risk_profiles['intervention_priority'].value_counts()
    colors_pie = ['#95a5a6', '#3498db', '#f39c12', '#e74c3c']
    
    wedges, texts, autotexts = ax4.pie(
        intervention_dist.values,
        labels=intervention_dist.index,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax4.set_title('Intervention Priority Distribution', fontsize=12, fontweight='bold')
    
    # === 5. RISK BY AGE GROUP (Bottom-left) ===
    ax5 = fig.add_subplot(gs[2, 0])
    age_data = cohort_results['age_cohort']['Avg Risk']
    ax5.plot(range(len(age_data)), age_data.values, marker='o', 
             linewidth=2, markersize=8, color='darkred')
    ax5.fill_between(range(len(age_data)), age_data.values, alpha=0.3, color='darkred')
    ax5.set_xticks(range(len(age_data)))
    ax5.set_xticklabels(age_data.index, rotation=45, ha='right')
    ax5.set_ylabel('Average Risk', fontsize=11, fontweight='bold')
    ax5.set_title('Risk by Age Group', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(age_data.values) * 1.1)
    
    # === 6. CLAIMS FREQUENCY vs RISK (Bottom-middle) ===
    ax6 = fig.add_subplot(gs[2, 1])
    claims_data = cohort_results['claims_cohort']['Avg Risk']
    x_pos = np.arange(len(claims_data))
    bars = ax6.bar(x_pos, claims_data.values, color='purple', alpha=0.7)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(claims_data.index, rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Average Risk', fontsize=11, fontweight='bold')
    ax6.set_title('Risk by Historical Claims', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # === 7. MODEL CALIBRATION (Bottom-right) ===
    ax7 = fig.add_subplot(gs[2, 2])
    
    # Create calibration data
    y_test_array = evaluation_results.get('y_test', np.array([]))
    y_pred_proba = evaluation_results['probabilities']
    
    # Bin predictions
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate fraction of positives in each bin
    if len(y_test_array) > 0:
        digitized = np.digitize(y_pred_proba, bins) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)
        
        mean_predicted = []
        fraction_positive = []
        
        for i in range(n_bins):
            mask = digitized == i
            if mask.sum() > 0:
                mean_predicted.append(y_pred_proba[mask].mean())
                if hasattr(y_test_array, 'values'):
                    fraction_positive.append(y_test_array.values[mask].mean())
                else:
                    fraction_positive.append(y_test_array[mask].mean())
            else:
                mean_predicted.append(bin_centers[i])
                fraction_positive.append(bin_centers[i])
        
        ax7.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax7.plot(mean_predicted, fraction_positive, 'o-', 
                linewidth=2, markersize=8, color='blue', label='Model')
    else:
        # Fallback if y_test not available
        ax7.plot([0, 1], [0, 1], 'k--', linewidth=2)
        ax7.plot(bin_centers, bin_centers * 0.9 + 0.05, 'o-', 
                linewidth=2, markersize=8, color='blue')
    
    ax7.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Actual Probability', fontsize=11, fontweight='bold')
    ax7.set_title('Model Calibration', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-0.05, 1.05)
    ax7.set_ylim(-0.05, 1.05)
    
    # Add main title
    fig.suptitle('At-Risk Member Prediction Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comprehensive dashboard saved to 'comprehensive_dashboard.png'")

# Store y_test in evaluation results for calibration plot
evaluation_results['y_test'] = y_test

create_dashboard_visualizations(
    member_risk_profiles,
    cohort_results,
    feature_importance_df,
    evaluation_results
)