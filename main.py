"""
Advanced Spotify Streaming Analysis
A comprehensive data science project featuring A/B testing, predictive modeling,
and statistical inference.

Author: Your Name
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class SpotifyAnalyzer:
    """
    Advanced Spotify streaming data analyzer with statistical testing,
    predictive modeling, and causal inference capabilities.
    """

    def __init__(self, data_path):
        """Initialize analyzer with data path."""
        self.df = pd.read_csv(data_path)
        self.df['Release Date'] = pd.to_datetime(self.df['Release Date'])
        self.df['Release Year'] = self.df['Release Date'].dt.year
        self.df['Release Month'] = self.df['Release Date'].dt.month
        self.df['Days Since Release'] = (pd.Timestamp.now() - self.df['Release Date']).dt.days

        # Feature engineering
        self._engineer_features()

    def _engineer_features(self):
        """Create advanced features for analysis."""
        # Streams in millions for easier interpretation
        self.df['Streams (Millions)'] = self.df['Streams (Billions)'] * 1000

        # Categorize release eras
        self.df['Era'] = pd.cut(self.df['Release Year'],
                                bins=[1970, 2000, 2015, 2020, 2025],
                                labels=['Classic', 'Modern', 'Recent', 'Current'])

        # Season of release
        self.df['Season'] = self.df['Release Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # Collaboration indicator
        self.df['Is_Collaboration'] = self.df['Artist'].str.contains('feat|&|,', case=False, na=False)

        # Streams per day (velocity metric)
        self.df['Streams_Per_Day'] = self.df['Streams (Millions)'] / self.df['Days Since Release']

    def exploratory_analysis(self):
        """Comprehensive exploratory data analysis with statistical insights."""
        print("=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        # Basic statistics
        print("\n📊 Streaming Statistics:")
        print(f"Total Songs: {len(self.df)}")
        print(f"Mean Streams: {self.df['Streams (Billions)'].mean():.3f}B")
        print(f"Median Streams: {self.df['Streams (Billions)'].median():.3f}B")
        print(f"Std Dev: {self.df['Streams (Billions)'].std():.3f}B")
        print(f"Skewness: {self.df['Streams (Billions)'].skew():.3f}")
        print(f"Kurtosis: {self.df['Streams (Billions)'].kurtosis():.3f}")

        # Top performers
        print("\n🏆 Top 5 Most Streamed Songs:")
        top5 = self.df.nlargest(5, 'Streams (Billions)')[['Song', 'Artist', 'Streams (Billions)']]
        print(top5.to_string(index=False))

        # Artist analysis
        artist_streams = self.df.groupby('Artist')['Streams (Billions)'].agg(['sum', 'count', 'mean'])
        artist_streams = artist_streams.sort_values('sum', ascending=False).head(10)
        print("\n🎤 Top 10 Artists by Total Streams:")
        print(artist_streams)

        # Collaboration analysis
        collab_stats = self.df.groupby('Is_Collaboration')['Streams (Billions)'].agg(['mean', 'median', 'count'])
        print("\n🤝 Collaboration vs Solo Performance:")
        print(collab_stats)

        # Era analysis
        era_stats = self.df.groupby('Era')['Streams (Billions)'].agg(['mean', 'median', 'count'])
        print("\n📅 Performance by Era:")
        print(era_stats)

    def ab_test_collaborations(self):
        """
        A/B Test: Do collaborations lead to significantly higher streams?
        """
        print("\n" + "=" * 80)
        print("A/B TEST: COLLABORATIONS vs SOLO SONGS")
        print("=" * 80)

        # Separate into treatment (collaborations) and control (solo)
        group_a = self.df[self.df['Is_Collaboration'] == True]['Streams (Billions)']
        group_b = self.df[self.df['Is_Collaboration'] == False]['Streams (Billions)']

        print(f"\nGroup A (Collaborations): n={len(group_a)}")
        print(f"  Mean: {group_a.mean():.3f}B streams")
        print(f"  Median: {group_a.median():.3f}B streams")
        print(f"  Std: {group_a.std():.3f}B")

        print(f"\nGroup B (Solo): n={len(group_b)}")
        print(f"  Mean: {group_b.mean():.3f}B streams")
        print(f"  Median: {group_b.median():.3f}B streams")
        print(f"  Std: {group_b.std():.3f}B")

        # Check normality
        _, p_norm_a = stats.shapiro(group_a)
        _, p_norm_b = stats.shapiro(group_b)
        print(f"\n🔍 Normality Tests:")
        print(f"  Group A p-value: {p_norm_a:.4f}")
        print(f"  Group B p-value: {p_norm_b:.4f}")

        # Independent t-test (parametric)
        t_stat, p_ttest = stats.ttest_ind(group_a, group_b)
        print(f"\n📈 Independent T-Test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_ttest:.4f}")

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, p_mann = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        print(f"\n📊 Mann-Whitney U Test (non-parametric):")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value: {p_mann:.4f}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_a) - 1) * group_a.std() ** 2 + (len(group_b) - 1) * group_b.std() ** 2) / (
                    len(group_a) + len(group_b) - 2))
        cohens_d = (group_a.mean() - group_b.mean()) / pooled_std
        print(f"\n📏 Effect Size (Cohen's d): {cohens_d:.4f}")

        # Interpretation
        alpha = 0.05
        print(f"\n💡 Interpretation (α={alpha}):")
        if p_mann < alpha:
            direction = "higher" if group_a.mean() > group_b.mean() else "lower"
            print(f"  ✓ SIGNIFICANT: Collaborations have {direction} streams (p={p_mann:.4f})")
            if abs(cohens_d) < 0.2:
                effect = "small"
            elif abs(cohens_d) < 0.5:
                effect = "medium"
            else:
                effect = "large"
            print(f"  Effect size is {effect} (d={cohens_d:.4f})")
        else:
            print(f"  ✗ NOT SIGNIFICANT: No statistical difference found (p={p_mann:.4f})")

    def ab_test_release_season(self):
        """
        A/B Test: Does release season impact streaming success?
        """
        print("\n" + "=" * 80)
        print("A/B TEST: RELEASE SEASON IMPACT")
        print("=" * 80)

        # Group by season
        season_groups = [self.df[self.df['Season'] == season]['Streams (Billions)']
                         for season in ['Winter', 'Spring', 'Summer', 'Fall']]

        print("\n📅 Streaming Performance by Season:")
        for season, group in zip(['Winter', 'Spring', 'Summer', 'Fall'], season_groups):
            print(f"  {season}: n={len(group)}, mean={group.mean():.3f}B, median={group.median():.3f}B")

        # Kruskal-Wallis H-test (non-parametric ANOVA)
        h_stat, p_kruskal = stats.kruskal(*season_groups)
        print(f"\n📊 Kruskal-Wallis H-Test:")
        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  p-value: {p_kruskal:.4f}")

        if p_kruskal < 0.05:
            print(f"  ✓ SIGNIFICANT: Release season impacts streams")
            # Post-hoc pairwise comparisons
            print("\n🔍 Pairwise Comparisons (Mann-Whitney with Bonferroni correction):")
            seasons = ['Winter', 'Spring', 'Summer', 'Fall']
            alpha_corrected = 0.05 / 6  # Bonferroni correction for 6 comparisons
            for i in range(len(seasons)):
                for j in range(i + 1, len(seasons)):
                    _, p_val = stats.mannwhitneyu(season_groups[i], season_groups[j])
                    sig = "✓" if p_val < alpha_corrected else "✗"
                    print(f"  {sig} {seasons[i]} vs {seasons[j]}: p={p_val:.4f}")
        else:
            print(f"  ✗ NOT SIGNIFICANT: No seasonal effect detected")

    def predictive_modeling(self):
        """
        Build predictive models to forecast streaming success.
        """
        print("\n" + "=" * 80)
        print("PREDICTIVE MODELING: STREAM SUCCESS")
        print("=" * 80)

        # Prepare features
        features_df = self.df.copy()

        # Encode categorical variables
        le_era = LabelEncoder()
        le_season = LabelEncoder()
        features_df['Era_Encoded'] = le_era.fit_transform(features_df['Era'])
        features_df['Season_Encoded'] = le_season.fit_transform(features_df['Season'])

        # Select features
        feature_cols = ['Release Year', 'Release Month', 'Days Since Release',
                        'Era_Encoded', 'Season_Encoded', 'Is_Collaboration']
        X = features_df[feature_cols]
        y = features_df['Streams (Billions)']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        }

        print("\n🤖 Model Performance:")
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                        scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())

            print(f"\n  {name}:")
            print(f"    RMSE: {rmse:.4f}B streams")
            print(f"    MAE: {mae:.4f}B streams")
            print(f"    R²: {r2:.4f}")
            print(f"    CV RMSE: {cv_rmse:.4f}B streams")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                print(f"\n    Top 3 Feature Importances:")
                for idx, row in importance.head(3).iterrows():
                    print(f"      {row['Feature']}: {row['Importance']:.4f}")

    def correlation_analysis(self):
        """
        Analyze correlations between features and streaming success.
        """
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)

        # Numeric columns
        numeric_cols = ['Streams (Billions)', 'Release Year', 'Days Since Release',
                        'Streams_Per_Day']

        # Correlation matrix
        corr_matrix = self.df[numeric_cols].corr()

        print("\n📊 Correlation with Streams:")
        stream_corr = corr_matrix['Streams (Billions)'].sort_values(ascending=False)
        for col, corr in stream_corr.items():
            if col != 'Streams (Billions)':
                # Test significance
                n = len(self.df)
                t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr ** 2)
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                sig = "✓" if p_val < 0.05 else "✗"
                print(f"  {sig} {col}: r={corr:.4f}, p={p_val:.4f}")

    def visualize_insights(self):
        """
        Create professional visualizations of key insights.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distribution with KDE
        ax1 = axes[0, 0]
        sns.histplot(self.df['Streams (Billions)'], bins=30, kde=True, ax=ax1, color='steelblue')
        ax1.axvline(self.df['Streams (Billions)'].mean(), color='red', linestyle='--', label='Mean')
        ax1.axvline(self.df['Streams (Billions)'].median(), color='orange', linestyle='--', label='Median')
        ax1.set_title('Distribution of Streams (with KDE)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Streams (Billions)')
        ax1.legend()

        # 2. Collaboration comparison
        ax2 = axes[0, 1]
        collab_data = self.df.groupby('Is_Collaboration')['Streams (Billions)'].apply(list)
        bp = ax2.boxplot([collab_data[False], collab_data[True]],
                         labels=['Solo', 'Collaboration'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
            patch.set_facecolor(color)
        ax2.set_title('Streams: Solo vs Collaboration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Streams (Billions)')

        # 3. Time series trend
        ax3 = axes[1, 0]
        yearly = self.df.groupby('Release Year')['Streams (Billions)'].agg(['mean', 'count'])
        yearly = yearly[yearly['count'] >= 3]  # Only years with 3+ songs
        ax3.plot(yearly.index, yearly['mean'], marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax3.fill_between(yearly.index, yearly['mean'], alpha=0.3, color='lightgreen')
        ax3.set_title('Average Streams by Release Year', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Release Year')
        ax3.set_ylabel('Avg Streams (Billions)')
        ax3.grid(True, alpha=0.3)

        # 4. Top artists
        ax4 = axes[1, 1]
        top_artists = self.df.groupby('Artist')['Streams (Billions)'].sum().nlargest(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_artists)))
        ax4.barh(range(len(top_artists)), top_artists.values, color=colors)
        ax4.set_yticks(range(len(top_artists)))
        ax4.set_yticklabels(top_artists.index, fontsize=9)
        ax4.set_title('Top 10 Artists by Total Streams', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Total Streams (Billions)')
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig('advanced_spotify_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'advanced_spotify_analysis.png'")

    def generate_report(self):
        """
        Generate comprehensive analysis report.
        """
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY")
        print("=" * 80)

        total_streams = self.df['Streams (Billions)'].sum()
        avg_streams = self.df['Streams (Billions)'].mean()

        collab_lift = (self.df[self.df['Is_Collaboration']]['Streams (Billions)'].mean() /
                       self.df[~self.df['Is_Collaboration']]['Streams (Billions)'].mean() - 1) * 100

        print(f"""
📈 Key Metrics:
  • Total Streams Analyzed: {total_streams:.2f}B
  • Average Streams per Song: {avg_streams:.3f}B
  • Total Songs: {len(self.df)}
  • Unique Artists: {self.df['Artist'].nunique()}

🤝 Collaboration Impact:
  • Collaboration Lift: {collab_lift:+.1f}%
  • Songs with Collaborations: {self.df['Is_Collaboration'].sum()} ({self.df['Is_Collaboration'].mean() * 100:.1f}%)

📅 Temporal Insights:
  • Year Range: {self.df['Release Year'].min()} - {self.df['Release Year'].max()}
  • Most Productive Era: {self.df['Era'].value_counts().index[0]}

🏆 Top Performer:
  • Song: {self.df.loc[self.df['Streams (Billions)'].idxmax(), 'Song']}
  • Artist: {self.df.loc[self.df['Streams (Billions)'].idxmax(), 'Artist']}
  • Streams: {self.df['Streams (Billions)'].max():.3f}B
        """)


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SpotifyAnalyzer('Streams.csv')

    # Run comprehensive analysis
    analyzer.exploratory_analysis()
    analyzer.correlation_analysis()
    analyzer.ab_test_collaborations()
    analyzer.ab_test_release_season()
    analyzer.predictive_modeling()
    analyzer.visualize_insights()
    analyzer.generate_report()

    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)