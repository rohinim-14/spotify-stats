"""
Statistical Testing Framework for A/B Tests
Provides comprehensive hypothesis testing with multiple methods and effect sizes.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = None
    interpretation: str = ""

    def __str__(self):
        sig_marker = "✓" if self.significant else "✗"
        result = f"{sig_marker} {self.test_name}\n"
        result += f"  Statistic: {self.statistic:.4f}\n"
        result += f"  P-value: {self.p_value:.4f}\n"
        if self.effect_size is not None:
            result += f"  Effect Size: {self.effect_size:.4f}\n"
        if self.interpretation:
            result += f"  {self.interpretation}\n"
        return result


class ABTest:
    """
    Comprehensive A/B testing framework with multiple statistical methods.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize A/B test framework.

        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        self.results = []

    def check_normality(self, group_a: np.ndarray, group_b: np.ndarray) -> Tuple[bool, bool]:
        """
        Test normality assumption using Shapiro-Wilk test.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            Tuple of (group_a_normal, group_b_normal)
        """
        _, p_a = stats.shapiro(group_a)
        _, p_b = stats.shapiro(group_b)
        return p_a > self.alpha, p_b > self.alpha

    def cohens_d(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            Cohen's d value
        """
        n_a, n_b = len(group_a), len(group_b)
        var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        return (np.mean(group_a) - np.mean(group_b)) / pooled_std

    def interpret_effect_size(self, d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible effect"
        elif abs_d < 0.5:
            return "small effect"
        elif abs_d < 0.8:
            return "medium effect"
        else:
            return "large effect"

    def independent_ttest(self, group_a: np.ndarray, group_b: np.ndarray) -> TestResult:
        """
        Perform independent samples t-test.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            TestResult object
        """
        t_stat, p_value = stats.ttest_ind(group_a, group_b)
        effect_size = self.cohens_d(group_a, group_b)
        significant = p_value < self.alpha

        interpretation = f"Groups are {'significantly' if significant else 'not significantly'} different. "
        interpretation += f"Effect size is {self.interpret_effect_size(effect_size)}."

        result = TestResult(
            test_name="Independent T-Test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation
        )
        self.results.append(result)
        return result

    def mann_whitney_u(self, group_a: np.ndarray, group_b: np.ndarray) -> TestResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            TestResult object
        """
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        effect_size = self.cohens_d(group_a, group_b)
        significant = p_value < self.alpha

        interpretation = f"Distributions are {'significantly' if significant else 'not significantly'} different. "
        interpretation += "This non-parametric test doesn't assume normality."

        result = TestResult(
            test_name="Mann-Whitney U Test",
            statistic=u_stat,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation
        )
        self.results.append(result)
        return result

    def welch_ttest(self, group_a: np.ndarray, group_b: np.ndarray) -> TestResult:
        """
        Perform Welch's t-test (doesn't assume equal variances).

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            TestResult object
        """
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        effect_size = self.cohens_d(group_a, group_b)
        significant = p_value < self.alpha

        interpretation = f"Groups are {'significantly' if significant else 'not significantly'} different. "
        interpretation += "This test doesn't assume equal variances."

        result = TestResult(
            test_name="Welch's T-Test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            interpretation=interpretation
        )
        self.results.append(result)
        return result

    def levene_test(self, group_a: np.ndarray, group_b: np.ndarray) -> TestResult:
        """
        Test equality of variances using Levene's test.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            TestResult object
        """
        w_stat, p_value = stats.levene(group_a, group_b)
        significant = p_value < self.alpha

        interpretation = f"Variances are {'significantly' if significant else 'not significantly'} different. "
        if significant:
            interpretation += "Use Welch's t-test instead of standard t-test."

        result = TestResult(
            test_name="Levene's Test (Variance Equality)",
            statistic=w_stat,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )
        self.results.append(result)
        return result

    def comprehensive_test(self, group_a: np.ndarray, group_b: np.ndarray,
                           group_names: Tuple[str, str] = ("A", "B")) -> Dict:
        """
        Run comprehensive A/B test battery with automatic test selection.

        Args:
            group_a: First group data
            group_b: Second group data
            group_names: Names for the groups

        Returns:
            Dictionary with all test results and recommendations
        """
        results = {
            'group_a_name': group_names[0],
            'group_b_name': group_names[1],
            'group_a_stats': {
                'n': len(group_a),
                'mean': np.mean(group_a),
                'median': np.median(group_a),
                'std': np.std(group_a, ddof=1)
            },
            'group_b_stats': {
                'n': len(group_b),
                'mean': np.mean(group_b),
                'median': np.median(group_b),
                'std': np.std(group_b, ddof=1)
            }
        }

        # Check assumptions
        normal_a, normal_b = self.check_normality(group_a, group_b)
        results['normality'] = {
            'group_a_normal': normal_a,
            'group_b_normal': normal_b
        }

        # Test variance equality
        levene_result = self.levene_test(group_a, group_b)
        equal_var = not levene_result.significant
        results['equal_variances'] = equal_var

        # Choose appropriate test
        if normal_a and normal_b:
            if equal_var:
                primary_test = self.independent_ttest(group_a, group_b)
                results['recommendation'] = "Data is normal with equal variances. Using Independent T-Test."
            else:
                primary_test = self.welch_ttest(group_a, group_b)
                results['recommendation'] = "Data is normal but variances differ. Using Welch's T-Test."
        else:
            primary_test = self.mann_whitney_u(group_a, group_b)
            results['recommendation'] = "Data is not normal. Using Mann-Whitney U Test (non-parametric)."

        # Always run Mann-Whitney as robustness check
        if normal_a and normal_b:
            self.mann_whitney_u(group_a, group_b)

        results['primary_test'] = primary_test
        results['all_tests'] = self.results[-2:] if len(self.results) >= 2 else self.results[-1:]

        # Calculate confidence intervals
        ci_a = stats.t.interval(0.95, len(group_a) - 1,
                                loc=np.mean(group_a),
                                scale=stats.sem(group_a))
        ci_b = stats.t.interval(0.95, len(group_b) - 1,
                                loc=np.mean(group_b),
                                scale=stats.sem(group_b))

        results['confidence_intervals'] = {
            'group_a': ci_a,
            'group_b': ci_b
        }

        # Effect size interpretation
        effect_size = primary_test.effect_size
        results['effect_size'] = {
            'value': effect_size,
            'interpretation': self.interpret_effect_size(effect_size),
            'direction': 'higher' if np.mean(group_a) > np.mean(group_b) else 'lower'
        }

        return results

    def print_comprehensive_results(self, results: Dict):
        """
        Print formatted comprehensive test results.

        Args:
            results: Dictionary from comprehensive_test()
        """
        print(f"\n{'=' * 80}")
        print(f"A/B TEST: {results['group_a_name']} vs {results['group_b_name']}")
        print(f"{'=' * 80}")

        print(f"\n📊 Group Statistics:")
        for group_name, stats_dict in [('Group A', results['group_a_stats']),
                                       ('Group B', results['group_b_stats'])]:
            print(f"\n  {group_name} ({results[group_name.lower().replace(' ', '_') + '_name']}):")
            print(f"    n = {stats_dict['n']}")
            print(f"    Mean = {stats_dict['mean']:.4f}")
            print(f"    Median = {stats_dict['median']:.4f}")
            print(f"    Std Dev = {stats_dict['std']:.4f}")

            if group_name == 'Group A':
                ci = results['confidence_intervals']['group_a']
            else:
                ci = results['confidence_intervals']['group_b']
            print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

        print(f"\n🔍 Assumption Checks:")
        print(f"  Group A Normal: {'✓' if results['normality']['group_a_normal'] else '✗'}")
        print(f"  Group B Normal: {'✓' if results['normality']['group_b_normal'] else '✗'}")
        print(f"  Equal Variances: {'✓' if results['equal_variances'] else '✗'}")

        print(f"\n💡 Test Selection:")
        print(f"  {results['recommendation']}")

        print(f"\n📈 Test Results:")
        print(results['primary_test'])

        print(f"\n📏 Effect Size Analysis:")
        es = results['effect_size']
        print(f"  Cohen's d = {es['value']:.4f}")
        print(f"  Interpretation: {es['interpretation']}")
        print(f"  Direction: Group A is {es['direction']} than Group B")

        if len(results['all_tests']) > 1:
            print(f"\n🔄 Robustness Check:")
            for test in results['all_tests'][1:]:
                print(
                    f"  {test.test_name}: p={test.p_value:.4f} ({'significant' if test.significant else 'not significant'})")


class MultiGroupTest:
    """
    Framework for testing differences across multiple groups (ANOVA/Kruskal-Wallis).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def kruskal_wallis(self, *groups) -> TestResult:
        """
        Perform Kruskal-Wallis H-test (non-parametric ANOVA).

        Args:
            *groups: Variable number of group arrays

        Returns:
            TestResult object
        """
        h_stat, p_value = stats.kruskal(*groups)
        significant = p_value < self.alpha

        interpretation = f"Groups are {'significantly' if significant else 'not significantly'} different. "
        if significant:
            interpretation += "Follow up with post-hoc pairwise tests."

        return TestResult(
            test_name="Kruskal-Wallis H-Test",
            statistic=h_stat,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )

    def one_way_anova(self, *groups) -> TestResult:
        """
        Perform one-way ANOVA.

        Args:
            *groups: Variable number of group arrays

        Returns:
            TestResult object
        """
        f_stat, p_value = stats.f_oneway(*groups)
        significant = p_value < self.alpha

        interpretation = f"Groups are {'significantly' if significant else 'not significantly'} different. "
        if significant:
            interpretation += "At least one group mean differs from the others."

        return TestResult(
            test_name="One-Way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation
        )

    def pairwise_tests(self, groups: List[np.ndarray], group_names: List[str],
                       correction: str = 'bonferroni') -> List[Tuple]:
        """
        Perform pairwise Mann-Whitney tests with multiple testing correction.

        Args:
            groups: List of group arrays
            group_names: Names for each group
            correction: Type of correction ('bonferroni', 'holm')

        Returns:
            List of (name1, name2, p_value, significant) tuples
        """
        results = []
        n_comparisons = len(groups) * (len(groups) - 1) // 2

        if correction == 'bonferroni':
            alpha_corrected = self.alpha / n_comparisons
        else:
            alpha_corrected = self.alpha

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                _, p_val = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                significant = p_val < alpha_corrected
                results.append((group_names[i], group_names[j], p_val, significant))

        return results