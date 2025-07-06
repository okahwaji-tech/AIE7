#!/usr/bin/env python3
"""
Enhanced RAG Distance Metrics Analysis
This script provides comprehensive statistical analysis and visualization
of RAG system performance across different distance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedRAGAnalyzer:
    """Enhanced analyzer for RAG system performance metrics."""
    
    def __init__(self, evaluation_data):
        """Initialize with evaluation data from the notebook."""
        self.data = evaluation_data
        self.metrics = ['cosine', 'euclidean', 'manhattan', 'dot_product']
        self.eval_dimensions = ['context_relevance', 'answer_faithfulness', 
                               'answer_relevance', 'context_utilization', 'hallucination_risk']
        
    def calculate_confidence_intervals(self, data, confidence=0.95):
        """Calculate confidence intervals for each metric."""
        alpha = 1 - confidence
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, mean - interval, mean + interval
    
    def perform_anova_test(self, metric_name):
        """Perform ANOVA test to check if there's significant difference between distance metrics."""
        groups = []
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric][metric_name].values
            groups.append(metric_data)
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value
    
    def pairwise_comparisons(self, metric_name):
        """Perform pairwise t-tests with Bonferroni correction."""
        results = {}
        n_comparisons = len(self.metrics) * (len(self.metrics) - 1) / 2
        alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
        
        for i, metric1 in enumerate(self.metrics):
            for metric2 in self.metrics[i+1:]:
                data1 = self.data[self.data['metric'] == metric1][metric_name].values
                data2 = self.data[self.data['metric'] == metric2][metric_name].values
                
                t_stat, p_value = ttest_ind(data1, data2)
                effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                
                results[f"{metric1}_vs_{metric2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha_corrected,
                    'effect_size': effect_size
                }
        
        return results
    
    def calculate_correlations(self):
        """Calculate correlations between evaluation dimensions."""
        correlation_data = {}
        
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            corr_matrix = metric_data[self.eval_dimensions].corr()
            correlation_data[metric] = corr_matrix
        
        # Calculate average correlation across all metrics
        avg_corr = pd.DataFrame(0, index=self.eval_dimensions, columns=self.eval_dimensions)
        for corr_matrix in correlation_data.values():
            avg_corr += corr_matrix
        avg_corr /= len(self.metrics)
        
        return correlation_data, avg_corr
    
    def create_radar_chart(self, save_path=None):
        """Create radar chart for multi-dimensional comparison."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(self.eval_dimensions), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            values = []
            for dim in self.eval_dimensions:
                if dim == 'hallucination_risk':
                    # Invert hallucination risk for better visualization (lower is better)
                    values.append(1 - (metric_data[dim].mean() / 3.0))
                else:
                    values.append(metric_data[dim].mean())
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=metric.title())
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        labels = [dim.replace('_', ' ').title() for dim in self.eval_dimensions]
        labels[-1] = 'Low Hallucination Risk'  # Rename for clarity
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.title('Multi-dimensional Performance Comparison', size=16, y=1.08)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_correlation_heatmap(self, save_path=None):
        """Create correlation heatmap between evaluation dimensions."""
        _, avg_corr = self.calculate_correlations()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(avg_corr, dtype=bool), k=1)
        
        sns.heatmap(avg_corr, mask=mask, annot=True, fmt='.3f', 
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, cbar_kws={"shrink": .8})
        
        plt.title('Average Correlation Between Evaluation Dimensions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_performance_scatter_plots(self, save_path=None):
        """Create scatter plots showing key performance trade-offs."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Answer Faithfulness vs Hallucination Risk
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            ax1.scatter(metric_data['answer_faithfulness'], 
                       metric_data['hallucination_risk'],
                       label=metric.title(), s=100, alpha=0.7)
        ax1.set_xlabel('Answer Faithfulness')
        ax1.set_ylabel('Hallucination Risk')
        ax1.set_title('Trade-off: Faithfulness vs Hallucination')
        ax1.legend()
        
        # 2. Answer Relevance vs Response Length
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            ax2.scatter(metric_data['answer_relevance'], 
                       metric_data['response_length'],
                       label=metric.title(), s=100, alpha=0.7)
        ax2.set_xlabel('Answer Relevance')
        ax2.set_ylabel('Response Length (chars)')
        ax2.set_title('Trade-off: Relevance vs Verbosity')
        ax2.legend()
        
        # 3. Context Utilization vs Context Relevance
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            ax3.scatter(metric_data['context_relevance'], 
                       metric_data['context_utilization'],
                       label=metric.title(), s=100, alpha=0.7)
        ax3.set_xlabel('Context Relevance')
        ax3.set_ylabel('Context Utilization')
        ax3.set_title('Context Usage Pattern')
        ax3.legend()
        
        # 4. Composite Score Components
        composite_scores = []
        for metric in self.metrics:
            metric_data = self.data[self.data['metric'] == metric]
            composite = (
                metric_data['context_relevance'].mean() * 0.25 +
                metric_data['answer_faithfulness'].mean() * 0.30 +
                metric_data['answer_relevance'].mean() * 0.30 +
                metric_data['context_utilization'].mean() * 0.15
            )
            composite_scores.append(composite)
        
        bars = ax4.bar([m.title() for m in self.metrics], composite_scores,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Overall Performance Ranking')
        ax4.set_ylim(0.6, 0.75)
        
        # Add value labels
        for bar, score in zip(bars, composite_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_statistical_summary(self):
        """Generate comprehensive statistical summary."""
        summary = []
        
        print("=" * 80)
        print("ENHANCED STATISTICAL ANALYSIS OF RAG DISTANCE METRICS")
        print("=" * 80)
        
        for eval_dim in self.eval_dimensions:
            print(f"\n### {eval_dim.replace('_', ' ').upper()} ###")
            
            # ANOVA test
            f_stat, p_value = self.perform_anova_test(eval_dim)
            print(f"\nANOVA Test: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print("✓ Significant differences exist between metrics")
                
                # Pairwise comparisons
                pairwise = self.pairwise_comparisons(eval_dim)
                print("\nPairwise Comparisons (with Bonferroni correction):")
                for comparison, results in pairwise.items():
                    if results['significant']:
                        print(f"  - {comparison}: p={results['p_value']:.4f}, "
                              f"effect size={results['effect_size']:.3f} "
                              f"({'large' if abs(results['effect_size']) > 0.8 else 'medium' if abs(results['effect_size']) > 0.5 else 'small'})")
            else:
                print("✗ No significant differences between metrics")
            
            # Confidence intervals
            print(f"\n95% Confidence Intervals:")
            for metric in self.metrics:
                metric_data = self.data[self.data['metric'] == metric][eval_dim].values
                mean, lower, upper = self.calculate_confidence_intervals(metric_data)
                print(f"  - {metric.title()}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
        
        return summary

# Example usage with simulated data based on your results
def create_sample_data():
    """Create sample data based on the visualization results."""
    np.random.seed(42)
    
    # Based on your heatmap values
    base_values = {
        'cosine': {'context_relevance': 0.333, 'answer_faithfulness': 0.900, 
                   'answer_relevance': 0.652, 'context_utilization': 1.000, 
                   'hallucination_risk': 2.80},
        'euclidean': {'context_relevance': 0.333, 'answer_faithfulness': 0.931, 
                      'answer_relevance': 0.701, 'context_utilization': 1.000, 
                      'hallucination_risk': 2.80},
        'manhattan': {'context_relevance': 0.333, 'answer_faithfulness': 0.927, 
                      'answer_relevance': 0.725, 'context_utilization': 1.000, 
                      'hallucination_risk': 2.20},
        'dot_product': {'context_relevance': 0.333, 'answer_faithfulness': 0.983, 
                        'answer_relevance': 0.663, 'context_utilization': 1.000, 
                        'hallucination_risk': 3.00}
    }
    
    # Response length distributions based on boxplot
    response_lengths = {
        'cosine': np.random.normal(1250, 200, 5),
        'euclidean': np.random.normal(1400, 250, 5),
        'manhattan': np.random.normal(1200, 300, 5),
        'dot_product': np.random.normal(1250, 350, 5)
    }
    
    data = []
    for metric, values in base_values.items():
        for i in range(5):  # 5 queries
            row = {
                'metric': metric,
                'query_id': i,
                'response_length': response_lengths[metric][i]
            }
            # Add small random variation to base values
            for key, value in values.items():
                if key != 'context_relevance' and key != 'context_utilization':  # These are constant
                    row[key] = value + np.random.normal(0, 0.02)
                else:
                    row[key] = value
            data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create sample data
    df = create_sample_data()
    
    # Initialize analyzer
    analyzer = EnhancedRAGAnalyzer(df)
    
    # Generate all analyses
    print("Generating enhanced RAG analysis...")
    
    # Statistical summary
    analyzer.generate_statistical_summary()
    
    # Create visualizations
    analyzer.create_radar_chart('radar_chart.png')
    analyzer.create_correlation_heatmap('correlation_heatmap.png')
    analyzer.create_performance_scatter_plots('performance_scatter.png')