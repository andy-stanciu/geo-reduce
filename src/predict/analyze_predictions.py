"""
analyze_predictions.py - Analyze model predictions and generate detailed metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def clean_tensor_values(df):
    """Convert tensor strings to floats"""
    for col in ['true_lat', 'true_lon']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float(str(x).split('(')[1].split(',')[0]) if 'tensor' in str(x) else float(x))
    return df


def compute_per_city_metrics(df):
    """Compute accuracy, confidence, and distance metrics per city"""
    cities = df['true_city'].unique()
    metrics = []
    
    for city in cities:
        city_df = df[df['true_city'] == city]
        
        total = len(city_df)
        correct = city_df['is_correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        avg_confidence = city_df['confidence'].mean()
        avg_distance = city_df['distance_error_km'].replace('N/A', np.nan).astype(float).mean()
        
        # Get most common misclassifications
        wrong_preds = city_df[~city_df['is_correct']]['predicted_city']
        top_mistake = wrong_preds.mode()[0] if len(wrong_preds) > 0 and not wrong_preds.empty else 'N/A'
        mistake_count = (wrong_preds == top_mistake).sum() if top_mistake != 'N/A' else 0
        
        metrics.append({
            'city': city,
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'avg_distance_km': avg_distance,
            'top_mistake': top_mistake,
            'mistake_count': mistake_count
        })
    
    return pd.DataFrame(metrics).sort_values('accuracy', ascending=False)


def plot_per_city_accuracy(metrics_df, save_dir):
    """Plot accuracy for each city"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort by accuracy
    metrics_df = metrics_df.sort_values('accuracy', ascending=True)
    
    colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' 
              for acc in metrics_df['accuracy']]
    
    ax.barh(metrics_df['city'], metrics_df['accuracy'] * 100, color=colors, alpha=0.7)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('City', fontsize=12)
    ax.set_title('Per-City Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.axvline(x=70, color='black', linestyle='--', alpha=0.3, label='70% threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_city_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: per_city_accuracy.png")


def plot_distance_errors(df, metrics_df, save_dir):
    """Plot distance error distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall distance distribution
    ax1 = axes[0, 0]
    distances = df['distance_error_km'].replace('N/A', np.nan).astype(float).dropna()
    ax1.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Distance Error (km)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Distance Error Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(distances.median(), color='red', linestyle='--', label=f'Median: {distances.median():.1f}km')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Distance error by city
    ax2 = axes[0, 1]
    top_cities = metrics_df.nlargest(15, 'total_samples')['city'].tolist()
    city_distances = [df[df['true_city'] == city]['distance_error_km'].replace('N/A', np.nan).astype(float).dropna().values 
                      for city in top_cities]
    ax2.boxplot(city_distances, labels=top_cities, vert=False)
    ax2.set_xlabel('Distance Error (km)', fontsize=12)
    ax2.set_ylabel('City', fontsize=12)
    ax2.set_title('Distance Error by City (Top 15)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Correct vs Incorrect distance errors
    ax3 = axes[1, 0]
    correct_dist = df[df['is_correct']]['distance_error_km'].replace('N/A', np.nan).astype(float).dropna()
    incorrect_dist = df[~df['is_correct']]['distance_error_km'].replace('N/A', np.nan).astype(float).dropna()
    
    ax3.hist([correct_dist, incorrect_dist], bins=50, label=['Correct', 'Incorrect'], 
             color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax3.set_xlabel('Distance Error (km)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distance Error: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Per-city average distance
    ax4 = axes[1, 1]
    metrics_sorted = metrics_df.sort_values('avg_distance_km', ascending=True).head(20)
    ax4.barh(metrics_sorted['city'], metrics_sorted['avg_distance_km'], color='steelblue', alpha=0.7)
    ax4.set_xlabel('Average Distance Error (km)', fontsize=12)
    ax4.set_ylabel('City', fontsize=12)
    ax4.set_title('Average Distance Error per City (Top 20)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distance_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: distance_errors.png")


def plot_confidence_analysis(df, save_dir):
    """Plot confidence-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Confidence distribution
    ax1 = axes[0, 0]
    ax1.hist(df['confidence'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["confidence"].mean():.3f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Confidence: Correct vs Incorrect
    ax2 = axes[0, 1]
    correct_conf = df[df['is_correct']]['confidence']
    incorrect_conf = df[~df['is_correct']]['confidence']
    ax2.hist([correct_conf, incorrect_conf], bins=50, 
             label=[f'Correct (μ={correct_conf.mean():.3f})', 
                    f'Incorrect (μ={incorrect_conf.mean():.3f})'],
             color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Accuracy by confidence bin
    ax3 = axes[1, 0]
    df['confidence_bin'] = pd.cut(df['confidence'], bins=10)
    accuracy_by_conf = df.groupby('confidence_bin', observed=True)['is_correct'].mean()
    bin_centers = [interval.mid for interval in accuracy_by_conf.index]
    ax3.plot(bin_centers, accuracy_by_conf.values * 100, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Confidence', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy vs Confidence (Calibration)', fontsize=14, fontweight='bold')
    ax3.plot([0, 1], [0, 100], 'r--', alpha=0.5, label='Perfect calibration')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Distance error vs confidence
    ax4 = axes[1, 1]
    valid_df = df[df['distance_error_km'] != 'N/A'].copy()
    valid_df['distance_error_km'] = valid_df['distance_error_km'].astype(float)
    ax4.scatter(valid_df['confidence'], valid_df['distance_error_km'], 
                alpha=0.3, s=10, c='steelblue')
    ax4.set_xlabel('Confidence', fontsize=12)
    ax4.set_ylabel('Distance Error (km)', fontsize=12)
    ax4.set_title('Distance Error vs Confidence', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confidence_analysis.png")


def plot_confusion_matrix(df, save_dir, top_n=20):
    """Plot confusion matrix for top N cities"""
    # Get top N most frequent cities
    top_cities = df['true_city'].value_counts().head(top_n).index.tolist()
    df_top = df[df['true_city'].isin(top_cities) & df['predicted_city'].isin(top_cities)]
    
    # Create confusion matrix
    confusion = pd.crosstab(df_top['true_city'], df_top['predicted_city'], normalize='index')
    
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted City', fontsize=12)
    ax.set_ylabel('True City', fontsize=12)
    ax.set_title(f'Confusion Matrix (Top {top_n} Cities)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: confusion_matrix.png")


def analyze_common_mistakes(df, save_dir, top_n=15):
    """Analyze most common misclassifications"""
    mistakes = df[~df['is_correct']][['true_city', 'predicted_city']]
    mistake_pairs = mistakes.groupby(['true_city', 'predicted_city']).size().reset_index(name='count')
    mistake_pairs = mistake_pairs.sort_values('count', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    labels = [f"{row['true_city']} → {row['predicted_city']}" 
              for _, row in mistake_pairs.iterrows()]
    
    ax.barh(labels, mistake_pairs['count'], color='coral', alpha=0.7)
    ax.set_xlabel('Number of Mistakes', fontsize=12)
    ax.set_ylabel('True City → Predicted City', fontsize=12)
    ax.set_title(f'Top {top_n} Most Common Misclassifications', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'common_mistakes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: common_mistakes.png")
    
    return mistake_pairs


def generate_summary_stats(df, metrics_df, save_dir):
    """Generate and save summary statistics"""
    summary = []
    
    summary.append("="*70)
    summary.append("PREDICTION ANALYSIS SUMMARY")
    summary.append("="*70)
    summary.append("")
    
    # Overall metrics
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = correct / total
    
    summary.append("Overall Metrics:")
    summary.append(f"  Total predictions: {total}")
    summary.append(f"  Correct: {correct} ({accuracy*100:.2f}%)")
    summary.append(f"  Incorrect: {total - correct} ({(1-accuracy)*100:.2f}%)")
    summary.append(f"  Average confidence: {df['confidence'].mean():.4f}")
    
    distances = df['distance_error_km'].replace('N/A', np.nan).astype(float).dropna()
    summary.append(f"  Average distance error: {distances.mean():.1f} km")
    summary.append(f"  Median distance error: {distances.median():.1f} km")
    summary.append("")
    
    # Best performing cities
    summary.append("Top 10 Best Performing Cities:")
    for i, row in metrics_df.head(10).iterrows():
        summary.append(f"  {row['city']:20s} - {row['accuracy']*100:5.1f}% "
                      f"({row['correct']}/{row['total_samples']}) "
                      f"Avg Dist: {row['avg_distance_km']:.1f}km")
    summary.append("")
    
    # Worst performing cities
    summary.append("Bottom 10 Worst Performing Cities:")
    for i, row in metrics_df.tail(10).iterrows():
        summary.append(f"  {row['city']:20s} - {row['accuracy']*100:5.1f}% "
                      f"({row['correct']}/{row['total_samples']}) "
                      f"Most confused with: {row['top_mistake']}")
    summary.append("")
    
    # Confidence analysis
    correct_conf = df[df['is_correct']]['confidence'].mean()
    incorrect_conf = df[~df['is_correct']]['confidence'].mean()
    summary.append("Confidence Analysis:")
    summary.append(f"  Average confidence (correct): {correct_conf:.4f}")
    summary.append(f"  Average confidence (incorrect): {incorrect_conf:.4f}")
    summary.append(f"  Confidence gap: {correct_conf - incorrect_conf:.4f}")
    summary.append("")
    
    # Save to file
    summary_text = "\n".join(summary)
    with open(os.path.join(save_dir, 'summary_stats.txt'), 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    print(f"\n✓ Saved: summary_stats.txt")


def main(csv_path, output_dir='./analysis'):
    """Main analysis function"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = clean_tensor_values(df)
    
    print(f"Loaded {len(df)} predictions")
    print(f"Overall accuracy: {df['is_correct'].mean()*100:.2f}%\n")
    
    # Compute per-city metrics
    print("Computing per-city metrics...")
    metrics_df = compute_per_city_metrics(df)
    metrics_df.to_csv(os.path.join(output_dir, 'per_city_metrics.csv'), index=False)
    print(f"✓ Saved: per_city_metrics.csv\n")
    
    # Generate plots
    print("Generating visualizations...")
    plot_per_city_accuracy(metrics_df, output_dir)
    plot_distance_errors(df, metrics_df, output_dir)
    plot_confidence_analysis(df, output_dir)
    plot_confusion_matrix(df, output_dir, top_n=20)
    
    # Analyze mistakes
    print("\nAnalyzing common mistakes...")
    mistake_pairs = analyze_common_mistakes(df, output_dir, top_n=20)
    mistake_pairs.to_csv(os.path.join(output_dir, 'common_mistakes.csv'), index=False)
    print(f"✓ Saved: common_mistakes.csv")
    
    # Generate summary
    print("\nGenerating summary statistics...")
    generate_summary_stats(df, metrics_df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze prediction results')
    parser.add_argument('--csv', type=str, required=True, 
                       help='Path to predictions CSV file')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    main(args.csv, args.output_dir)
