import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/scripts"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/utils"))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_data(file_path, model_name):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data.values())
        df['model'] = model_name
        return df
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
        return pd.DataFrame()


def plot_metrics(df):
    """Generates and saves separate plots to visualize model performance."""
    if df.empty:
        print("DataFrame is empty. No plots will be generated.")
        return

    sns.set_theme(style="whitegrid")

    # # Plot 1: Boxplots for Precision, Recall, F1
    # plt.figure(figsize=(12, 8))
    # metrics_to_plot = ['levenshtein_precision', 'levenshtein_recall', 'levenshtein_f1']
    # plot_df = df.melt(id_vars='model', value_vars=metrics_to_plot, var_name='metric', value_name='score')
    #
    # sns.boxplot(x='metric', y='score', hue='model', data=plot_df, palette='viridis')
    # plt.title('Distribution of Performance Scores', fontsize=16, weight='bold')
    # plt.xlabel('Metric', fontsize=12)
    # plt.ylabel('Score', fontsize=12)
    # plt.xticks(rotation=15)
    # plt.legend(title='Model')
    # plt.tight_layout()
    # plt.savefig(os.path.join(IMAGES_DIR, 'performance_scores_distribution.png'), dpi=300)
    # plt.close()
    # print("Plot saved as performance_scores_distribution.png")
    #
    # # Plot 2: Bar chart for Mean Scores
    # plt.figure(figsize=(12, 8))
    # mean_scores = df.groupby('model')[metrics_to_plot].mean().reset_index()
    # mean_scores_melted = mean_scores.melt(id_vars='model', var_name='metric', value_name='average_score')
    #
    # sns.barplot(x='metric', y='average_score', hue='model', data=mean_scores_melted, palette='viridis')
    # plt.title('Average Performance Scores', fontsize=16, weight='bold')
    # plt.xlabel('Metric', fontsize=12)
    # plt.ylabel('Average Score', fontsize=12)
    # plt.xticks(rotation=15)
    # plt.ylim(0, 1)
    # plt.legend(title='Model')
    # plt.tight_layout()
    # plt.savefig(os.path.join(IMAGES_DIR, 'average_performance_scores.png'), dpi=300)
    # plt.close()
    # print("Plot saved as average_performance_scores.png")
    metrics_to_plot = ['levenshtein_precision', 'levenshtein_recall', 'levenshtein_f1']
    plt.figure(figsize=(12, 8))
    mean_scores = df.groupby('model')[metrics_to_plot].mean().reset_index()
    mean_scores_melted = mean_scores.melt(id_vars='model', var_name='metric', value_name='average_score')

    ax = sns.barplot(x='metric', y='average_score', hue='model', data=mean_scores_melted, palette='viridis')

    # Add the percentage on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}',  # Format the number as a percentage with 2 decimal places
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the text
                    ha='center', va='center',  # Alignment
                    xytext=(0, 10),  # Offset text by 10 points
                    textcoords='offset points',
                    fontsize=14)

    plt.title('Average Performance Scores', fontsize=16, weight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.xticks(rotation=15)
    plt.ylim(0, 1.1)  # Extend the y-axis limit to make room for the percentages
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'average_performance_scores.png'), dpi=300)
    plt.close()
    print("Plot saved as average_performance_scores.png")



    # Plot 4: JSON Validity Rate
    plt.figure(figsize=(12, 8))

    validity_data = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        if 'is_json_valid' in model_df.columns:
            valid_count = model_df['is_json_valid'].sum() if not model_df.empty else 0
            total_count = len(model_df)
            validity_rate = valid_count / total_count if total_count > 0 else 0
            validity_data.append({'model': model, 'valid_rate': validity_rate})
        else:
            validity_data.append({'model': model, 'valid_rate': 0})

    validity_df = pd.DataFrame(validity_data)

    ax = sns.barplot(x='model', y='valid_rate', data=validity_df, palette='viridis')
    plt.title('JSON Output Validity Rate', fontsize=16, weight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Rate of Valid JSONs', fontsize=12)
    plt.ylim(0, 1.1)

    for index, row in validity_df.iterrows():
        ax.text(index, row.valid_rate + 0.02, f"{row.valid_rate:.2%}",
                color='black', ha="center", fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'json_validity_rate.png'), dpi=300)
    plt.close()
    print("Plot saved as json_validity_rate.png")


def main():
    # List of model files and their names
    model_files = {
        'gemma-3-270m': os.path.join(PROJECT_ROOT,'data', 'test_results_full_finetuned_gemma-3-270m-it_150.json'),
        'gemma-3-1b': os.path.join(PROJECT_ROOT,'data', 'test_results_lora_finetuned_gemma-3-1b-it-4bit_epoch_4.0.json'),
        'gemma-3-4b': os.path.join(PROJECT_ROOT,'data', 'test_results_lora_finetuned_gemma-3-4b-it-4bit_epoch_1.0.json'),
    }

    # Load and combine data from all files
    all_data = [load_data(path, name) for name, path in model_files.items()]
    combined_df = pd.concat(all_data, ignore_index=True)

    # Generate and save the plots
    plot_metrics(combined_df)


if __name__ == '__main__':
    main()