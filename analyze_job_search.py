import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn')
sns.set_palette("husl")

def load_data():
    """Load and preprocess the CSV data."""
    df = pd.read_csv('Resumes_Submissions_Submitted.csv')
    
    # Convert date string to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    return df

def plot_high_quality_interview_table(df):
    """Create a table visualization of high-quality jobs (Quality 1-2) that resulted in interviews."""
    # Filter for high quality interviews
    mask = (df['Quality'].isin([1, 2])) & (df['Interviews'] == 'Y')
    high_quality_interviews = df[mask].copy()
    
    # Sort by date
    high_quality_interviews = high_quality_interviews.sort_values('Date')
    
    # Format date for display
    high_quality_interviews['Date'] = high_quality_interviews['Date'].dt.strftime('%m/%d/%Y')
    
    # Select and rename columns for display
    display_cols = ['Date', 'Company', 'Title', 'Quality', 'Local/Remote']
    table_data = high_quality_interviews[display_cols].values
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, len(table_data) * 0.5 + 1))  # Adjust height based on number of rows
    
    # Remove axis
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=display_cols,
        cellLoc='left',
        loc='center',
        colWidths=[0.1, 0.2, 0.4, 0.1, 0.2]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Color the header row
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#E6E6E6')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color rows based on Quality
    for i in range(len(table_data)):
        quality = table_data[i][3]  # Quality column
        row_color = '#E8F5E9' if quality == 1 else '#FFF9C4'  # Light green for Q1, light yellow for Q2
        for j in range(len(display_cols)):
            table[(i + 1, j)].set_facecolor(row_color)
    
    # Adjust cell heights
    table.scale(1, 1.5)
    
    plt.title('High Quality Jobs with Interviews', pad=20)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('high_quality_interview_table.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_basic_metrics(df):
    """Generate basic metrics about the job search."""
    metrics = {
        'Total Applications': len(df),
        'Unique Companies': df['Company'].nunique(),
        'Applications with Interviews': len(df[df['Interviews'] == 'Y']),
        'Applications with Recruiters': len(df[df['Recruiter'] == 'Y']),
        'Remote Positions': len(df[df['Local/Remote'] == 'Remote']),
        'Local Positions': len(df[df['Local/Remote'] == 'Local']),
        'Average Quality Score': df['Quality'].mean()
    }
    
    print("\n=== Basic Metrics ===")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

def plot_applications_over_time(df):
    """Create a plot showing applications over time."""
    plt.figure(figsize=(12, 6))
    
    # Create applications per month
    monthly_apps = df.resample('M', on='Date').size()
    
    plt.plot(monthly_apps.index, monthly_apps.values, marker='o')
    plt.title('Applications Submitted Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Applications')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('applications_over_time.png')
    plt.close()

def plot_interviews_per_month(df):
    """Create a plot showing interviews per month."""
    plt.figure(figsize=(12, 6))
    
    # Create a mask for interviews
    interviews_mask = df['Interviews'] == 'Y'
    
    # Get interviews per month
    monthly_interviews = df[interviews_mask].resample('M', on='Date').size()
    
    # Create x-axis labels with month names
    month_labels = monthly_interviews.index.strftime('%B %Y')
    
    # Create x-axis positions
    x_positions = range(len(monthly_interviews))
    
    # Plot the data
    plt.bar(x_positions, monthly_interviews.values, color='green', alpha=0.7)
    plt.title('Interviews Per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Interviews')
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, month_labels, rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for i, v in enumerate(monthly_interviews.values):
        if v > 0:  # Only add label if there were interviews
            plt.text(i, v, str(int(v)), 
                    ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('interviews_per_month.png')
    plt.close()

def plot_high_quality_interviews_per_month(df):
    """Create a plot showing interviews per month for positions with Quality 1 or 2."""
    plt.figure(figsize=(12, 6))
    
    # Filter for interviews only
    interviews_df = df[df['Interviews'] == 'Y'].copy()
    
    # Create separate dataframes for Quality 1 and 2
    q1_interviews = interviews_df[interviews_df['Quality'] == 1]
    q2_interviews = interviews_df[interviews_df['Quality'] == 2]
    
    # Get monthly counts for each quality
    monthly_q1 = q1_interviews.resample('M', on='Date').size()
    monthly_q2 = q2_interviews.resample('M', on='Date').size()
    
    # Ensure both series have the same index
    all_months = sorted(list(set(monthly_q1.index) | set(monthly_q2.index)))
    monthly_q1 = monthly_q1.reindex(all_months, fill_value=0)
    monthly_q2 = monthly_q2.reindex(all_months, fill_value=0)
    
    # Create x-axis labels and positions
    month_labels = [d.strftime('%B %Y') for d in all_months]
    x_positions = range(len(all_months))
    
    # Create the stacked bar chart - Quality 2 at bottom, Quality 1 on top
    plt.bar(x_positions, monthly_q2.values, color='yellow', alpha=0.7, label='Quality 2')
    plt.bar(x_positions, monthly_q1.values, bottom=monthly_q2.values, color='green', alpha=0.7, label='Quality 1')
    
    plt.title('High Quality Interviews Per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Interviews')
    plt.legend()
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, month_labels, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i in range(len(all_months)):
        q1_val = monthly_q1.values[i]
        q2_val = monthly_q2.values[i]
        total = q1_val + q2_val
        
        if total > 0:
            # If there's a mix of qualities, show both numbers
            if q1_val > 0 and q2_val > 0:
                plt.text(i, total, f'Q1:{int(q1_val)}\nQ2:{int(q2_val)}', 
                        ha='center', va='bottom')
            # If it's only Quality 1
            elif q1_val > 0:
                plt.text(i, q1_val, str(int(q1_val)), 
                        ha='center', va='bottom')
            # If it's only Quality 2
            elif q2_val > 0:
                plt.text(i, q2_val, str(int(q2_val)), 
                        ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('high_quality_interviews_per_month.png')
    plt.close()

def plot_quality_distribution(df):
    """Create a plot showing the distribution of job quality ratings."""
    plt.figure(figsize=(10, 6))
    
    sns.countplot(data=df, x='Quality')
    plt.title('Distribution of Job Quality Ratings')
    plt.xlabel('Quality Rating')
    plt.ylabel('Number of Applications')
    plt.tight_layout()
    plt.savefig('quality_distribution.png')
    plt.close()

def analyze_interview_success(df):
    """Analyze factors related to interview success."""
    print("\n=== Interview Success Analysis ===")
    
    # Interview rate by quality
    interview_rate_by_quality = df.groupby('Quality')['Interviews'].apply(
        lambda x: (x == 'Y').mean() * 100
    )
    print("\nInterview Rate by Quality Rating:")
    for quality, rate in interview_rate_by_quality.items():
        print(f"Quality {quality}: {rate:.1f}%")
    
    # Interview rate with/without recruiter
    recruiter_interview_rate = df[df['Recruiter'] == 'Y']['Interviews'].apply(
        lambda x: x == 'Y'
    ).mean() * 100
    no_recruiter_interview_rate = df[df['Recruiter'] == 'N']['Interviews'].apply(
        lambda x: x == 'Y'
    ).mean() * 100
    
    print(f"\nInterview Rate with Recruiter: {recruiter_interview_rate:.1f}%")
    print(f"Interview Rate without Recruiter: {no_recruiter_interview_rate:.1f}%")

def main():
    # Load the data
    df = load_data()
    
    # Generate and display metrics
    generate_basic_metrics(df)
    
    # Create visualizations
    plot_applications_over_time(df)
    plot_quality_distribution(df)
    plot_interviews_per_month(df)
    plot_high_quality_interviews_per_month(df)
    plot_high_quality_interview_table(df)  # Added new visualization
    
    # Analyze interview success
    analyze_interview_success(df)

if __name__ == "__main__":
    main() 