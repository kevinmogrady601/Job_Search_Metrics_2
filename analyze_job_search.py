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
    display_cols = ['Date', 'Company', 'Title', 'Quality', 'Local/Remote', 'Closed']
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
        colWidths=[0.1, 0.2, 0.3, 0.1, 0.15, 0.15]
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
        'Closed Positions': len(df[df['Closed'] == 'Y']),
        'Open Positions': len(df[df['Closed'] == 'N']),
        'Unknown Status Positions': len(df[df['Closed'] == 'I']),
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

def plot_closed_positions_distribution(df):
    """Create a visualization showing the distribution of position closure status."""
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of position status
    status_counts = df['Closed'].value_counts()
    status_labels = {
        'Y': 'Closed',
        'N': 'Open', 
        'I': 'Unknown/In Progress'
    }
    
    # Map the values to readable labels
    labels = [status_labels.get(status, status) for status in status_counts.index]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, teal, blue
    
    ax1.pie(status_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Position Status Distribution')
    
    # Bar chart of closure rate by quality
    quality_closure = []
    quality_labels = []
    
    for quality in sorted(df['Quality'].unique()):
        quality_df = df[df['Quality'] == quality]
        closed_count = len(quality_df[quality_df['Closed'] == 'Y'])
        total_count = len(quality_df)
        if total_count > 0:
            closure_rate = closed_count / total_count * 100
            quality_closure.append(closure_rate)
            quality_labels.append(f'Quality {quality}')
    
    bars = ax2.bar(quality_labels, quality_closure, color=['#4ECDC4', '#FFE66D', '#FF6B6B'])
    ax2.set_title('Position Closure Rate by Quality')
    ax2.set_ylabel('Closure Rate (%)')
    ax2.set_xlabel('Quality Rating')
    
    # Add value labels on bars
    for bar, rate in zip(bars, quality_closure):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('closed_positions_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_closed_positions(df):
    """Analyze patterns in closed vs open positions."""
    print("\n=== Position Closure Analysis ===")
    
    # Overall closure rates
    total_positions = len(df)
    closed_positions = len(df[df['Closed'] == 'Y'])
    open_positions = len(df[df['Closed'] == 'N'])
    unknown_positions = len(df[df['Closed'] == 'I'])
    
    print(f"\nPosition Status Distribution:")
    print(f"Closed: {closed_positions} ({closed_positions/total_positions*100:.1f}%)")
    print(f"Open: {open_positions} ({open_positions/total_positions*100:.1f}%)")
    print(f"Unknown/In Progress: {unknown_positions} ({unknown_positions/total_positions*100:.1f}%)")
    
    # Closure rate by quality
    print(f"\nClosure Rate by Quality:")
    for quality in sorted(df['Quality'].unique()):
        quality_df = df[df['Quality'] == quality]
        quality_closed = len(quality_df[quality_df['Closed'] == 'Y'])
        quality_total = len(quality_df)
        if quality_total > 0:
            closure_rate = quality_closed / quality_total * 100
            print(f"Quality {quality}: {quality_closed}/{quality_total} ({closure_rate:.1f}%)")
    
    # Interview success for closed vs open positions
    closed_df = df[df['Closed'] == 'Y']
    open_df = df[df['Closed'] == 'N']
    
    if len(closed_df) > 0:
        closed_interview_rate = len(closed_df[closed_df['Interviews'] == 'Y']) / len(closed_df) * 100
        print(f"\nInterview Rate for Closed Positions: {closed_interview_rate:.1f}%")
    
    if len(open_df) > 0:
        open_interview_rate = len(open_df[open_df['Interviews'] == 'Y']) / len(open_df) * 100
        print(f"Interview Rate for Open Positions: {open_interview_rate:.1f}%")
    
    # Time analysis - when were positions closed?
    if len(closed_df) > 0:
        print(f"\nClosed Position Timeline:")
        closed_by_month = closed_df.resample('M', on='Date').size()
        for date, count in closed_by_month.items():
            if count > 0:
                print(f"{date.strftime('%B %Y')}: {count} positions closed")

def generate_html_dashboard(df):
    """Generate an HTML dashboard with current metrics and charts."""
    # Calculate current metrics
    total_apps = len(df)
    unique_companies = df['Company'].nunique()
    interviews = len(df[df['Interviews'] == 'Y'])
    interview_rate = (interviews / total_apps * 100) if total_apps > 0 else 0
    recruiters = len(df[df['Recruiter'] == 'Y'])
    avg_quality = df['Quality'].mean()
    
    closed_positions = len(df[df['Closed'] == 'Y'])
    open_positions = len(df[df['Closed'] == 'N'])
    unknown_positions = len(df[df['Closed'] == 'I'])
    
    closed_pct = (closed_positions / total_apps * 100) if total_apps > 0 else 0
    open_pct = (open_positions / total_apps * 100) if total_apps > 0 else 0
    unknown_pct = (unknown_positions / total_apps * 100) if total_apps > 0 else 0
    
    # Calculate interview rates by quality
    interview_rates_by_quality = {}
    for quality in sorted(df['Quality'].unique()):
        quality_df = df[df['Quality'] == quality]
        quality_interviews = len(quality_df[quality_df['Interviews'] == 'Y'])
        quality_total = len(quality_df)
        if quality_total > 0:
            rate = quality_interviews / quality_total * 100
            interview_rates_by_quality[quality] = rate
    
    # Calculate recruiter impact
    recruiter_interview_rate = 0
    no_recruiter_interview_rate = 0
    
    recruiter_df = df[df['Recruiter'] == 'Y']
    no_recruiter_df = df[df['Recruiter'] == 'N']
    
    if len(recruiter_df) > 0:
        recruiter_interview_rate = len(recruiter_df[recruiter_df['Interviews'] == 'Y']) / len(recruiter_df) * 100
    
    if len(no_recruiter_df) > 0:
        no_recruiter_interview_rate = len(no_recruiter_df[no_recruiter_df['Interviews'] == 'Y']) / len(no_recruiter_df) * 100
    
    # Generate the HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job Search Metrics Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}

        .header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 300;
        }}

        .header p {{
            color: #7f8c8d;
            font-size: 1.1rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 0.5rem;
        }}

        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .charts-section {{
            margin-bottom: 2rem;
        }}

        .section-title {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}

        .section-title h2 {{
            color: #2c3e50;
            font-size: 1.5rem;
            font-weight: 400;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 2rem;
        }}

        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}

        .chart-container:hover {{
            transform: translateY(-3px);
        }}

        .chart-container h3 {{
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 500;
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .table-container {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            overflow-x: auto;
        }}

        .table-container h3 {{
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 500;
        }}

        .analysis-section {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }}

        .analysis-section h3 {{
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.3rem;
            font-weight: 500;
        }}

        .insight-item {{
            background: #f8f9fa;
            padding: 1rem;
            border-left: 4px solid #3498db;
            margin-bottom: 1rem;
            border-radius: 0 8px 8px 0;
        }}

        .insight-item h4 {{
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}

        .insight-item p {{
            color: #555;
            line-height: 1.5;
        }}

        .footer {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-top: 2rem;
        }}

        .footer p {{
            color: #7f8c8d;
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            
            .chart-container {{
                min-width: auto;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }}
        }}

        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .status-closed {{ background-color: #e74c3c; }}
        .status-open {{ background-color: #27ae60; }}
        .status-unknown {{ background-color: #f39c12; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Job Search Metrics Dashboard</h1>
            <p>Comprehensive analysis of job applications, interviews, and position tracking</p>
        </div>

        <!-- Key Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_apps}</div>
                <div class="metric-label">Total Applications</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{unique_companies}</div>
                <div class="metric-label">Unique Companies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{interviews}</div>
                <div class="metric-label">Interviews Secured</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{interview_rate:.1f}%</div>
                <div class="metric-label">Interview Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{recruiters}</div>
                <div class="metric-label">Recruiter Contacts</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_quality:.2f}</div>
                <div class="metric-label">Avg Quality Score</div>
            </div>
        </div>

        <!-- Position Status Metrics -->
        <div class="section-title">
            <h2>Position Status Overview</h2>
        </div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" style="color: #e74c3c;">{closed_positions}</div>
                <div class="metric-label">
                    <span class="status-indicator status-closed"></span>Closed Positions ({closed_pct:.1f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #27ae60;">{open_positions}</div>
                <div class="metric-label">
                    <span class="status-indicator status-open"></span>Open Positions ({open_pct:.1f}%)
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #f39c12;">{unknown_positions}</div>
                <div class="metric-label">
                    <span class="status-indicator status-unknown"></span>Unknown Status ({unknown_pct:.1f}%)
                </div>
            </div>
        </div>

        <!-- Charts Section -->
        <div class="charts-section">
            <div class="section-title">
                <h2>Visualizations & Trends</h2>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <h3>Applications Over Time</h3>
                    <img src="applications_over_time.png" alt="Applications Over Time Chart">
                </div>
                
                <div class="chart-container">
                    <h3>Quality Distribution</h3>
                    <img src="quality_distribution.png" alt="Quality Distribution Chart">
                </div>
                
                <div class="chart-container">
                    <h3>Interviews Per Month</h3>
                    <img src="interviews_per_month.png" alt="Interviews Per Month Chart">
                </div>
                
                <div class="chart-container">
                    <h3>High Quality Interviews Per Month</h3>
                    <img src="high_quality_interviews_per_month.png" alt="High Quality Interviews Per Month Chart">
                </div>
                
                <div class="chart-container">
                    <h3>Position Status Distribution</h3>
                    <img src="closed_positions_distribution.png" alt="Closed Positions Distribution Chart">
                </div>
            </div>
        </div>

        <!-- High Quality Interview Table -->
        <div class="table-container">
            <h3>High Quality Positions with Interviews</h3>
            <img src="high_quality_interview_table.png" alt="High Quality Interview Table" style="width: 100%; height: auto;">
        </div>

        <!-- Key Insights Section -->
        <div class="analysis-section">
            <h3>Key Insights & Analysis</h3>
            
            <div class="insight-item">
                <h4>Interview Success by Quality</h4>
                <p>'''
    
    # Add quality-specific interview rates
    quality_text = " | ".join([f"<strong>Quality {q}:</strong> {r:.1f}% interview rate" for q, r in interview_rates_by_quality.items()])
    html_content += quality_text + "</p>\n            </div>\n            \n"
    
    html_content += f'''            <div class="insight-item">
                <h4>Recruiter Impact</h4>
                <p>Applications with recruiter involvement have a <strong>{recruiter_interview_rate:.1f}%</strong> interview rate compared to <strong>{no_recruiter_interview_rate:.1f}%</strong> without recruiters, highlighting the critical importance of networking and recruiter relationships.</p>
            </div>
            
            <div class="insight-item">
                <h4>Position Closure Patterns</h4>
                <p>Position closure analysis shows {closed_positions} confirmed closed positions ({closed_pct:.1f}%) out of {total_apps} total applications. Most positions ({unknown_pct:.1f}%) still have unknown status, indicating potential for future developments.</p>
            </div>
            
            <div class="insight-item">
                <h4>Geographic Distribution</h4>
                <p><strong>Remote positions:</strong> {len(df[df['Local/Remote'] == 'Remote'])} applications | <strong>Local positions:</strong> {len(df[df['Local/Remote'] == 'Local'])} applications. Strategic focus on location preferences can optimize application success.</p>
            </div>
            
            <div class="insight-item">
                <h4>Market Activity</h4>
                <p>With <strong>{unknown_pct:.1f}%</strong> of positions still having unknown status, there's significant potential for future developments. The data shows consistent application activity with strategic focus on higher-quality opportunities.</p>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Dashboard generated from job search data analysis | Last updated: <span id="lastUpdated"></span></p>
        </div>
    </div>

    <script>
        // Set the last updated date
        document.getElementById('lastUpdated').textContent = new Date().toLocaleDateString();
        
        // Add smooth scrolling for better UX
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
        
        // Add loading animation for images
        document.querySelectorAll('img').forEach(img => {{
            img.addEventListener('load', function() {{
                this.style.opacity = '1';
                this.style.transform = 'scale(1)';
            }});
            
            img.style.opacity = '0';
            img.style.transform = 'scale(0.95)';
            img.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        }});
    </script>
</body>
</html>'''
    
    # Write the HTML file
    with open('job_search_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n=== HTML Dashboard Generated ===")
    print("Dashboard saved as 'job_search_dashboard.html'")
    print("Open this file in your web browser to view the interactive dashboard.")

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
    plot_closed_positions_distribution(df)  # New visualization for closed positions
    
    # Analyze interview success and position closure
    analyze_interview_success(df)
    analyze_closed_positions(df)
    
    # Generate HTML dashboard
    generate_html_dashboard(df)

if __name__ == "__main__":
    main() 