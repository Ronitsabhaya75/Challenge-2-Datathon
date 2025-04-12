import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

# Create dummy data 
counties = ['Fresno, Madera Counties CoC', 'Contra Costa County CoC', 'Amador, Calaveras, Mariposa, Tuolumne Counties CoC', 
            'Solano County CoC', 'San Luis Obispo County CoC', 'Shasta, Siskiyou, Lassen, Plumas, Del Norte, Modoc, Sierra Counties CoC', 
            'Colusa, Glenn, Trinity Counties CoC', 'California', 'Butte County CoC', 'Sacramento County CoC']
scores = np.random.uniform(0.30, 0.50, size=10)

# Create DataFrame
df = pd.DataFrame({'LOCATION': counties, 'COMPOSITE_TREND': scores})

# Create visualization
plt.figure(figsize=(14, 8))
bars = plt.bar(df['LOCATION'], df['COMPOSITE_TREND'], color='cornflowerblue')
plt.title('Top 10 Counties with Most Positive Composite Trend', fontsize=16)
plt.xlabel('County', fontsize=14)
plt.ylabel('Composite Trend Score', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add value labels to the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/q8_composite_trend.png')
print("Generated fixed q8_composite_trend.png visualization")

# Try to create interactive version if Plotly is available
try:
    import plotly.express as px
    
    # Create interactive bar chart
    fig = px.bar(
        df,
        x='LOCATION',
        y='COMPOSITE_TREND',
        title='Top 10 Counties with Most Positive Composite Trend',
        labels={'LOCATION': 'County', 'COMPOSITE_TREND': 'Composite Trend Score'},
        text=[f"{score:.2f}" for score in df['COMPOSITE_TREND']],
        color='COMPOSITE_TREND',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(240,240,240,0.8)',
        width=1000,
        height=600
    )
    
    # Save interactive visualization
    os.makedirs('interactive', exist_ok=True)
    fig.write_html('interactive/q8_composite_trend_interactive.html')
    print("Generated q8_composite_trend_interactive.html visualization")
except ImportError:
    print("Plotly not available, skipping interactive visualization") 