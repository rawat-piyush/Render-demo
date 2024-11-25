import pandas as pd 
import dash
from dash import dcc, html, dash_table
import base64
from dash.dependencies import Input, Output
import geopandas as gpd
import random
from datetime import datetime, timedelta
import plotly.express as px
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt 
import base64
from nrclex import NRCLex

df = pd.read_csv('Twitter_Data.csv')
df1 = df[60000: 80000]
indian_states_map = gpd.read_file('states_india.geojson')
indian_states_map.loc[31, 'st_nm'] = 'Dadra and Nagar Haveli'
india_states = indian_states_map['st_nm']
random_states = random.choices(india_states, k=20000)
df1['states'] = random_states

def random_date(start_date, end_date):
    time_delta = end_date - start_date
    random_seconds = random.randint(0, int(time_delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)
# Define start and end date (from 01-01-2023 to 31-12-2023)
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
random_dates = [random_date(start_date, end_date) for _ in range(20000)]
df1['timestamp'] = random_dates
df1['timestamp'] = pd.to_datetime(df1['timestamp'])

df1['month'] = df1['timestamp'].dt.to_period('M')
monthly_trend = df1.groupby('month')['category'].mean().reset_index()
monthly_trend['month'] = monthly_trend['month'].dt.to_timestamp()

# Plot the trend line using Plotly
fig = px.line(monthly_trend, x='month', y='category', title="Monthly Trend of Sentiment across India",
              labels={'month': 'Month', 'category': 'Average Category'})
# Show the plot
#fig.show()

monthly_trend['states'] = 'All states'
monthly_trend = monthly_trend[['states'] + [col for col in monthly_trend.columns if col != 'states']]

monthly_trend_states = df1.groupby(['states', 'month'])['category'].mean().reset_index()


# Convert the 'month' column back to a timestamp to make it easier to plot
monthly_trend_states['month'] = monthly_trend_states['month'].dt.to_timestamp()

monthly_trend_states = pd.concat([monthly_trend_states, monthly_trend], ignore_index=True)
# Plot the trend line for each state

fig = px.line(
    monthly_trend_states,
    x='month',
    y='category',
    color='states',
    title="Monthly Trend of Sentiment for Each State",
    labels={'month': 'Month', 'category': 'Average Category', 'states': 'State'}
)

#fig.show()

state_sentiment = df1.groupby('states')['category'].mean().reset_index()

# Filter for rows where sentiment is -1
negative_sentiment_df1 = df1[df1['category'] == -1]

# Group by 'state' and count the occurrences of -1 sentiment for each state
state_negative_counts = negative_sentiment_df1.groupby('states').size().reset_index(name='negative_count')

# Sort in descending order to get top states with the most -1 sentiment
top_negative_states = state_negative_counts.sort_values(by='negative_count', ascending=False).reset_index(drop= True)

top_negative_states.index += 1

# Display the top states (top 10 for example)
print(top_negative_states.head(10))

fig = px.bar(
    top_negative_states.head(10),  # Use only the top 10 states
    x='negative_count',  # Values for the x-axis
    y='states',          # Values for the y-axis
    orientation='h',     # Horizontal orientation
    title='Top 10 States with Most Negative Sentiments',
    labels={'states': 'States', 'negative_count': 'Negative Sentiment Count'},    
)

positive_sentiment_df1 = df1[df1['category'] == 1]

# Group by 'state' and count the occurrences of -1 sentiment for each state
state_positive_counts = positive_sentiment_df1.groupby('states').size().reset_index(name='positive_count')

# Sort in descending order to get top states with the most 1 sentiment
top_positive_states = state_positive_counts.sort_values(by='positive_count', ascending=False).reset_index(drop =True)

top_positive_states.index += 1

fig = px.bar(
    top_positive_states.head(10),  # Use only the top 10 states
    x='positive_count',  # Values for the x-axis
    y='states',          # Values for the y-axis
    orientation='h',     # Horizontal orientation
    title='Top 10 States with Most Positive Sentiments',
    labels={'states': 'States', 'positive_count': 'Positive Sentiment Count'},    
)

neutral_sentiment_df1 = df1[df1['category'] == 0]

# Group by 'state' and count the occurrences of -1 sentiment for each state
state_neutral_counts = neutral_sentiment_df1.groupby('states').size().reset_index(name='neutral_count')

# Sort in descending order to get top states with the most 1 sentiment
top_neutral_states = state_neutral_counts.sort_values(by='neutral_count', ascending=False).reset_index(drop =True)

top_neutral_states.index += 1

# Calculate sentiment percentages
sentiment_counts = df1['category'].value_counts()
total_rows = len(df1)
sentiment_percentages = (sentiment_counts / total_rows) * 100

# Convert to a DataFrame for easier manipulation
sentiment_percentages_df1 = sentiment_percentages.reset_index()
sentiment_percentages_df1.columns = ['sentiment', 'percentage']

# Mapping sentiment codes to readable labels
sentiment_map = {
    -1: 'Negative',
    0: 'Neutral',
    1: 'Positive'
}
sentiment_percentages_df1['sentiment'] = sentiment_percentages_df1['sentiment'].map(sentiment_map)

# Create a pie chart using Plotly
fig_pie = px.pie(
    sentiment_percentages_df1, 
    values='percentage', 
    names='sentiment', 
    title='Sentiment Distribution',
    color='sentiment',  # Optional: Add color differentiation
    color_discrete_map={'Negative': '#ff9999', 'Neutral': '#66b3ff', 'Positive': '#99ff99'}  # Custom color scheme
)

# Show the chart
#fig_pie.show()

def preprocess_text(text):
    #print("Original Text:", text)
    if not isinstance(text, str):
        return ''

    # Remove HTML tags
    text = re.sub('<[^>]*>', '', text)

#    print("Text after removing HTML tags:", text)

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()

#    print("Text after removing non-alphabetic characters and converting to lowercase:", text)

    # Tokenize the text
    words = word_tokenize(text)

 #   print("Text after tokenization:", words)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.discard('no')
    stop_words.discard('not')
    
    words = [word for word in words if word not in stop_words]

  #  print("Text after removing stop words:", words)

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

   # print("Text after Lemmatization:", words)

    # Combine words back into a single string
    preprocessed_text = ' '.join(words)

    #print("Final pre-processed text:", preprocessed_text)

    return preprocessed_text

df1['preprocessed_text'] = df1['clean_text'].apply(preprocess_text)
all_text = ' '.join(df1['preprocessed_text'])

# Generate word frequencies
word_counts = Counter(all_text.split())

# Generate the word cloud
wordcloud = WordCloud(
    width=800,               # Increase width
    height=800,              # Increase height
    max_words=150,            # Reduce max_words if needed
    background_color='white',
    contour_color='black',
    contour_width=1
).generate_from_frequencies(word_counts)

# Display the word cloud
plt.figure(figsize=(6,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
wordcloud_img_path = 'wordcloud.png'
wordcloud.to_file(wordcloud_img_path)

# Clean and tokenize the text
def preprocess_text(text):
    # Remove any non-alphabetic characters (punctuation, numbers, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize the text
    return [word for word in words if word not in stopwords.words('english')]

# Apply the cleaning and tokenization function to each tweet
df1['tokens'] = df1['clean_text'].apply(preprocess_text)

# Combine all tokens into a single list
all_words = [word for tokens in df1['tokens'] for word in tokens]

# Calculate word frequencies
word_counts = Counter(all_words)

# Get the top 10 most common words
top_20_words = word_counts.most_common(20)

# Display the result
top_20_df = pd.DataFrame(top_20_words, columns=['word', 'frequency'])

# Create a bar plot using Plotly
fig_top_20 = px.bar(
    top_20_df,
    x='word',                # Words on the x-axis
    y='frequency',           # Frequencies on the y-axis
    title='Top 20 Words by Frequency',
    labels={'word': 'Word', 'frequency': 'Frequency'},  # Axis labels
    color='frequency',       # Color bars by frequency
)

# Add customization

fig_top_20.update_layout(
    xaxis_tickangle=-45,    # Rotate x-axis labels for better readability
    height=500,
    width=800
)

# Define a function to classify emotions for each tweet
def classify_emotions(text):
    emotion = NRCLex(text)
    raw_emotions = emotion.raw_emotion_scores
    top_emotion = emotion.top_emotions[0][0] if emotion.top_emotions else "None"  # Get only the emotion name
    return raw_emotions, top_emotion

# Apply the function to classify emotions and top emotion for each cleaned tweet
df1['raw_emotions'], df1['label'] = zip(*df1['preprocessed_text'].apply(classify_emotions))

# Expand the `raw_emotions` dictionary into separate columns for each emotion
emotions_df = df1['raw_emotions'].apply(pd.Series).fillna(0)

# Create a new DataFrame with just the original tweet, expanded emotions, and top emotion label
final_df = pd.concat([df1[['clean_text', 'label']], emotions_df], axis=1)

# List of emotion columns
emotion_columns = ['negative', 'sadness', 'anger', 'fear', 'positive', 'trust', 'anticipation', 'joy', 'surprise', 'disgust']

# Aggregate (sum) the emotion scores across all tweets
emotion_sums = final_df[emotion_columns].sum()

# Normalize emotion scores by converting them to percentages
emotion_percentage = (emotion_sums / emotion_sums.sum()) * 100

emotion_percentage_df = pd.DataFrame({'Emotion': emotion_columns, 'percentage': emotion_percentage})
fig = px.bar(
    emotion_percentage_df,
    x = 'Emotion',
    y= 'percentage' ,          
    title='Emotion Distribution Across All Tweets (Percentage)',
    labels={'emotion': 'Emotion', 'percentage': 'Percentage (%)'},  # Axis labels
    
)

# Customize layout
fig.update_layout(
    xaxis_tickangle=-45,         # Rotate x-axis labels for better readability
    height=500,
    width=800,
    title_x=0.5                 # Center the title
)

final_df['states'] = df1['states']

# Group by 'states' and sum each emotion
state_emotions = final_df.groupby('states')[emotion_columns].sum()

# Optional: Normalize the data by state (convert each state's emotions to percentages)
state_emotions_percentage = state_emotions.div(state_emotions.sum(axis=1), axis=0) * 100

# Visualization using a heatmap
import seaborn as sns 

plt.figure(figsize=(12, 8))
plt.title("Emotion Distribution by State (Percentage)")
sns.heatmap(state_emotions_percentage, cmap="YlGnBu", annot=True, fmt=".1f")
plt.xlabel("Emotions")
plt.ylabel("State")
plt.xticks(rotation=45)
plt.tight_layout()
emotion_heatmap_path = 'emotion_heatmap.png'
plt.savefig(emotion_heatmap_path, bbox_inches = 'tight')

state_mean_sentiment = pd.DataFrame({
    'state': state_sentiment['states'],  # Column from india_map for state names
    'sentiment_score': state_sentiment['category']
})

# Merge sentiment data with geographical data
merged = indian_states_map.merge(state_mean_sentiment, left_on=indian_states_map['st_nm'], right_on="state", how="left")

# Plot the heatmap
plt.figure(figsize=(12, 10))
merged.plot(column='sentiment_score', cmap='coolwarm', legend=True, edgecolor='black', linewidth=0.5)

# Add titles and labels
plt.title("Sentiment Analysis Heatmap for India", fontsize=16)
plt.axis('off')
heatmap_img_path = 'sentiment_heatmap.png'
plt.savefig(heatmap_img_path, bbox_inches='tight')


# Sort the counts in descending order to get the top states
top_positive_states = state_positive_counts.sort_values(by='positive_count', ascending=False).head(10)
top_neutral_states = state_neutral_counts.sort_values(by='neutral_count', ascending=False).head(10)
top_negative_states = state_negative_counts.sort_values(by='negative_count', ascending=False).head(10)

with open(wordcloud_img_path, "rb") as img_file:
    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

# Encode heatmap image 
with open(heatmap_img_path, "rb") as img_file:
    encoded_heatmap = base64.b64encode(img_file.read()).decode('utf-8')

# Encode heatmap image
with open(emotion_heatmap_path, "rb") as img_file:
    encoded_emotion_heatmap = base64.b64encode(img_file.read()).decode('utf-8')


# Create Dash app
app = dash.Dash(__name__)

server = app.server

state_options = [{'label': state, 'value': state} for state in df1['states'].unique()]

state_emotions_percentage = state_emotions_percentage.reset_index()
# Melt the DataFrame for easier Plotly handling
state_emotions_melted = state_emotions_percentage.melt(id_vars="states", var_name="Emotion", value_name="Percentage")

app.layout = html.Div([
    html.H1("Sentiment Analysis of Tweets", style={'textAlign': 'center'}),

    # Dropdown to select sentiment category
    dcc.Dropdown(
        id='sentiment-dropdown',
        options=[
            {'label': 'Positive Sentiment', 'value': 'positive'},
            {'label': 'Neutral Sentiment', 'value': 'neutral'},
            {'label': 'Negative Sentiment', 'value': 'negative'}
        ],
        value='positive',  # Default value
        style={'width': '50%'}
    ),
    
    # Create a flex container to place pie chart and bar chart side by side
    html.Div([
        # Bar chart on the left
        html.Div([
            dcc.Graph(id='sentiment-bar-chart', style={'height': '500px'})
        ], style={'width': '55%', 'display': 'inline-block'}),
        # Pie chart on the right
        html.Div([
            dcc.Graph(id='sentiment-pie-chart', figure=fig_pie)
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),

        
    ]),

    html.Div([
        # Top 20 Words Bar Chart
        html.Div([
            dcc.Graph(figure=fig_top_20)
        ], style={'width': '48%', 'padding': '10px'}),  # Half width for the top 20 bar chart

        # WordCloud image
        html.Div([
            html.Img(src=f"data:image/png;base64,{encoded_image}", style={'width': '75%', 'height': 'auto'})
        ], style={'width': '48%', 'padding': '10px'}),  # Half width for the WordCloud image    
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'marginTop': '20px' }),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='state-dropdown',
                options=state_options,
                value=[state_options[0]['value']],  # Default to the first state
                multi=True,  # Allow multiple selections
                placeholder='Select states',
                style={'width': '80%', 'margin': '0 auto'}
            ),
        ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='trend-line-graph')  # Graph will update based on dropdown selection
    ]),

    html.Div([
        # Scrollable table
        html.Div([
            html.H4("State Sentiment Scores"),
            dash_table.DataTable(
                id='state-sentiment-table',
                columns=[
                    {'name': 'State', 'id': 'states'},
                    {'name': 'Sentiment Score', 'id': 'category'}
                ],
                data=state_sentiment.to_dict('records'),
                style_table={'height': '400px', 'overflowY': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Arial'
                },
                style_header={
                    'backgroundColor': 'lightgrey',
                    'fontWeight': 'bold'
                }
            )
        ], style={'width': '45%', 'padding': '10px', 'border': '1px solid lightgrey'}),

        # Heatmap
        html.Div([
            html.H4("Sentiment Heatmap"),
            html.Img(src=f"data:image/png;base64,{encoded_heatmap}", style={'width': '80%', 'height': 'auto'})
        ], style={'width': '50%', 'padding': '10px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'marginTop': '20px'}),

    html.Div([
        html.H4("Emotion Distribution Heatmap"),
        html.Img(src=f"data:image/png;base64,{encoded_emotion_heatmap}", 
                 style={'width': '80%', 'height': 'auto', 'margin': '0 auto'})
    ], style={'marginTop': '20px', 'textAlign': 'center'}),

    html.Div([
        # Emotion distribution by state
        html.Div([
            dcc.Dropdown(
                id="emotion-dropdown",
        options=[
            {"label": state, "value": state}
            for state in state_emotions_percentage["states"].unique()
                ],
        value=state_emotions_percentage["states"].iloc[0],  # Default state
                        ),
    dcc.Graph(id="emotion-bar-chart"),

                ])
            ])
])

# Define callback to update the bar graph based on selected sentiment
@app.callback(
    Output('sentiment-bar-chart', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_bar_chart(selected_sentiment):
    # Select the appropriate data based on the dropdown value
    if selected_sentiment == 'positive':
        data = top_positive_states
        title = 'Top 10 States with Positive Tweets'
        x_axis_label = 'State'
        y_axis_label = 'Positive Tweet Count'
    elif selected_sentiment == 'neutral':
        data = top_neutral_states
        title = 'Top 10 States with Neutral Tweets'
        x_axis_label = 'State'
        y_axis_label = 'Neutral Tweet Count'
    else:
        data = top_negative_states
        title = 'Top 10 States with Negative Tweets'
        x_axis_label = 'State'
        y_axis_label = 'Negative Tweet Count'

    # Create a bar chart using Plotly Express
    fig = px.bar(
        data,
        x='states',  # State names on x-axis
        y=data.columns[1],  # The tweet count column
        title=title,
        labels={data.columns[1]: y_axis_label, 'states': x_axis_label},
        color=data.columns[1],  # Color by count
        color_continuous_scale='Viridis'  # Optional: Customize color scheme
    )

    # Customize layout for better visualization
    fig.update_layout(
        xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
        height=500,
        width=800
    )
    return fig


# Callback to update trend graph based on selected states
@app.callback(
    Output('trend-line-graph', 'figure'),
    Input('state-dropdown', 'value')
)
def update_trend_graph(selected_states):
    filtered_data = monthly_trend_states[monthly_trend_states['states'].isin(selected_states)]

    fig = px.line(
        filtered_data,
        x='month',
        y='category',
        color='states',
        title="Monthly Trend of Sentiment for Selected States",
        labels={'month': 'Month', 'category': 'Average Sentiment', 'states': 'State'}
    )
    fig.update_layout(height=500, width=1200, title_x=0.5)  # Center title and adjust dimensions
    return fig


# Callback to update the bar chart based on selected state
@app.callback(
    Output("emotion-bar-chart", "figure"),
    [Input("emotion-dropdown", "value")]
            )

def update_emotion_bar_chart(selected_state):
    filtered_data = state_emotions_melted[state_emotions_melted["states"] == selected_state]

    # Create a bar chart using Plotly
    fig = px.bar(
        filtered_data,
        x='Emotion',
        y='Percentage',
        color='Emotion',
        title=f"Emotion Percentage for {selected_state}",
        labels={'Emotion': 'Emotion Type', 'Percentage': 'Percentage'},
        height=500
                )

    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Percentage (%)",
        template="plotly_white"
                    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
