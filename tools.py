import spacy
from collections import Counter, defaultdict
import time
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud


def sample_data(data, sample=True, sample_size=1000):
    """
    Conditionally samples the provided DataFrame using random selection.

    Parameters:
        data (pd.DataFrame): The DataFrame to potentially sample.
        sample (bool): A flag to determine whether to sample the DataFrame.
        sample_size (int): The number of rows to sample if sampling is enabled.

    Returns:
        pd.DataFrame: The original or randomly sampled DataFrame based on the 'sample' flag.
    """
    if sample:
        # Return a random sample of the specified size from the DataFrame
        # If sample_size is greater than the number of rows in the DataFrame, it returns the DataFrame unaltered.
        return data.sample(n=sample_size, replace=False, random_state=42)
    else:
        # Return the original DataFrame if sampling is not required
        return data

def identify_adjectives(data):
    
    # Load English model of spaCy
    nlp = spacy.load("en_core_web_sm")

    negative_adjectives = defaultdict(int)
    positive_adjectives = defaultdict(int)

    # Custom corrections for known misclassified adjectives
    corrections = {'sure': None}  # We remove "sure" because it was the most common adjective somehow?

    for _, row in data.iterrows():
        doc = nlp(row["Text_Review"])
        for token in doc:
            if token.pos_ == 'ADJ':  # Check if the token is an adjective
                corrected_text = corrections.get(token.lemma_, token.lemma_)
                if corrected_text is None:
                    continue  # Skip this token if it's in the corrections list to be ignored

                # Determine the presence of a negative dependency
                is_negative = any(child.dep_ == 'neg' for child in token.children)
                key = (row["Product_ID"], row["Type"], corrected_text)
                
                if is_negative:
                    negative_adjectives[key] += 1
                else:
                    positive_adjectives[key] += 1

    # Convert the defaultdicts to DataFrames
    neg_df = pd.DataFrame(list(negative_adjectives.items()), columns=['Key', 'Count'])
    neg_df[['Product_ID', 'Type', 'Adjective']] = pd.DataFrame(neg_df['Key'].tolist(), index=neg_df.index)
    neg_df.drop('Key', axis=1, inplace=True)

    pos_df = pd.DataFrame(list(positive_adjectives.items()), columns=['Key', 'Count'])
    pos_df[['Product_ID', 'Type', 'Adjective']] = pd.DataFrame(pos_df['Key'].tolist(), index=pos_df.index)
    pos_df.drop('Key', axis=1, inplace=True)

    return neg_df, pos_df

def plot_adjective_counts(negative_adjectives_df, positive_adjectives_df):
    # Grouping the data by type and aggregating the sum of adjectives for negative and positive separately
    negative_grouped_data = negative_adjectives_df.groupby('Type')['Count'].sum().reset_index()
    positive_grouped_data = positive_adjectives_df.groupby('Type')['Count'].sum().reset_index()

    # Create a figure and a single subplot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar plot for total positive adjectives in blue
    ax1.bar(positive_grouped_data['Type'], positive_grouped_data['Count'], color='b', label='Positive', width=0.4, align='center')

    # Setting the labels and title for positive adjectives
    ax1.set_xlabel('Type', fontsize=12)
    ax1.set_ylabel('Total Positive Adjectives', color='blue', fontsize=12)
    ax1.set_title('Total Positive and Negative Adjectives by Type', fontsize=14)

    # Adding a legend for positive adjectives
    ax1.legend(loc='upper left')

    # Create a second y-axis for negative adjectives
    ax2 = ax1.twinx()
    
    # Bar plot for total negative adjectives in red
    ax2.bar(negative_grouped_data['Type'], negative_grouped_data['Count'], color='r', label='Negative', width=0.4, align='edge')

    # Setting the label for negative adjectives
    ax2.set_ylabel('Total Negative Adjectives', color='red', fontsize=12)

    # Adding a legend for negative adjectives
    ax2.legend(loc='upper right')

    # Adjust layout
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    # Get sentiment scores using VADER
    sentiment = analyzer.polarity_scores(text)
    # Return only the compound score, which represents overall sentiment
    return sentiment['compound']

def generate_wordcloud(adjectives_df, positive=True, product_type=None):
    # Filter the dataframe based on the product_type if provided
    if product_type:
        adjectives_df = adjectives_df[adjectives_df['Type'] == product_type]

    # Determine the type of adjectives based on the 'positive' parameter
    adjective_type = "Positive" if positive else "Negative"

    # Combine all adjectives into a single string
    all_adjectives = ' '.join(adjectives_df['Adjective'])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_adjectives)

    # Display the word cloud with the appropriate title
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of {adjective_type} Adjectives')
    if product_type:
        plt.title(f'Word Cloud of {adjective_type} Adjectives for {product_type}')
    plt.show()

