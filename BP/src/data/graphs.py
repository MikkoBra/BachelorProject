import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

from data import data_processing

matplotlib.use('TkAgg')


# Plots the distribution of emotion classifications in the dataset in percentages
def class_percent_graph(labels, oversampled=False):
    if oversampled:
        # Load the clean data after oversampling
        clean_data, clean_labels = data_processing.load_dataset('datasets/train_data_clean.pkl')
        clean_labels = data_processing.one_hot_to_int(clean_labels)
        # Retrieve count of each emotion, convert to percentage
        emotion_count = pd.Series(clean_labels).value_counts(normalize=True) * 100
    else:
        # Retrieve count of each emotion, convert to percentage
        emotion_count = data_processing.read_train_data()['emotion'].value_counts(normalize=True) * 100

    # Sort in ascending order such that highest is plotted higher on y-axis
    emotion_sorted = emotion_count.sort_values(ascending=True)

    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, emotion_sorted.values.tolist(), height=0.5, color=(0.2, 0.4, 0.6, 0.6))
    ax.bar_label(bars)
    ax.set_ylabel('Emotion', fontsize=12)
    ax.set_xlabel('Percentage of Occurrence', fontsize=12)
    ax.set_title('Emotion Distribution', fontsize=14)
    ax.set_xlim([0, 50])
    plt.show()


# Create a word cloud out of all social media texts per emotion classification
def word_cloud():
    # List the 7 emotions
    df = data_processing.read_train_data()
    emotions = df['emotion'].unique()
    print(emotions)

    for emotion in emotions:
        # Reduce dataframe to texts from one emotion
        essays_by_emotion = df.loc[df['emotion'] == emotion, 'essay']
        # Join all texts
        text = " ".join(essay for essay in essays_by_emotion)
        # Define words to filter out (based on visual analysis)
        stopwords = set(STOPWORDS)
        # Create wordcloud
        wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white", colormap="viridis") \
            .generate(text)
        # Display wordcloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
