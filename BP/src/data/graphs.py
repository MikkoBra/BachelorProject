import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from data import data_reader
matplotlib.use('TkAgg')


# Plots the distribution of emotion classifications in the dataset in percentages
def class_percent_graph():
    # Retrieve count of each emotion, convert to percentage
    emotion_count = data_reader.read_train_data()['emotion'].value_counts(normalize=True) * 100
    # Sort in ascending order such that highest is plotted higher on y-axis
    emotion_sorted = emotion_count.sort_values(ascending=True)

    # Generate plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(emotion_sorted.index, emotion_sorted.values, height=0.5, color=(0.2, 0.4, 0.6, 0.6))
    ax.bar_label(bars)
    ax.set_ylabel('Emotion', fontsize=12)
    ax.set_xlabel('Percentage of Occurrence', fontsize=12)
    ax.set_title('Emotion Distribution', fontsize=14)
    ax.set_xlim([0, 50])
    plt.show()


def word_cloud():
    df = data_reader.read_train_data()
    emotions = df['emotion'].unique()
    print(emotions)
    for emotion in emotions:
        essays_by_emotion = df.loc[df['emotion'] == emotion, 'essay']
        text = " ".join(essay for essay in essays_by_emotion)
        stopwords = set(STOPWORDS)
        stopwords.update(["people", "really", "article", "think", "will", "make", "need", "know", "one", "thing"])
        wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white", colormap="viridis").generate(text)
        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
