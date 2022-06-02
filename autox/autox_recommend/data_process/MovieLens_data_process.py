import os
import datetime
import pandas as pd


def inter_df(interactions):
    """
    Create a dataframe with the interaction information
    Considering valid interactions only rating of 4 or higher
    :param interactions:
    :return:
    """
    interactions = interactions.loc[interactions['rating'] >= 4]
    interactions.drop('rating', axis=1, inplace=True)
    interactions.loc[:, 'time'] = interactions.loc[:, 'timestamp'].apply(
        lambda ts: datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    interactions['time'] = pd.to_datetime(interactions['time'])
    interactions.drop('timestamp', axis=1, inplace=True)
    return interactions


def item_df(movies, genome_scores):
    """
    Create a dataframe with the item information
    genome_scores提供了每部电影和不同tag的相关性
    将每个tag作为一列，共有tags_count列
    :param movies:
    :param genome_scores:
    :return:
    """
    movie_tags_r = genome_scores.pivot(
        index='movieId',
        columns='tagId',
        values='relevance')
    movie_tags_r = movie_tags_r.reset_index()
    tags_count = movie_tags_r.shape[1]
    movie_tags_r.columns = ['movieId'] + \
        ['tag_' + str(i) for i in range(1, tags_count)]
    movies = movies.merge(movie_tags_r, on='movieId', how='left')
    movies.rename(columns={'movieId':'itemId'}, inplace=True)
    return movies


def user_df(interactions):
    """
    Create a dataframe with the user information
    :param interactions:
    :return:
    """
    user_ids = interactions['userId'].unique()
    users = pd.DataFrame(user_ids, columns=['userId'])
    return users


def data_split(interactions, days=7):
    """
    Split the data into train and test sets
    :param interactions:
    :return:
    """
    split_time = interactions['time'].max() - datetime.timedelta(days=days)
    train = interactions.loc[interactions['time'] < split_time]
    test = interactions.loc[interactions['time'] >= split_time]
    return train, test


if __name__ == '__main__':
    path = '../datasets/ml-25m/'

    print('Loading data...')
    # tags 为用户为电影加的标签，暂不使用
    # tags = pd.read_csv(os.path.join(path, 'tags.csv'))
    # links 为电影在imdb中的id，暂不使用；可作为item信息的增加
    # links = pd.read_csv(os.path.join(path, 'links.csv'))
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
    genome_tags = pd.read_csv(os.path.join(path, 'genome-tags.csv'))
    genome_scores = pd.read_csv(os.path.join(path, 'genome-scores.csv'))
    movies = pd.read_csv(os.path.join(path, 'movies.csv'))

    print('Formatting data...')
    interactions = inter_df(ratings)
    items = item_df(movies, genome_scores)
    users = user_df(interactions)

    print('Splitting data...')
    train, test = data_split(interactions)

    print('Saving data...')
    output_path = os.path.join(path, 'data_formatted')
    os.makedirs(output_path, exist_ok=True)
    train.to_csv(
        os.path.join(
            output_path,
            'interactions_train.csv'),
        index=False)
    test.to_csv(
        os.path.join(
            output_path,
            'interactions_test.csv'),
        index=False)
    items.to_csv(os.path.join(output_path, 'items.csv'), index=False)
    users.to_csv(os.path.join(output_path, 'users.csv'), index=False)

    print('Done!')
