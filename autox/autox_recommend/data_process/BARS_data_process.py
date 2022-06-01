import os
import datetime
import pandas as pd


def inter_df(interactions):
    """
    Create a dataframe with the interaction information
    :param interactions:
    :return:
    """
    return interactions


def item_df(items):
    """
    Create a dataframe with the item information
    :param movies:
    :return:
    """
    items = items['remap_id'].to_frame()
    # rename the columns
    items.columns = ['itemId']
    return items


def user_df(users):
    """
    Create a dataframe with the user information
    :param users:
    :return:
    """
    users = users['remap_id'].to_frame()
    # rename the columns
    users.columns = ['userId']
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


def load_bars_data(file_path, nrows=None):
    """
    Load the BARS data
    :param file_path:
    :param nrows:
    :return:
    """
    data = []
    row_count = 0
    with open(file_path) as f:
        for l in f.readlines():
            if nrows is not None and row_count >= nrows:
                break
            row_count += 1
            if len(l) > 0:
                l = l.strip().split(' ')
                uid = int(l[0])
                pairs = [(uid, int(i)) for i in l[1:]]
                data.extend(pairs)
    return pd.DataFrame(data, columns=['userId', 'itemId'])


if __name__ == '__main__':
    path = '../datasets/BARS'
    data_lists = ['amazon-book', 'gowalla', 'yelp2018']

    for data in data_lists:
        print('* Processing {}...'.format(data))
        print('Loading data...')
        interactions_train = load_bars_data(
            os.path.join(path, data, 'train.txt'))
        interactions_test = load_bars_data(
            os.path.join(path, data, 'test.txt'))
        users = pd.read_csv(
            os.path.join(
                path,
                data,
                'user_list.txt'),
            delimiter=' ')
        items = pd.read_csv(
            os.path.join(
                path,
                data,
                'item_list.txt'),
            delimiter=' ')

        print('Formatting data...')
        # interactions_train = inter_df(interactions_train)
        # interactions_test = inter_df(interactions_test)
        items = item_df(items)
        users = user_df(users)

        print('Saving data...')
        output_path = os.path.join(path, data, 'data_formatted')
        os.makedirs(output_path, exist_ok=True)
        interactions_train.to_csv(
            os.path.join(
                output_path,
                'interactions_train.csv'),
            index=False)
        interactions_test.to_csv(
            os.path.join(
                output_path,
                'interactions_test.csv'),
            index=False)
        items.to_csv(os.path.join(output_path, 'items.csv'), index=False)
        users.to_csv(os.path.join(output_path, 'users.csv'), index=False)

        print('Done!')
