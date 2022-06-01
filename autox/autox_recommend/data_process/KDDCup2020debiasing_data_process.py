import os
import datetime
import pandas as pd


def time_transform(time):
    random_number_1 = 41152582
    random_number_2 = 1570909091
    return datetime.datetime.fromtimestamp(
        time * random_number_2 + random_number_1)


def inter_df(interactions):
    """
    Create a dataframe with the interaction information
    :param interactions:
    :return:
    """
    interactions['time'] = interactions['time'].apply(time_transform)
    return interactions


def item_df(items):
    """
    Create a dataframe with the item information
    :param movies:
    :return:
    """
    dim = 128
    items['txtVec'] = pd.Series(
        items.iloc[:, 1:dim + 1].fillna('').values.tolist()).str.join('')
    items['imgVec'] = pd.Series(
        items.iloc[:, dim + 1:].fillna('').values.tolist()).str.join('')
    # drop the columns between 1 and dim*2
    items.drop(items.columns[1:dim + dim + 1], axis=1, inplace=True)
    items.columns = ['itemId', 'txtVec', 'imgVec']
    return items


def user_df(users):
    """
    Create a dataframe with the user information
    :param users:
    :return:
    """
    # rename the columns
    users.columns = ['userId', 'userAgeLevel', 'userGender', 'userCityLevel']
    return users


def data_split(interactions_a, interactions_b, days=7):
    """
    Split the data into train and test sets
    :param interactions:
    :return:
    """
    interactions = pd.concat([interactions_a, interactions_b])
    split_time = interactions['time'].max() - datetime.timedelta(days=days)
    train = interactions.loc[interactions['time'] < split_time]
    test = interactions.loc[interactions['time'] >= split_time]
    return train, test


def load_kdd_interactions(file_path, type):
    interactions = pd.DataFrame()
    if type == 'train':
        filename = "underexpose_train_click-{i}.csv"
    elif type == 'test':
        filename = "underexpose_test_click-{i}/underexpose_test_click-{i}.csv"
    for i in range(0, 10):
        cur_file = os.path.join(file_path, filename.format(i=i))
        tmp = pd.read_csv(
            cur_file, header=None, names=[
                'userId', 'itemId', 'time'])
        interactions = pd.concat([interactions, tmp])
    return interactions


if __name__ == '__main__':
    path = '../datasets/kdd2020-debiasing/'

    print('Loading data...')
    underexpose_train = load_kdd_interactions(
        os.path.join(path, 'underexpose_train'), type='train')
    underexpose_test = load_kdd_interactions(
        os.path.join(path, 'underexpose_test'), type='test')
    items = pd.read_csv(
        os.path.join(
            path,
            'underexpose_train/underexpose_item_feat.csv'),
        header=None,
        dtype=str)
    users = pd.read_csv(
        os.path.join(
            path,
            'underexpose_train/underexpose_user_feat.csv'),
        header=None,
        dtype=str)

    print('Formatting data...')
    underexpose_train = inter_df(underexpose_train)
    underexpose_test = inter_df(underexpose_test)
    items = item_df(items)
    users = user_df(users)

    print('Splitting data...')
    # KDD Cup 2020 debiasing数据集是淘宝的两周内点击数据
    # 目前划分时选择最后3天为测试集，test占总数据集的比例约为0.16
    interactions_train, interactions_test = data_split(
        underexpose_train, underexpose_test, days=3)

    print('Saving data...')
    output_path = os.path.join(path, 'data_formatted')
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
