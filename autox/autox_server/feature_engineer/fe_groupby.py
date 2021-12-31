import warnings
warnings.filterwarnings('ignore')
import time
from autox.autox_server.util import log

# test拼上train重新groupby

def fe_groupby(G_df_dict, G_data_info, G_hist, is_train, remain_time):
    # 对G_df_dict['BIG']表做扩展特征

    top_imp_k = 20

    start = time.time()
    log('[+] feature engineer, groupby')

    Id = G_data_info['target_id']
    target = G_data_info['target_label']

    data_type_ = G_data_info['entities'][G_data_info['target_entity']]['columns']
    big_cols = [list(x.keys())[0] for x in data_type_]

    imp_features = []
    if 'base_lgb' in G_hist.keys():
        if 'feature_importances' in G_hist['base_lgb'].keys():
            imp_features = list(G_hist['base_lgb']['feature_importances'].head(top_imp_k)['feature'])

    big_data_type = {}
    for item in data_type_:
        key_ = list(item.keys())[0]
        value_ = list(item.values())[0]
        big_data_type[key_] = value_

    if is_train:
        G_hist['FE_groupby'] = {}

        # 获得类别型和数值型的变量
        cat_cols = []
        num_cols = []

        data_size = G_df_dict['BIG'].shape[0]
        for col in G_df_dict['BIG'].columns:
            # if col in [target] or ('__max' in col) or ('__min' in col) or ('__median' in col) or \
            #         ('__mean' in col) or ('__std' in col) or ('__nunique' in col):
            #     continue
            if col not in big_cols:
                continue
            if col not in imp_features:
                continue
            if big_data_type[col] == 'Str':
                if G_df_dict['BIG'][col].nunique() >= data_size * 0.2 or G_df_dict['BIG'][col].nunique() <= 100:
                    continue
                cat_cols.append(col)
            elif big_data_type[col] == 'Num':
                num_cols.append(col)

        G_hist['FE_groupby']['cat_cols'] = cat_cols
        G_hist['FE_groupby']['num_cols'] = num_cols
        log("cat_cols:{}".format(cat_cols))
        log("num_cols:{}".format(num_cols))

    group_res = G_df_dict['BIG'].copy()
    for groupby_key in G_hist['FE_groupby']['cat_cols']:

        for cur_agg_num in G_hist['FE_groupby']['num_cols']:
            if cur_agg_num == groupby_key:
                continue
            log('groupby_key: {}, cur_agg_num: {}, num'.format(groupby_key, cur_agg_num))
            cur_temp = group_res.groupby(groupby_key).agg({cur_agg_num: ['max', 'min', 'median', 'mean', 'std']})
            cur_temp = cur_temp.reset_index()
            cur_temp.columns = [groupby_key] + ['groupby_' + groupby_key + "__" + cur_agg_num + "__" + x for x in
                                                ['max', 'min', 'median', 'mean', 'std']]
            group_res = group_res.merge(cur_temp, on=groupby_key, how='left')

        for cur_agg_cat in G_hist['FE_groupby']['cat_cols']:
            if cur_agg_cat == groupby_key:
                continue
            log('groupby_key: {}, cur_agg_cat: {}, cat'.format(groupby_key, cur_agg_cat))
            cur_temp = group_res.groupby(groupby_key).agg({cur_agg_cat: ['nunique']})
            cur_temp = cur_temp.reset_index()
            cur_temp.columns = [groupby_key] + ['groupby_' + groupby_key + "__" + cur_agg_cat + "__" + x for x in
                                                ['nunique']]
            group_res = group_res.merge(cur_temp, on=groupby_key, how='left')

    groupby_features = [x for x in group_res.columns if x not in G_df_dict['BIG'].columns]
    group_res = group_res[groupby_features + Id]

    G_df_dict['FE_groupby'] = group_res

    end = time.time()
    remain_time -= (end - start)
    log("remain_time: {} s".format(remain_time))
    return remain_time