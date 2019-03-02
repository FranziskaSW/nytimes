import pickle
import pandas as pd
import numpy as np
from itertools import chain
from itertools import combinations
import os
import math

global cwd
cwd = os.getcwd()


def keyword_edges(field):
    """
    creates list of edges between keywords
    :param field: list of keyword ids
    :return: list of edges
    """
    field.sort()
    edges = []
    for subset in combinations(field, 2):
        edge = str(subset[0]) + ',' + str(subset[1])
        edges.append(edge)
    return edges


def edge_weight(edges_row, table_keywords):
    """
    calculates weight of edge based on conditional probability.
    In the beginning pro(edges, table_keywords) babilities in log-scale, for weight translated to normal scale
    :param edges_row: one row from edges table (one edge with information)
    :param table_keywords: keywords DataFrame
    :return: weight of edge
    """
    p1 = (edges_row.prob - table_keywords[table_keywords.id == edges_row.Target].prob).get_values()[0]
    p2 = (edges_row.prob - table_keywords[table_keywords.id == edges_row.Source].prob).get_values()[0]
    p1, p2 = np.exp(p1), np.exp(p2)
    p = (p1 + p2)*100
    return p


def edges_nodes(article_keywords, table_keywords, article_amount):
    """
    creates edges and nodes of the article keyword
    :param article_keywords: keywords of articles, every article has a list of keywords
    :param table_keywords: keywords DataFrame
    :param article_amount: amount of articles (same as rows in article_keywords)
    :return: edges DataFrame, nodes DataFrame
    """
    edges_list = article_keywords.apply(lambda x: keyword_edges(x)).tolist()  # each article has a list of keywords
    edges_df = pd.Series(list(chain.from_iterable(edges_list)))  # write everything in one list
    edges_counts = edges_df.value_counts()

    edges = pd.DataFrame([x.split(',') for x in edges_counts.index], columns=['keyword_1', 'keyword_2'])
    edges['Source'] = edges.keyword_1.apply(lambda x: int(x))
    edges['Target'] = edges.keyword_2.apply(lambda x: int(x))
    edges['Counts'] = edges_counts.reset_index()[0]

    e = edges[['Source', 'Target', 'Counts']]

    # only keep edges where both Source and Target are in table_keywords
    e_red = e[e.Source.isin(table_keywords.id) & e.Target.isin(table_keywords.id)]

    e_red['prob'] = np.log(e_red.Counts/article_amount)
    e_red['Weight'] = e_red.apply(lambda x: edge_weight(x, table_keywords), axis=1)

    t = table_keywords[['id', 'section', 'value']]
    ids_1 = e_red.Source.value_counts().index.get_values().tolist()  # unique ids in Source
    ids_2 = e_red.Target.value_counts().index.get_values().tolist()  # unique ids in Target
    mask = [any(y) for y in zip(t.id.isin(ids_1), t.id.isin(ids_2))]  # if id was either in Source or in Target or both
    n = t[mask]
    n.columns = ['id', 'Section', 'Label']

    return e_red, n


def translate_id(table_keywords, edges, nodes):
    """
    resets the keyword id in table_keywords to index of this table, in case some of the rows were deleted (ids would be missing)
    renames ids in edges and nodes accordingly
    :param table_keywords: keywords DataFrame
    :param edges: edges DataFrame
    :param nodes: nodes DataFrame
    :return:
    """
    tr = pd.DataFrame(table_keywords.id)
    tr['id_new'] = tr.index

    edges_Source = pd.merge(edges, tr, left_on='Source', right_on='id')
    edges_Source.columns = ['Source_old', 'Target', 'Counts', 'prob', 'Weight', 'id', 'Source']
    edges_Target = pd.merge(edges_Source, tr, left_on='Target', right_on='id')
    edges_Target.columns = ['Source_old', 'Target_old', 'Counts', 'prob', 'Weight', 'id_x', 'Source', 'id_y', 'Target']
    edges = edges_Target[['Source', 'Target', 'Counts', 'prob', 'Weight']]

    nodes_new = pd.merge(nodes, tr, left_on='id', right_on='id')
    nodes_new.columns = ['id_old', 'Section', 'Label', 'id']
    nodes = nodes_new[['id', 'Section', 'Label']]

    table_keywords.id = tr.id_new
    return table_keywords, edges, nodes


def reduce_edges(edges, nodes, percentage, table_keywords, min_edges):
    """
    reduces the edges according to following:
    - only keep edges that are in top mutual 'percentage'% edges of their nodes
    - only keep nodes that have at least min_edges edges
    :param edges: edges DataFrame
    :param nodes: nodes DataFrame
    :param percentage: cutoff precentage
    :param table_keywords: keywords DataFrame
    :param min_edges: minimum amount of edges per node, included
    :return: lower dimensional edges DataFrame, lower dimensional nodes DataFrame
    """
    # find top x% of edges to each node
    # matrix of edges, nodes*nodes
    mat = np.zeros([nodes.id.max()+1, nodes.id.max()+1])
    for keyword_id in nodes.id:
        # the other keywords that keyword_id is connected to
        connected_t = edges[edges.Source == keyword_id][['Target', 'Counts']]
        connected_t.columns = ['Node', 'Counts']
        connected_s = edges[edges.Target == keyword_id] [['Source', 'Counts']]
        connected_s.columns = ['Node', 'Counts']

        total_connections = (connected_s).append(connected_t)
        idf = table_keywords.idf[table_keywords.id == keyword_id]
        max_edges = math.ceil(total_connections.shape[0]*percentage*idf)
        tc = total_connections.sort_values(by='Counts', ascending=False)
        tc = tc[:max_edges]

        # entry = 1 if edge is in top x% of row-node
        mat[keyword_id, tc.Node.tolist()] = 1

    # only keep the edges that are in top x% of row-node AND column-node
    keep_edges = dict()
    for keyword_id in nodes.id:
        keyword_has = mat[keyword_id, :]
        keyword_appears_in = mat[:, keyword_id]

        l1 = pd.Series(keyword_has).nonzero()[0].tolist()
        l2 = pd.Series(keyword_appears_in).nonzero()[0].tolist()
        intersection = set(l1) - (set(l1) - set(l2))
        dict_update = {keyword_id: intersection}
        keep_edges.update(dict_update)

    mask = []
    for idx in range(0, edges.shape[0]):
        mask.append(edges.Target[idx] in keep_edges[edges.loc[idx].Source])

    edges_reduced = edges[mask]

    # remove the nodes that are not left after the x% filtering
    s = edges_reduced.Source.value_counts()
    t = edges_reduced.Target.value_counts()
    st = pd.merge(pd.DataFrame(s), pd.DataFrame(t), left_index=True, right_index=True, how='outer').fillna(0)

    mask = st.index.tolist()

    nodes.index = nodes.id.tolist()
    nodes_reduced = nodes.loc[mask]

    # delete nodes that only have one edge
    s = edges_reduced.Source.value_counts()
    t = edges_reduced.Target.value_counts()

    st = pd.merge(pd.DataFrame(s), pd.DataFrame(t), left_index=True, right_index=True, how='outer').fillna(0)
    st['counts'] = st.Source + st.Target

    # drop nodes that don't have enough edges
    mask = (st.counts > min_edges)
    idx = st[mask].index.get_values().tolist()
    nodes_reduced = nodes_reduced[nodes_reduced.id.isin(idx)]

    # drop edges where we had one of those nodes
    mask = [all(tup) for tup in zip(edges_reduced.Source.isin(idx), edges_reduced.Target.isin(idx))]

    edges_reduced = edges_reduced[mask]

    return edges_reduced, nodes_reduced


def main():

    with open(cwd + "/data/table_keywords_16-18.pickle", "rb") as f:
        table_keywords = pickle.load(f)

    with open(cwd + "/data/df_16-18.pickle", "rb") as f:
        df = pickle.load(f)

    article_amount = df.shape[0]
    keywords = df.keywords

    # only use keywords that will be relevant for us later,
    # because will sort out less frequent ones in reduce_edges anyways
    min_edges = 2
    percentage = 0.35
    table_keywords = table_keywords[table_keywords.counts >= min_edges / percentage]
    table_keywords.index = range(0, table_keywords.shape[0])


    edges, nodes = edges_nodes(keywords, table_keywords, article_amount)
    print(edges.shape, nodes.shape)

    with open(cwd + "/data/edges_16-18.pickle", "wb") as f:
        pickle.dump(edges, f)
    with open(cwd + "/data/nodes_16-18.pickle", "wb") as f:
        pickle.dump(nodes, f)


    # reduce_edges
    table_keywords, edges, nodes = translate_id(table_keywords, edges, nodes)

    edges_reduced, nodes_reduced = reduce_edges(edges, nodes, percentage, table_keywords, min_edges)
    print(edges_reduced.shape, nodes_reduced.shape)

    series = '02'
    name = 'idf-mutual_16-18_3'
    nodes_reduced.to_csv(cwd + '/data/gephi/' + series + 'nodes_' + name + '.csv', sep=';', index=False)
    edges_reduced.to_csv(cwd + '/data/gephi/' + series + 'edges_' + name + '.csv', sep=';', index=False)


if __name__ == '__main__':
    main()
