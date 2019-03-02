import json
import requests
import pickle
import pandas as pd
import numpy as np
import os

global cwd
cwd = os.getcwd()

def get_data(year, month):
    """
    pulls the meta data of the articles that were published during that month and saves it in archive,
    uses nytimes search api
    :param year: str
    :param month: str
    """
    archive_key = 'Jctp3rj1ZdOaLQiMArs79ioGnwvfK1pC'
    month_api = year + '/' + month
    if len(month) == 1:
        month = '0' + month
    data_suffix = year + '_' + month
    url = 'https://api.nytimes.com/svc/archive/v1/' + month_api + '.json?api-key=' + archive_key

    print('-------------- load', url, ' --------------')
    html = requests.get(url)  # load page
    a = html.text
    api_return = json.loads(a)
    articles = api_return['response']['docs']
    # articles = response['docs']
    df = pd.DataFrame(articles)
    with open(cwd + "/data/archive/articles_" + data_suffix + ".pickle", "wb") as f:
        pickle.dump(df, f)



def getSectionDict(name):
    """
    groups section_name into 12 meta-sections
    :param name: section_name in from search api
    :return: name of meta-section
    """
    world = ['World', 'Africa', 'Americas', 'Asia', 'Asia Pacific', 'Australia', 'Canada', 'Europe', 'Middle East',
             'What in the World', 'Opinion | The World', 'Foreign']
    if name in world: return 'World'
    us = ['U.S.', 'National']
    if name in us: return 'U.S.'
    politics = ['Elections', 'Politics', 'Tracking Trumps Agenda', 'The Upshot', 'Opinion | Politics', 'Upshot',
                'Washington ']
    if name in politics: return 'Politics'
    ny = ['N.Y. / Region', 'New York Today', 'Metro', 'Metropolitan']
    if name in ny: return 'New York'
    business_technology = ['Business Day', 'Economy', 'Media', 'Money', 'DealBook', 'Markets', 'Energy', 'IPhone App',
                           'Media', 'Technology', 'Personal Tech', 'Entrepreneurship', 'Your Money', 'Business',
                           'SundayBusiness']
    if name in business_technology: return 'Business & Technology'
    sports = ['Skiing', 'Rugby', 'Sailing', 'Cycling', 'Cricket', 'Auto Racing', 'Horse Racing', 'World Cup',
              'Olympics', 'Pro Football', 'Pro Basketball', 'Sports', 'Baseball', 'NFL', 'College Football', 'NBA',
              'College Basketball', 'Hockey', 'Soccer', 'Golf', 'Tennis']
    if name in sports: return 'Sports'
    arts = ['Opinion | Culture', 'Arts', 'Art & Design', 'Books', 'Book Review', 'BookReview', 'Best Sellers',
            'By the Book', 'Crime', 'Children\'s Books', 'Book Review Podcast', 'Now read this', 'Dance', 'Movies',
            'Music', 'Television', 'Theater', 'Pop Culture', 'Watching', 'Culture', 'Arts&Leisure']
    if name in arts: return 'Culture'
    style = ['Men\'s Style', 'Style', 'Styles', 'TStyle', 'Fashion & Style', 'Fashion & Beauty', 'Fashion', 'Weddings',
             'Self-Care']
    if name in style: return 'Style'
    science = ['Energy & Environment', 'Science', 'Climate', 'Opinion | Environment', 'Space & Cosmos', 'Trilobites',
               'Sciencetake', 'Out There']
    health = ['Mind', 'Health Guide', 'Health', 'Health Policy', 'Live', 'Global Health', 'The New Old Age', 'Science',
              'Well', 'Move']
    sci_hel = science + health + ['Family', 'Live']
    if name in sci_hel: return 'Health & Science'
    food = ['Eat', 'Wine, Beer & Cocktails', 'Restaurant Reviews', 'Dining', 'Food']
    travel = ['36 Hours', 'Frugal Traveler', '52 Places to go', 'Travel']
    magazine = ['Smarter Living', 'Wirecutter', 'Automobiles', 'T Magazine', 'Magazine', 'Design & Interiors',
                'Entertainment', 'Video', 'Weekend']
    leisure = food + travel + magazine
    if name in leisure: return 'Leisure'
    opinion = ['Opinion', 'Letters', 'Contributors', 'Editorials', 'Columnists', 'OpEd', 'Sunday Review', 'Games',
               'Editorial']
    realestate = ['Real Estate', 'RealEstate', 'Commercial Real Estate', 'The High End', 'Commercial', 'Find a Home',
                  'Mortgage Calculator', 'Your Real Estate', 'List a Home']
    education = ['Education', 'Education Life', 'The Learning Network', 'Lesson Plans', 'Learning']
    delete = (['Blogs', 'Insider Events', 'Retirement', 'América', 'Multimedia/Photos', 'The Daily',
               'Briefing', 'Sunday Review', 'Crosswords & Games', 'Times Insider', 'Corrections', 'NYTNow',
               'Corrections', 'Podcasts', 'Insider', 'Obits', 'Summary']
              + opinion + education + realestate)
    if name in delete: return '*DELETE*'
    else: return '*UNKNOWN*'


def extr_headline_main(field):
    """
    extracts main headline from api entry
    :param field: api entry structure
    :return: headline
    """
    return field['main']


def clean_articles(df, word_count):
    """
    clean the articles, only keep articles with
    - more than 20 words
    - that are certain 'type_of_material'
    - drop duplicate articles, if same headline appears in same section
    :param df: DataFrame of articles
    :param word_count: minimum amount of words, not included
    :return: cleaned DataFrame of articles
    """
    df = df[~(df.word_count.isnull())]
    df['word_count'] = df.word_count.apply(lambda x: int(x))
    df = df[df.word_count > word_count]
    df['headline'] = df.headline.apply(lambda x: extr_headline_main(x))
    df = df.drop_duplicates(['headline', 'section_name'])
    mask = ['News', 'Brief', 'briefing']
    df = df[df.type_of_material.isin(mask)]
    return df

def clean_sections(df):
    """
    uses getSectionDict to rename sections to their meta-section
    :param df: DataFrame of articles
    :return: DataFrame of articles, section renamed
    """
    df['section'] = df.section_name.apply(lambda x: getSectionDict(x))
    without_section = df[df.section == '*UNKNOWN*']  # the articles that haven't had a section_name,
                                                     # many of them have news_desk entry
    sections_from_newsdesk = without_section.news_desk.apply(lambda x: getSectionDict(x))
    idx = sections_from_newsdesk.index.get_values()
    df.loc[idx, 'section'] = sections_from_newsdesk
    return df

################################################################################# keyword stuff


def extr_keywords_step1(field):
    """
    brings entry as it comes from api in more handy format
    :param field: 'keywords' entry of api
    :return: tupel (name, value)
    """
    keyword = field
    keyword_tup = (keyword['name'], keyword['value'])
    return keyword_tup


def create_keyword_table_partial(df):
    """
    uses article DataFrame to create table of keywords. How often keyword appeared in which section
    :param df: articles DataFrame
    :return: DataFrame of keywords (keyword, section, counts)
    """
    dfs = df[['_id', 'section', 'pub_date', 'headline', 'keywords']]
    # expand columns from keyword_dict
    d1 = dfs.keywords.apply(pd.Series).merge(dfs, left_index=True, right_index=True).drop(["keywords"], axis=1)
    # columns are additional rows
    d2 = d1.melt(id_vars=['_id', 'section', 'pub_date', 'headline'], value_name="keyword").drop("variable", axis=1)

    mask = d2.keyword.isna()
    d3 = d2[~mask]

    d3 = d3.sort_values(by=['pub_date', '_id'])

    d3['keyword'] = d3.keyword.apply(lambda x: extr_keywords_step1(x))

    keyword_table = d3[['keyword', 'section', '_id']]
    table = keyword_table.groupby(by=['keyword', 'section']).count()
    table = table.reset_index()
    table.columns = ['keyword', 'section', 'counts']
    return table


def create_keyword_table(table, threshold, article_amount):
    """
    table: table of keywords where one keyword can have multiply rows, if it appeared in different sections
    function reduces this table to keyword_table, where each keyword only appears once and section is the most likely
    section (if section is more frequent than threshold value), if no section stands out, tag as '*UNSPECIFIC*'
    :param table: table of keywords
    :param threshold: to what percentage keyword needs to appear in one section, that this section overweights
    the others
    :param article_amount: amount of articles of full data set, used to calculate frequency of keywords
    :return: table of keywords where every keyword only appears once
    """
    keyword_table = pd.DataFrame([['keyword', 'name', 'value', 0, 'section']],
                                 columns=['keyword', 'name', 'value', 'counts', 'section'])
    for i, kw in enumerate(table.keyword.unique()):
        if i%100 == 0: print(str(i) + ' / 64537')

        entries = table[table.keyword == kw]
        entries_comb = entries.groupby(by=['keyword', 'section']).sum()
        max_count = entries_comb.max()[0]
        total_counts = entries_comb.sum()[0]
        if max_count >= threshold*total_counts:
            section = entries_comb.idxmax()[0][1]
            # idx = entries['counts'].idxmax()
            # section = table.loc[idx, 'section']
        else:
            section = '*UNSPECIFIC*'
        new_row = pd.DataFrame(data=  [[ kw,        kw[0],  kw[1],   total_counts, section]],
                               columns=['keyword', 'name', 'value', 'counts',     'section'])
        keyword_table = keyword_table.append(new_row)
        keyword_table['id'] = range(0, keyword_table.shape[0])
        keyword_table['prob'] = np.log(keyword_table.counts / article_amount)
    keyword_table = keyword_table[1:]

    # weight for how many edges we reduce later
    idf = np.log(article_amount / keyword_table.counts)
    keyword_table['idf'] = idf / max(idf)

    return keyword_table


def extr_keywords(field, table_keywords):
    """
    translate keywords structure as it comes from api to list of keywords ids (ids from table_keywords)
    :param field: article keywords as it comes from api
    :param table_keywords: table of keywords (created by create_keyword_table)
    :return: list of keyword ids
    """
    keyword_list = list()
    for keyword in field:
        try:
            id = table_keywords.id[
                (table_keywords.name == keyword['name']) &
                (table_keywords.value == keyword['value'])]._get_values(0)
            keyword_list.append(id)
        except IndexError:
            pass
    return keyword_list



def main():
    for year in ['2016', '2017', '2018']:
        # get and save articles
        for m in range(1,13):
            month = str(m)
            get_data(year, month)

        # load articles, clean them
        # concat dfs to df_year and then clean and translate keywords
        for m in range(1, 13):
            month = str(m)
            if len(month) == 1:
                month = '0' + month
            suffix = year + "_" + month
            print(suffix)

            with open(cwd + "/data/archive/articles_" + suffix + ".pickle", "rb") as f:
                df_new = pickle.load(f)

            if m == 1:
                df_year = df_new
            else:
                df_year = pd.concat([df_year, df_new], ignore_index=True)

        with open(cwd + "/data/archive/articles_" + year + ".pickle", "rb") as f:
            df_year = pickle.load(f)

        print(df_year.shape)
        df_year = clean_articles(df=df_year, word_count=20)
        df_year = clean_sections(df_year)
        # drop sections that are not interesting for keyword-analysis
        df_year = df_year[~(df_year['section'] == '*DELETE*')]
        print(df_year.shape)

        with open(cwd + "/data/archive/articles_" + year + "_clean.pickle", "wb") as f:
            pickle.dump(df_year, f)

        # create keyword table for one year
        table_year = create_keyword_table_partial(df_year)
        with open(cwd + "/data/table_keywords_partial_" + year + ".pickle", "wb") as f:
            pickle.dump(table_year, f)

    # combine keyword tables of singe years to full keyword table
    for i, year in enumerate(['2016', '2017', '2018']):

        with open(cwd + "/data/archive/articles_" + year + "_clean.pickle", "rb") as f:
            df_year = pickle.load(f)
        with open(cwd + "/data/table_keywords_partial_" + year + ".pickle", "rb") as f:
            table_year = pickle.load(f)

        if i == 0:
            table = table_year
            df = df_year
        else:
            table = pd.concat([table, table_year], ignore_index=True)
            df = pd.concat([df, df_year], ignore_index=True)
        print(df.shape, table.shape)

    with open(cwd + "/data/archive/df_16-18.pickle", "wb") as f:
        pickle.dump(df, f)
    with open(cwd + "/data/archive/df_16-18.pickle", "rb") as f:
        df = pickle.load(f)

    article_amount = df.shape[0]

    # combine keyword_tables from different years (counts, idf, major section)
    table_keywords = create_keyword_table(table, 0.35, article_amount)

    with open(cwd + "/data/table_keywords_16-18.pickle", "wb") as f:
        pickle.dump(table_keywords, f)

    # use this table to translate keyword to ids
    df['keywords'] = df.keywords.apply(lambda x: extr_keywords(x, table_keywords))
    with open(cwd + "/data/df_16-18.pickle", "wb") as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    main()
