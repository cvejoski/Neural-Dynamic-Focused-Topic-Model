import glob
import os
import pathlib
import re
import unicodedata
import click
import pandas as pd
import tqdm


@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Input directory of the raw data.")
@click.option('-o', '--output', type=click.Path(exists=False), required=True, help="Output directory of the pre-processed data.")
def convert(input: str, output: str):
    abstract_keywords = ['Abstract', 'ABSTRACT', 'ABSTRACT.', 'ABSTRACT:', '1. Abstract', 'Abstract:', 'Abstract.', 'abstract', 'abstract:', 'abstract.',
                         'SUMMARY',
                         'SUMMARY:', 'SUMMARY.', 'Summary', 'Summary.', 'Summary:', 'summary', 'summary.', 'summary:', 'A. Overview', '1. INTRODUCTION.',
                         '1. INTRODUCTION', '1. Introduction', 'Introduction', '1 Introduction', '1 Introduction and Background']

    articles = []
    for article in tqdm.tqdm(sorted(glob.glob(os.path.join(input, '*.txt')), key=os.path.basename), desc='Processing ACL articles'):
        file_name: str = os.path.split(article)[1]
        year_event, paper_id = os.path.splitext(file_name)[0].split('-')
        event_type, year = year_event[0], int(year_event[1:])
        year = int(year) + 2000 if 0 <= int(year) <= 6 else int(year) + 1900
        if year < 1973:
            continue
        text = pathlib.Path(article).read_text(encoding='windows-1252')
        text = re.sub(r'[^\x00-\x7F]', '', text)
        text = re.sub(r'\x14', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', text)
        text = re.sub(r'\(|[a-zA-Z][0-9]+\.\S+', '', text)
        text = re.sub(r'\d\.\d', '', text)
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(\x00|\\xx[0-9]{4})', ' ', text)
        text = re.sub(r',\n', ', ', text)
        text = re.sub(r'\n([a-z]|\W)', r'\1', text)
        text = re.sub(r'(\s\n|\t|-)', ' ', text)
        text = remove_non_ascii(text)
        text = re.sub('[0-9]+', '', text).splitlines()
        ix_abstract = find_abstract(text, abstract_keywords)
        is_abstract = ix_abstract > -1
        # print(f'file_name: {file_name}, event_type: {event_type}, year: {year}, paper_id: {paper_id}, abstract: {is_abstract}')
        try:
            if is_abstract:
                abstract = text[ix_abstract + 1]
            else:
                if len(text[0].split(' ')) < 8:
                    abstract = text[1]
                else:
                    abstract = text[0]
            doc = dict([('file_name', file_name), ('event_type', event_type), ('year', year),
                        ('paper_id', paper_id), ('abstract', abstract), ('text', "\n".join(text))])
            articles.append(doc)
        except:
            print(article)
    data = pd.DataFrame(articles)
    data.to_csv(os.path.join(output, 'acl-abstract-papers.csv'), index=False, encoding='utf-8')


def find_abstract(doc, abstract) -> int:
    for ab in abstract:
        for i, p in enumerate(doc):
            if ab == p:
                return i
    return -1


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in text.split(' '):
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ' '.join(new_words)


if __name__ == '__main__':
    convert()
