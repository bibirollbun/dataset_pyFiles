from IPython.display import Markdown, display
from typing import Callable, List, Dict, Optional
import matplotlib.pyplot as plt
import base64
from io import BytesIO


class DisplayCompetition:
    styles = {
        'h1': 'background-color:#eef; color:#006; padding:20px; border-radius:10px;',
        'h2': 'background-color:#fff; color:#006; padding:20px; border-bottom:solid 2px #006;',
        'h3': 'background-color:#fff; color:#006; padding-left:20px; clear:both;',
        'h4': 'background-color:#fff; color:#006; padding-left:20px; clear:both;',
        'p': '',
        '.tbl_selected': 'background-color:#fa9',
        '.column': 'padding:20px; margin:20px; box-shadow: 0 0 1em #006;',
    }

    def __init__(self):
        self.__in_section = False
        self.__html = ''

    def __getattr__(self, tag) -> Optional[Callable]:
        if tag in self.styles:
            def tag_generator(text):
                self._p(f'<{tag} style="{self.styles[tag]}">{text}</{tag}>')
            return tag_generator

    def fig(self, fig: plt.figure) -> None:
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        plt.close(fig)
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        self._p('<img src=\'data:image/png;base64,{}\'>'.format(encoded))

    def table(self, headings: List[str], data: Dict[str, List[str]], select: Optional[Callable] = None) -> None:
        if select is None:
            def select(x): return False

        html = '<table>'
        html += '<tr><td></td><th>' + '</th><th>'.join(headings) + '</th></tr>'

        for name, row in data.items():
            if select is None:
                html += '<tr><th>' + name + '</th>' + ''.join(f'<td>{item}</td>' for item in row) + '</tr>'
            else:
                html += '<tr><th>' + name + '</th>' + ''.join(f'<td style="' + (self.styles['.tbl_selected'] if select(item) else '') + f'">{item}</td>' for item in row) + '</tr>'
        html += '<table>'
        self._p(html)

    def table2level(self, headings: Dict[str, List[str]], row_headings: Dict[str, List[str]], data: List, select: Optional[Callable] = None) -> None:
        if select is None:
            def select(x): return False

        html = '<table>'

        # Headers
        h1line = ''
        h2line = ''
        for h1, h2s in headings.items():
            h1line += f'<th colspan="{len(h2s)}">{h1}</th>'
            h2line += '<th>' + '</th><th>'.join(h2s) + '</th>'
        html += f'<tr><td></td><td></td>{h1line}</tr>'
        html += f'<tr><td></td><td></td>{h2line}</tr>'

        # Row headers and data
        rows = ''
        rn = 0
        for h1, h2s in row_headings.items():
            for n, h2 in enumerate(h2s):
                rows += '<tr>'
                if n == 0:
                    rows += f'<th rowspan="{len(h2s)}">{h1}</th>'
                rows += f'<th>{h2}</th><td>' + '</td><td>'.join(data[rn]) + '</td></tr>'
                rn += 1
        html += rows
        html += '<table>'
        self._p(html)

    def ol(self, data: List[str]) -> None:
        html = '<ol>'
        if len(data):
            html += '<li>' + ('</li><li>'.join(data)) + '</li>'
        html += '</ol>'
        self._p(html)
        
    def ul(self, data: List[str]) -> None:
        html = '<ul>'
        if len(data):
            html += '<li>' + ('</li><li>'.join(data)) + '</li>'
        html += '</ul>'
        self._p(html)
        
    def start_section(self, style=None) -> None:
        self.__html = f'<div style="{self.styles[style]}">' if style in self.styles else '<div>'
        self.__in_section = True

    def stop_section(self) -> None:
        self.__in_section = False
        self.__html += '<div>'
        self._p()

    def _p(self, html: Optional[str] = None) -> None:
        if self.__in_section:
            self.__html += html
        elif html is None:
            self._p(self.__html)
            self.__html = ''
        else:
            display(Markdown(html))

dc = DisplayCompetition()


dc.h1('1. So, we have a time series prediction competition. Let\'s load the data and look on it')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 14})


TRAIN_CSV = '/kaggle/input/playground-series-s5e1/train.csv'
TEST_CSV = '/kaggle/input/playground-series-s5e1/test.csv'

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df.head()


dc.h1('2. Simple questions and answers')

dc.h2('2.1. What countries do we have in train and test dataframes? Do they differ?')
dc.p('In train we have countries: ' + ', '.join(train_df.country.unique()))
dc.p('In test we have countries: ' + ', '.join(test_df.country.unique()))
dc.p('<b>Conclusion:</b> we have 5 different countries , the same for train and test. Note, they are in completely different parts of the world, so they have different holidays, etc.')


dc.h2('2.2. What stores do we have in train and test dataframes? Do they differ?')
dc.p('In train we have stores: ' + ', '.join(train_df.store.unique()))
dc.p('In test we have stores: ' + ', '.join(test_df.store.unique()))
dc.p('<b>Conclusion:</b> we have 3 different stores, the same for train and test.')


dc.h2('2.3. What products do we have in train and test dataframes? Do they differ?')
dc.p('In train we have products: ' + ', '.join(train_df['product'].unique()))
dc.p('In test we have products: ' + ', '.join(test_df['product'].unique()))
dc.p('<b>Conclusion:</b> we have 5 different products, the same for train and test.')


dc.h2('2.4. Does every store work in every country?')
dc.p('In train we have contry/store pairs: <br/>' + ', '.join((train_df.country + '/' + train_df.store).unique()))
dc.p('In test we have contry/store pairs: <br/>' + ', '.join((test_df.country + '/' + test_df.store).unique()))
dc.p('<b>Conclusion:</b> yes, every store works in every country, and this is true both for train and test.')


dc.h2('2.5. Does every store in every country offer the same products?')
train_num = len((train_df.country + '/' + train_df.store + '/' +  train_df['product']).unique())
test_num = len((test_df.country + '/' + test_df.store + '/' +  test_df['product']).unique())
dc.p(f'We have {train_num} distinct country/store/product tripples in train.')
dc.p(f'We have {test_num} distinct country/store/product tripples in test.')
dc.p('<b>Conclusion:</b> As 6*3*5 = 90, yes, the assortiment of each store is the same!')


dc.h2('2.6. What time intervals do we have for each country/store pair and each dataframe?')
dc.h3('Train dataframe:')
dc._p(train_df.groupby(by=['country', 'store']).date.agg(('min', 'max')).to_html())
dc.h3('Test dataframe:')
dc._p(test_df.groupby(by=['country', 'store']).date.agg(('min', 'max')).to_html())
dc.p('<b>Conclusion:</b> Train starts at 2010-01-01 and ends at 2016-12-31, and test starts at 2017-01-01 and ends at 2019-12-31.')

dc.h2('2.7. What about NaNs in the train dataframe? What country/store/product combinations are missed?')
dc.h3('Train dataframe:')
df_nans = train_df.copy()
df_nans['num_nans'] = pd.isna(train_df.num_sold)
total_nans = df_nans.groupby(by=['country', 'store', 'product'])['num_nans'].sum().to_frame()
total = df_nans.groupby(by=['country', 'store', 'product'])['id'].count().rename('total').to_frame()
total = total_nans[total_nans > 0].join(total, on=['country', 'store', 'product'], how='left')
dc._p(pd.DataFrame(total[total.num_nans > 0]).to_html())

dc.p('<b>Conclusion:</b> Canada & Kenya / Discount Stickers / Holographic Goose are completely missed!')


product = 'Kaggle'
country = 'Italy'
store = 'Discount Stickers'

dc.h1('3. Time series comparision')

dc.h2(f'3.1. The same `country`="{country}" and `store`="{store}", but different products')
df = train_df[(train_df.country == country)&(train_df.store == store)]
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(24, 10)
sns.lineplot(data=df, x='date', y='num_sold', hue='product', ax=ax[0])
sns.lineplot(data=df[:365], x='date', y='num_sold', hue='product', ax=ax[1])
ax[0].set_title('All data')
ax[1].set_title('First year')
dc.fig(fig)

dc.h2(f'3.2. The same `product`="{product}" and `store`="{store}", but different countries')
df = train_df[(train_df['product'] == product)&(train_df.store == store)]
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(24, 10)
sns.lineplot(data=df, x='date', y='num_sold', hue='country', ax=ax[0])
sns.lineplot(data=df[:365], x='date', y='num_sold', hue='country', ax=ax[1])
ax[0].set_title('All data')
ax[1].set_title('First year')
dc.fig(fig)

dc.h2(f'3.3. The same `country`="{country}" and `product`="{product}", but different stores')
df = train_df[(train_df.country == country)&(train_df['product'] == product)]
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(24, 10)
sns.lineplot(data=df, x='date', y='num_sold', hue='store', ax=ax[0])
sns.lineplot(data=df[:365], x='date', y='num_sold', hue='store', ax=ax[1])
ax[0].set_title('All data')
ax[1].set_title('First year')
dc.fig(fig)

dc.p('<b>Conclusion:</b> we definitely see trends as well as cyclic patterns. So SARIMA-like models should be apropriate to describe this.')




