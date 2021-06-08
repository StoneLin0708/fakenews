import pandas as pd
import re
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--tag', action='store_true')
    return parser.parse_args()


def genrp(d, fmt):
    cstyle = '"border: 1px solid black;"'
    t = f"    <div style={cstyle}>{{}}</div>"
    r = ''
    for idx, (_, *res) in d.iterrows():
        r += '\n'.join(map(lambda i: t.format(fmt(i)), res))
    h = '\n'.join(map(lambda i: t.format(fmt(i)), d.columns[1:]))
    r = """<div style="border: 2px solid black; display:grid; grid-template-columns: {}">
{}
{}
</div>""".format("1fr "*len(d.columns[1:]), h, r)
    return r


def fmt_tag(txt):
    txt = re.sub(r'(<sep>|,|。)', r'\1<br>', txt)
    txt = re.sub(r'<(.)..([0-9]*)>',
                 r'<b style="color:lightgreen">\1\2</b>', txt)
    txt = re.sub(r'<en>', r'<b style="color:lightgreen">&lten&gt</b>', txt)
    return txt


if __name__ == '__main__':
    args = get_args()
    ffmt = fmt_tag if args.tag else lambda x: x
    open(args.output, 'w').write(genrp(pd.read_csv(args.input), ffmt))
