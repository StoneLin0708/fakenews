
def markdown_pred_summary(x):
    return '|input|output|target|\n|-|-|-|\n' + '\n'.join(
        map(lambda i: f'|{i[0]}|{i[1]}|{i[2]}|', x)
    )