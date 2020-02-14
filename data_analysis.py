from collections import Counter

import matplotlib as plt

from io_util import read_file


def analyze_document(input_path):
    number_lists = read_file(input_path, preprocess=lambda x: [int(ele) for ele in x.strip().split()])
    counter = Counter()
    for ele in number_lists:
        counter.update(ele)
    a = 1
    values = list(counter.values())
    total = sum(values)
    partial = sum(values[:30])
    print(partial / total)
    new_counter = sorted(counter.items())[:30]
    keys, values = zip(*new_counter)
    probs = [ele * 100.0 / total for ele in values]
    cdf = [0 for _ in range(len(probs))]
    for idx, ele in enumerate(probs):
        if idx > 0:
            cdf[idx] = cdf[idx - 1] + ele
        else:
            cdf[idx] = ele

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.plot(keys, probs)
    ax1.set(xlabel='index of sentences', ylabel='probability',
            title='Fig 1 Probability Distribution of A sentence that is closest to sentences in summary')
    ax2.plot(keys, cdf)
    ax2.set(xlabel='index of sentences', ylabel='probability',
            title='Fig 2 Cumulative Distribution of A sentence that is closest to sentences in summary')

    plt.show()
