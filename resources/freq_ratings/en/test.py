import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('all_words.txt', delimiter='\t')
    print(df[df['Word'] == 'day'])
