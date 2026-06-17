from weird_ai.dataset import LyricsDataset


def test_lyrics_dataset_getitem():
    tokens = [1, 2, 3, 4, 5]
    dataset = LyricsDataset(tokens, block_size=3)

    x, y = dataset[0]

    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]