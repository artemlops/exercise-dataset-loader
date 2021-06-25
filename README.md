# Test task: Dataset loader

See task instructions [here](https://docs.google.com/document/d/14G2BOo-Q96EG4wYUBDFNWck1qj6Ur_qIHgPoxPneGoc/edit).

To run tests manually:
```
$ python -m venv venv/
$ make init
$ make test_unit
```

Usage:
```
>>>>>> ! cat ./data/my_dataset/touch/per_observation_timestamps.txt
;this file contains observations for 10 arbitrary moments
;of time in video, sorted chronologically.
;MILLISECOND OBSERVATION_ID
000000033 000000
000001066 000001
000002100 000002
000002833 000003
000004000 000004
000005000 000005
000006033 000006
000006366 000007
000006500 000008
000006600 000009
>>>
>>> from dataset_loader import MyDataset
>>> from torch.utils.data import DataLoader
>>>
>>> # Regular dataset:
>>> ds = MyDataset("./data/my_dataset")
>>> dl = DataLoader(ds)
>>> for i, elem in enumerate(dl):
...     # Note: timestamps are yielded exactly as they
...     # appear in 'per_observation_timestamps.txt':
...     print(i, elem.touch_timestamp_i)
0 tensor([33])
1 tensor([1066])
2 tensor([2100])
3 tensor([2833])
4 tensor([4000])
5 tensor([5000])
6 tensor([6033])
7 tensor([6366])
8 tensor([6500])
9 tensor([6600])
>>>
>>> # Alternative dataset:
>>> ds_alt = MyDataset("./data/my_dataset", linearize=True)
>>> dl_alt = DataLoader(ds_alt)
>>> for i, elem in enumerate(dl_alt):
...     # Note: timestamps are yielded as they
...     # appear in 'per_observation_timestamps.txt'
...     # with gaps filled with step 33:
...     print(i, elem.touch_timestamp_i)
...     if i == 10:
...         break
...
0 tensor([33])
1 tensor([66])
2 tensor([99])
3 tensor([132])
4 tensor([165])
5 tensor([198])
6 tensor([231])
7 tensor([264])
8 tensor([297])
9 tensor([330])
10 tensor([363])
```
Please find tests on dataset in `tests/unit/test_dataset_loader.py::TestMyDataset`, tests on generic algorithm of iterating over two sequences with or without linearization (method `dataset_loader.utils.zip_closest`) in `tests/unit/test_utils.py`.