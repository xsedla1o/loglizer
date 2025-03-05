"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import os
import re
from collections import OrderedDict
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def cyclic_read(data: np.ndarray, samples: int, offset: int) -> Tuple[np.ndarray, int]:
    end = offset + samples
    if end <= data.shape[0]:
        return data[offset:end], end
    else:
        # join the end of the dataset to the beginning
        end = end - data.shape[0]
        if end > data.shape[0]:
            raise ValueError(f"The number of samples {samples} is too large "
                             f"for the dataset {data.shape[0]}")
        return np.append(data[offset:], data[:end], axis=0), end


def _split_data(x_data: np.ndarray, y_data: Optional[np.ndarray] = None,
                train_ratio=0.0, split_type="uniform",
                offset=0.0, val_ratio=0.01):
    if split_type == "uniform" and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == "sequential":
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    elif split_type == "sequential_validation":
        num_train = int(train_ratio * x_data.shape[0])
        num_validation = int(val_ratio * x_data.shape[0])
        num_test = x_data.shape[0] - num_train - num_validation

        train_begin = int(offset * x_data.shape[0])

        x_train, train_end = cyclic_read(x_data, num_train, train_begin)
        x_validation, validation_end = cyclic_read(x_data, num_validation, train_end)
        x_test, _ = cyclic_read(x_data, num_test, validation_end)

        if y_data is None:
            y_train = None
            y_validation = None
            y_test = None
        else:
            y_train, _ = cyclic_read(y_data, num_train, train_begin)
            y_validation, _ = cyclic_read(y_data, num_validation, train_end)
            y_test, _ = cyclic_read(y_data, num_test, validation_end)

        return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)
    elif split_type == "none":
        return x_data, y_data
    else:
        raise ValueError(f"Unknown split type {split_type}")
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)


def load_HDFS(
    log_file,
    label_file=None,
    window="session",
    train_ratio=0.5,
    split_type="sequential",
    save_csv=False,
    window_size=0,
        offset=0,
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print("====== Input data summary ======")

    if log_file.endswith(".npz"):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data["x_data"]
        y_data = data["y_data"]
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio,
                                                           split_type, offset)

    elif log_file.endswith(".seqs.csv"):
        data = pd.read_csv(
            log_file,
            engine="c",
            na_filter=False,
            memory_map=True,
            converters={"EventSequence": lambda x: [int(i) - 1 for i in x.split(" ")]},
        )
        x_data = data["EventSequence"].values
        y_data = data["Label"].values
        return _split_data(x_data, y_data, train_ratio, split_type, offset)

    elif log_file.endswith(".csv"):
        assert window == "session", "Only window=session is supported for HDFS dataset."
        print("Loading", log_file)
        struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r"(blk_-?\d+)", row["Content"])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row["EventId"])
        data_df = pd.DataFrame(
            list(data_dict.items()), columns=["BlockId", "EventSequence"]
        )

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(
                label_file, engine="c", na_filter=False, memory_map=True
            )
            label_data = label_data.set_index("BlockId")
            label_dict = label_data["Label"].to_dict()
            data_df["Label"] = data_df["BlockId"].apply(
                lambda x: 1 if label_dict[x] == "Anomaly" else 0
            )

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(
                data_df["EventSequence"].values, data_df["Label"].values, train_ratio,
                split_type, offset)

            print(y_train.sum(), y_test.sum())

        if save_csv:
            data_df.to_csv("data_instances.csv", index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(
                log.format(
                    "Train:",
                    x_train.shape[0],
                    y_train.sum(),
                    y_train.shape[0],
                    (1 - y_train).sum(),
                    y_train.shape[0],
                )
            )
            print(
                log.format(
                    "Test:",
                    x_test.shape[0],
                    y_test.sum(),
                    y_test.shape[0],
                    (1 - y_test).sum(),
                    y_test.shape[0],
                )
            )
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == "uniform":
                split_type = "sequential"
                print(
                    "Warning: Only split_type=sequential is supported \
                if label_file=None.".format(split_type)
                )
            # Split training and validation set sequentially
            x_data = data_df["EventSequence"].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio,
                                                    split_type=split_type,
                                                    offset=offset)
            print(
                "Total: {} instances, train: {} instances, test: {} instances".format(
                    x_data.shape[0], x_train.shape[0], x_test.shape[0]
                )
            )
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError("load_HDFS() only support csv and npz files!")

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print(
        "Total: {} instances, {} anomaly, {} normal".format(
            num_total, num_pos, num_total - num_pos
        )
    )
    print(
        "Train: {} instances, {} anomaly, {} normal".format(
            num_train, num_train_pos, num_train - num_train_pos
        )
    )
    print(
        "Test: {} instances, {} anomaly, {} normal\n".format(
            num_test, num_test_pos, num_test - num_test_pos
        )
    )

    return (x_train, y_train), (x_test, y_test)


def slice_hdfs(x, y, window_size):
    results_data = []
    print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i : i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i : i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(
        results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"]
    )
    print("Slicing done, {} windows generated".format(results_df.shape[0]))
    return (
        results_df[["SessionId", "EventSequence"]],
        results_df["Label"],
        results_df["SessionLabel"],
    )


def load_BGL(
    log_file,
    split_type="sequential",
    window="sliding",
    time_interval=60,
    stepping_size=60,
    train_ratio=0.8,
):
    """
    TBD, for now only preprocessed csv input is supported
    """
    if log_file.endswith(".seqs.csv"):
        data = pd.read_csv(
            log_file,
            engine="c",
            na_filter=False,
            memory_map=True,
            converters={"EventSequence": lambda x: [int(i) - 1 for i in x.split(" ")]},
        )
        x_data = data["EventSequence"].values
        y_data = data["Label"].values
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio,
                                                           split_type)
    else:
        raise NotImplementedError("currently supporting only .seqs.csv files!")

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print(
        f"Total: {num_total} instances, {num_pos} anomaly, {num_total - num_pos} normal"
    )
    print(
        f"Train: {num_train} instances, "
        f"{num_train_pos} anomaly, {num_train - num_train_pos} normal"
    )
    print(
        f"Test: {num_test} instances, "
        f"{num_test_pos} anomaly, {num_test - num_test_pos} normal\n"
    )

    return (x_train, y_train), (x_test, y_test)


def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para["save_path"]):
        os.mkdir(para["save_path"])
    log_size = raw_data.shape[0]
    sliding_file_path = (
        para["save_path"]
        + "sliding_"
        + str(para["window_size"])
        + "h_"
        + str(para["step_size"])
        + "h.csv"
    )

    # =============divide into sliding windows=========#
    start_end_index_list = []  # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:, 0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if cur_time < start_time + para["window_size"] * 3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para["step_size"] * 3600
            end_time = end_time + para["step_size"] * 3600
            for i in range(start_index, end_index):
                if time_data[i] < start_time:
                    i += 1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j += 1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print(
            "there are %d instances (sliding windows) in this dataset\n" % inst_number
        )
        np.savetxt(sliding_file_path, start_end_index_list, delimiter=",", fmt="%d")
    else:
        print("Loading start_end_index_list from file")
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print("there are %d instances (sliding windows) in this dataset" % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list = []
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    event_num = len(list(set(event_mapping_data)))
    print("There are %d log events" % event_num)

    # =============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number, event_num))
    for j in range(inst_number):
        label = 0  # 0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies" % sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels
