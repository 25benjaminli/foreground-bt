import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CenterSpatialCropd,
    Resized,
    RandShiftIntensity,
    RandFlipd,
    RandZoomd,
    NormalizeIntensityd,
    MapTransform,
    CastToTyped,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
    Rand3DElasticd,
    Rand2DElasticd,
    BatchInverseTransform,
    Transform,
    RandomizableTrait,
    CropForegroundd,
)

from .datasets import TrainDataset

import json
import os
import torch

# load environmental variables
from dotenv import load_dotenv
load_dotenv(override=True)


class ConvertToMultiChannel(MapTransform):
    def __init__(self, keys, allow_missing_keys=False, use_softmax=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.use_softmax = use_softmax

    def __call__(self, data):
        # call it on label
        
        # label is 240x240x155, must be 3x240x240x155

        d = dict(data)
        if not self.use_softmax:
            result = [(d["label"] == 1), (d["label"] == 2), (d["label"] == 3)]

        else:
            result = [(d["label"] == 0), (d["label"] == 1), (d["label"] == 2), (d["label"] == 3)]
            
        d["label"] = torch.stack(result, dim=1).squeeze()
        # print("stacked label shape", d["label"].shape)

        return d
    
class ConvertToSingleChannel(MapTransform):
    def __init__(self, keys, allow_missing_keys=False, use_softmax=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        # call it on label
        
        # label is 3x240x240x155, must be 240x240x155

        d = dict(data)
        # d["label"] = d["label"].squeeze()
        d["label"] = torch.argmax(d["label"], dim=0)
        # binarize
        d["label"] = (d["label"] > 0).int()

        # unsqueeze again
        d["label"] = d["label"].unsqueeze(0)

        # print("label shape", d["label"].shape)

        # print("label shape", d["label"].shape)
        return d
    

class AddNameField(MapTransform):
    def __init__(self, keys, send_real_path=False, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.send_real_path = send_real_path

    def __call__(self, data):
        d = dict(data)

        if not "image_title" in d:
            # print("------- d image -----------", d["image"])
            # arbitrary select the first index
            # name = os.path.splitext(d["image"][0])[0].split("_")[-2] if "_" in d["image"][0] else os.path.splitext(d["image"][0])[0].replace('-seg', '')
            if not self.send_real_path:
                name = os.path.basename(os.path.dirname(d["image"][0]))
                # if it's brats normal, then just take the last part
                if "BraTS20" in name:
                    name = name.split("_")[-1] # volume num
                else:
                    name = str(int(name.split("-")[-2])).zfill(3) # refill with z because it has 5 digits for some reason originally
                d["image_title"] = name
            else:
                d["image_title"] = d["image"]
                d["label_title"] = d["label"]

        return d

class EmptyTransform(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        # super().__init__(data)
        super().__init__(keys, allow_missing_keys)
        self.keys = keys

    def __call__(self, data):
        return data

def datafold_read(dataset_path, fold_train,fold_val, key="training", cap=60, modalities = ['flair', 't1ce', 't1', 't2'], json_path='train.json'):
    with open(json_path) as f:
        json_data = json.load(f)


    json_data = json_data[key]
    npy_num = 0
    for d in json_data:
        for k in d:
            # print(d[k])
            if isinstance(d[k], list):
                # d[k] = [os.path.join(basedir, iv) for iv in d[k]]
                d[k] = [os.path.join(dataset_path, iv) for iv in d[k] if any(sub in iv for sub in modalities)] # only select the modalities
                # print(d[k])
                # if it's npy then add
                if 'npy' in d[k][0]:
                    npy_num += 1
            elif isinstance(d[k], str):
                d[k] = os.path.join(dataset_path, d[k]) if len(d[k]) > 0 else d[k]

    print("npy num", npy_num)
    tr = []
    val = []
    test = []

    for d in json_data:
        # print(d, fold_val)
        if "fold" in d and d["fold"] in fold_val:
            # remove fold key
            d.pop("fold")
            val.append(d)

        elif fold_train == "-1":
            d.pop("fold")
            tr.append(d) # everything else goes to training
        elif fold_train != "-1" and "fold" in d and d["fold"] in fold_train:
            d.pop("fold")
            tr.append(d)
        else:
            d.pop("fold")
            test.append(d)

    # print("sanity", val[:2])

    if cap is not None:
        return tr, val, test[:cap] # only first 60 for testing
    else:
        return tr, val, test


# TODO: train with multiple modalities AND multiple labels
def get_train_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]), # assuming loading multiple at the same time.
            ConvertToMultiChannel(keys="label", use_softmax=False),
            CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            ConvertToSingleChannel(keys="label", use_softmax=False),
            
            # insert single channel here if needed
            # ConvertToSingleChannel(keys="label", use_softmax=False) if not args.multichannel else EmptyTransform(keys=["image", "label"]),

            # # now, randomized stuff
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2), # !!
            
            # # !! Nearest exact for randzoomd. This already keeps the same size
            RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode="nearest-exact"),

            RandShiftIntensityd(keys=["image"], offsets=0.5, prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.3),
            ToTensord(keys=["image", "label"], track_meta=False)
        ]
    )

    # val_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image", "label"]), # assuming loading multiple at the same time.
    #         ConvertToMultiChannel(keys="label", use_softmax=False),
    #         CastToTyped(keys=["image", "label"], dtype=(torch.float16, torch.uint8)),
            
    #         # ConvertToSingleChannel(keys="label", use_softmax=False) if not args.multichannel else EmptyTransform(keys=["image", "label"]),

    #         ToTensord(keys=["image", "label"], track_meta=False),
    #     ]
    # )
    # ! TODO: change learning rate
    
    json_path = os.path.join(os.getenv("PROJECT_PATH"), "FSMSA", "train_preprocessed.json")
    dataset_path = os.getenv("PREPROCESSED_PATH")

    fold_train, fold_val = "-1", [0]
    train_files, validation_files, test_files = datafold_read(dataset_path=dataset_path, fold_val=fold_val, fold_train=fold_train,
                                                              modalities=["t2f", "t1c", "t1n"], json_path=json_path)


    print("length of train, validation files", len(train_files), len(validation_files))
    print("first train", train_files[0])
    # send to a yaml file
    file_paths = {
        "train": train_files,
        "val": validation_files,
        "test": test_files
    }

    # train_dataset = monai.data.Dataset(data=train_files, transform=train_transforms)
    # print("MULTIMODAL", args.multimodal, "MULTICHANNEL", args.multichannel)

    train_dataset = TrainDataset(args=args, files=train_files, transforms=train_transforms)
    train_loader = monai.data.DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=True)

    # ensure train loader is working
    # for i, data in enumerate(train_loader):
    #     print("train loader", i)
    #     print(data["image"].shape, data["label"].shape)
    #     break

    # val_dataset = TrainDataset(args=args, files=validation_files, transforms=val_transforms)
    # val_loader = monai.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # need to resolve training slow ness??
    # return train_loader, val_loader
    return train_loader
